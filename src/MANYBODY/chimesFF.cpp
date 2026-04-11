/*
	ChIMES Calculator
	Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
	Contributing Author:  Rebecca K. Lindsey (2020)
*/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef CHIMES_LOOP_STYLE
#define CHIMES_LOOP_STYLE 3
#endif

using namespace std;

#include "chimesFF.h"

template <typename T> int get_index(const vector<T> &vec, const T &element)
{
  auto it = find(vec.begin(), vec.end(), element);

  if (it != vec.end()) {
    return distance(vec.begin(), it);
  } else {
    cout << "chimesFF: " << "ERROR: Could not find element in vector" << endl;
    exit(0);
  }
}

template <typename T>
int get_index_if(const vector<T> &vec, const T &element, vector<bool> &disqualified)
{

  if (disqualified.size() != vec.size()) {
    cout << "chimesFF: "
         << "ERROR: get_index_if(...): Qualification criteria does not match vector length" << endl;
    cout << "chimesFF: " << "vec.size(): " << vec.size() << endl;
    cout << "chimesFF: " << "disqualified.size(): " << disqualified.size() << endl;
    exit(0);
  }

  for (int i = 0; i < vec.size(); i++) {
    if ((vec[i] == element) && (!disqualified[i])) {
      disqualified[i] = true;
      return i;
    }
  }

  cout << "chimesFF: " << "ERROR: Could not find element in vector: " << element << endl;

  for (int i = 0; i < vec.size(); i++)
    cout << "chimesFF: " << "\t" << vec[i] << " " << disqualified[i] << endl;

  exit(0);
}

int chimesFF::get_proper_pair(string ty1, string ty2)
{

  for (int i = 0; i < pair_params_atm_chem_1.size(); i++) {
    if (ty1 == pair_params_atm_chem_1[i])
      if (ty2 == pair_params_atm_chem_2[i]) return i;

    if (ty2 == pair_params_atm_chem_1[i])
      if (ty1 == pair_params_atm_chem_2[i]) return i;
  }

  cout << "chimesFF: " << "ERROR: No proper pair name found for atom types" << ty1 << ", " << ty2
       << endl;
  exit(0);
}

chimesFF::chimesFF()
{
  natmtyps = 0;
  penalty_params.resize(2);

  // Dense coefficients treats all chimes parameters as potentially non-zero,
  // improving loop efficiency in some cases.
#ifdef CHIMES_DENSE_COEFFS
  dense_coeffs = true;
#else
  dense_coeffs = false;
#endif

  // Set defaults

  fcut_type = fcutType::CUBIC;

  penalty_params[0] = 0.01;
  penalty_params[1] = 1.0E4;

  inner_smooth_distance = 0.05;
  //inner_smooth_distance = 0.01 ;
}
chimesFF::~chimesFF() {}

void chimesFF::init(int mpi_rank)
{
  rank = mpi_rank;
  print_pretty_stuff();
}

void chimesFF::print_pretty_stuff()
{
  if (rank == 0) {
    cout << "chimesFF: " << endl;
    cout << "chimesFF: "
         << "01000011011010001001001010011010100010101010011 "
            "0100010101101110110011101101001011011101100101	 "
         << endl;
    cout << "chimesFF: " << endl;
    cout << "chimesFF: " << "	   _____  _		 _____	__	__	______	 _____	 ______		   _			  "
         << endl;
    cout << "chimesFF: "
         << "	  / ____|| |	|_	 _||  \\/  ||  ____| / ____| |	____|		   (_)			  " << endl;
    cout << "chimesFF: "
         << "	 | |	 | |__	  | |  | \\	 / || |__	| (___	 | |__	  _ __	  __ _	_  _ __	   ___	 "
            " "
         << endl;
    cout << "chimesFF: "
         << "	 | |	 | '_ \\   | |	| |\\/| ||	__|	  \\___ \\	|  __|	| '_ \\	 / _` || || '_ \\  "
            "/ _ \\ "
         << endl;
    cout << "chimesFF: "
         << "	 | |____ | | | | _| |_ | |	| || |____	____) | | |____ | | | || (_| || || | | ||  "
            "__/	  "
         << endl;
    cout << "chimesFF: "
         << "	  \\_____||_| |_||_____||_|	 |_||______||_____/	 |______||_| |_| \\__, ||_||_| |_| "
            "\\___|	  "
         << endl;
    cout << "chimesFF: " << "									 __/ |			  " << endl;
    cout << "chimesFF: " << "									|___/			  " << endl;
    cout << "chimesFF: " << endl;
    cout << "chimesFF: " << "			  Copyright (C) 2020 R.K. Lindsey, L.E. Fried, N. Goldman			  "
         << endl;
    cout << "chimesFF: " << endl;
    cout << "chimesFF: "
         << "01000011011010001001001010011010100010101010011 "
            "0100010101101110110011101101001011011101100101	  "
         << endl;
    cout << "chimesFF: " << endl;
  }
}

int chimesFF::split_line(string line, vector<string> &items)
{
  // Break a line up into tokens based on space separators.
  // Returns the number of tokens parsed.

  string contents;
  stringstream sstream;

  // Strip comments beginining with ! or ## and terminal new line

  int pos = line.find('!');

  if (pos != string::npos) line.erase(pos, line.length() - pos);

  pos = line.find("##");
  if (pos != string::npos) line.erase(pos, line.length() - pos);

  pos = line.find('\n');
  if (pos != string::npos) line.erase(pos, 1);

  sstream.str(line);

  items.clear();

  while (sstream >> contents) items.push_back(contents);

  return items.size();
}

string chimesFF::get_next_line(istream &str)
{
  // Read a line and return it, with error checking.

  string line;

  getline(str, line);

  if (!str.good()) {
    if (rank == 0) cout << "chimesFF: " << "Error reading line" << line << endl;
    exit(0);
  }

  return line;
}

void chimesFF::read_parameters(string paramfile)
{
  // Open the parameter file, run sanity checks

  ifstream param_file;
  param_file.open(paramfile.data());

  if (rank == 0) cout << "chimesFF: " << "Reading parameters from file: " << paramfile << endl;

  if (!param_file.is_open()) {
    if (rank == 0)
      cout << "chimesFF: " << "ERROR: Cannot open parameter file: " << paramfile << endl;
    exit(0);
  }

  // Declare parsing variables

  bool found_end = false;
  string line;
  string tmp_str;
  vector<string> tmp_str_items;
  int tmp_no_items;
  int tmp_int;
  int no_pairs;

  // Check that this is actually a chebyshev parameter set

  while (!found_end) {
    line = get_next_line(param_file);

    // Break out of loop

    if (line.find("ENDFILE") != string::npos) {
      if (rank == 0) {
        cout << "chimesFF: " << "ERROR: Could not find line containing: \" PAIRTYP: CHEBYSHEV\" "
             << endl;
        cout << "chimesFF: " << "	...Is this a ChIMES force field parameter file?" << endl;
      }
      exit(0);
    }

    if (line.find("PAIRTYP: CHEBYSHEV") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      if (tmp_no_items < 3) {
        if (rank == 0)
          cout << "chimesFF: "
               << "ERROR: \"PAIRTYP: CHEBYSHEV\" line must at least contain the 2-body order"
               << endl;
        exit(0);
      }

      poly_orders.push_back(stoi(tmp_str_items[2]));

      if (tmp_no_items >= 4) poly_orders.push_back(stoi(tmp_str_items[3]));

      if (tmp_no_items >= 5) poly_orders.push_back(stoi(tmp_str_items[4]));

      while (poly_orders.size() < 3) poly_orders.push_back(0);

      if (rank == 0) {
        cout << "chimesFF: " << "Using respective 2, 3, and 4-body orders of: " << poly_orders[0]
             << " " << poly_orders[1] << " " << poly_orders[2] << endl;

        cout << "chimesFF: " << "Note: Ignoring polynomial domain; assuming [-1,1]" << endl;
      }

      break;
    }
  }

  // If we've made it to here, then this should contain Chebyshev params. Rewind and start looking for general information

  param_file.seekg(0);

  found_end = false;

  while (!found_end) {
    line = get_next_line(param_file);

    if (line.find("ENDFILE") != string::npos) break;

    if (line.find("ATOM TYPES:") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      natmtyps = stoi(tmp_str_items[2]);

      if (rank == 0) cout << "chimesFF: " << "Will consider " << natmtyps << " atom types:" << endl;

      energy_offsets.resize(natmtyps);

      for (int i = 0; i < natmtyps; i++) energy_offsets[i] = 0.0;
    }

    if (line.find("# TYPEIDX #") != string::npos) {
      atmtyps.resize(natmtyps);
      masses.resize(natmtyps);
      for (int i = 0; i < natmtyps; i++) {
        line = get_next_line(param_file);
        split_line(line, tmp_str_items);
        atmtyps[i] = tmp_str_items[1];
        masses[i] = stod(tmp_str_items[3]);

        if (rank == 0) cout << "chimesFF: " << "\t" << i << " " << atmtyps[i] << endl;
      }
    }

    if (line.find("ATOM PAIRS:") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      no_pairs = stoi(tmp_str_items[2]);

      if (rank == 0)
        cout << "chimesFF: " << "Will consider " << no_pairs << " atom pair types" << endl;
    }

    if (line.find("# PAIRIDX #") != string::npos) {
      if (line.find("# USEOVRP #") != string::npos) continue;

      pair_params_atm_chem_1.resize(no_pairs);
      pair_params_atm_chem_2.resize(no_pairs);
      chimes_2b_cutoff.resize(no_pairs);
      morse_var.resize(no_pairs);

      ncoeffs_2b.resize(no_pairs);
      chimes_2b_pows.resize(no_pairs);
      chimes_2b_params.resize(no_pairs);
      chimes_2b_cutoff.resize(no_pairs);

      string tmp_xform_style;

      for (int i = 0; i < no_pairs; i++) {
        line = get_next_line(param_file);

        tmp_no_items = split_line(line, tmp_str_items);

        int pair_input_version = 0;

        if (tmp_no_items == 8) {
          if (rank == 0 && i == 0)
            cout << "chimesFF: Detected version 1 pair specification (with S_DELTA)\n";
          pair_input_version = 1;
        } else if (tmp_no_items == 7) {
          if (rank == 0 && i == 0)
            cout << "chimesFF: Detected version 2 pair specification (no S_DELTA)\n";
          pair_input_version = 2;
        } else {
          if (rank == 0) {
            cout << "Incorrect input in line: " << line << endl;
            cout << "Expect 7 or 8 entries\n";
          }
          exit(0);
        }

        pair_params_atm_chem_1[i] = tmp_str_items[1];
        pair_params_atm_chem_2[i] = tmp_str_items[2];

        if (rank == 0)
          cout << "chimesFF: " << "\t" << i << " " << pair_params_atm_chem_1[i] << " "
               << pair_params_atm_chem_2[i] << endl;

        chimes_2b_cutoff[i].push_back(stod(tmp_str_items[3]));    // Inner cutoff
        chimes_2b_cutoff[i].push_back(stod(tmp_str_items[4]));    // Outer cutoff

        int xform_style_idx, morse_idx;

        if (pair_input_version == 1) {
          xform_style_idx = 6;
          morse_idx = 7;
        } else if (pair_input_version == 2) {
          xform_style_idx = 5;
          morse_idx = 6;
        } else {
          if (rank == 0) cout << "Bad pair input version\n";
          exit(0);
        }

        if (i == 0) {
          tmp_xform_style = tmp_str_items[xform_style_idx];
        } else if (tmp_str_items[xform_style_idx] != tmp_xform_style) {
          if (rank == 0)
            cout << "chimesFF: "
                 << "Distance transformation style must be the same for all pair types" << endl;
          exit(0);
        }

        if (tmp_xform_style == "MORSE") {
          if (tmp_no_items > morse_idx)
            morse_var[i] = stod(tmp_str_items[morse_idx]);
          else {
            if (rank == 0)
              cout << "chimesFF: Missing morse lambda value in line: \n" << line << endl;
            exit(0);
          }
        }
      }

      xform_style = tmp_xform_style;

      if (rank == 0) cout << "chimesFF: " << "Read the following pair type information:" << endl;

      for (int i = 0; i < no_pairs; i++) {
        if (rank == 0)
          cout << "chimesFF: " << "\t" << pair_params_atm_chem_1[i] << " "
               << pair_params_atm_chem_2[i] << " r_cut_in: " << fixed << right << setprecision(5)
               << chimes_2b_cutoff[i][0] << " r_cut_out: " << chimes_2b_cutoff[i][1] << " "
               << xform_style;

        if (xform_style == "MORSE") {
          if (rank == 0) cout << " " << morse_var[i] << endl;
        } else if (rank == 0)
          cout << endl;
      }
    }

    if (line.find("FCUT TYPE:") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      if (tmp_str_items[2] == "CUBIC")
        fcut_type = fcutType::CUBIC;
      else if (tmp_str_items[2] == "TERSOFF")
        fcut_type = fcutType::TERSOFF;
      else {
        if (rank == 0) cout << "Error: unknown FCUT TYPE: " << tmp_str_items[2] << endl;
        exit(1);
      }

      if (rank == 0) cout << "chimesFF: " << "Will use cutoff style " << tmp_str_items[2] << endl;

      if (fcut_type == fcutType::TERSOFF) {
        fcut_var = stod(tmp_str_items[3]);

        if (rank == 0) cout << " " << fcut_var << endl;
      } else if (rank == 0)
        cout << endl;
    }

    if (line.find("PAIR CHEBYSHEV PENALTY DIST:") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      penalty_params[0] = stod(tmp_str_items[4]);

      if (rank == 0)
        cout << "chimesFF: " << "Will use penalty distance: " << penalty_params[0] << endl;
    }

    if (line.find("PAIR CHEBYSHEV PENALTY SCALING:") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      penalty_params[1] = stod(tmp_str_items[4]);

      if (rank == 0)
        cout << "chimesFF: " << "Will use penalty scaling: " << penalty_params[1] << endl;
    }

    if (line.find("NO ENERGY OFFSETS:") != string::npos) {
      int tmp_no = split_line(line, tmp_str_items);

      if (stoi(tmp_str_items[tmp_no - 1]) != natmtyps) {
        cout << "chimesFF: " << "ERROR: Number of energy offsets do not match number of atom types"
             << endl;
        exit(0);
      }

      // Expects atom offsets in the same order as atom types were provided originally

      if (rank == 0) cout << "chimesFF: " << "Will use single atom energy offsets: " << endl;

      int tmp_idx;

      for (int i = 0; i < natmtyps; i++) {
        line = get_next_line(param_file);
        split_line(line, tmp_str_items);
        tmp_idx = stoi(tmp_str_items[2]);

        energy_offsets[tmp_idx - 1] = stod(tmp_str_items[3]);

        if (rank == 0)
          cout << "chimesFF: " << "\t" << tmp_idx << " " << atmtyps[tmp_idx - 1] << " "
               << energy_offsets[tmp_idx - 1] << endl;
      }
    }
  }

  // Rewind and read the 2-body Chebyshev pair parameters

  param_file.seekg(0);

  found_end = false;

  while (!found_end) {
    line = get_next_line(param_file);

    if (line.find("ENDFILE") != string::npos) break;

    if (line.find("PAIRTYPE PARAMS:") != string::npos) {
      tmp_no_items = split_line(line, tmp_str_items);

      tmp_int = stoi(tmp_str_items[2]);

      if (rank == 0)
        cout << "chimesFF: " << "Read 2B parameters for pair: " << tmp_int << " "
             << tmp_str_items[3] << " " << tmp_str_items[4] << endl;

      line = get_next_line(param_file);

      split_line(line, tmp_str_items);    // Empty line

      ncoeffs_2b[tmp_int] = poly_orders[0];

      for (int i = 0; i < poly_orders[0]; i++) {
        line = get_next_line(param_file);
        split_line(line, tmp_str_items);

        chimes_2b_pows[tmp_int].push_back(stoi(tmp_str_items[0]));
        chimes_2b_params[tmp_int].push_back(stod(tmp_str_items[1]));

        if (rank == 0)
          cout << "chimesFF: " << "\t" << chimes_2b_pows[tmp_int][i] << " "
               << chimes_2b_params[tmp_int][i] << endl;
      }
    }

    if (line.find("PAIRMAPS:") != string::npos) {
      // Read the slow map and build the fast map

      tmp_no_items = split_line(line, tmp_str_items);

      n_pair_maps = stoi(tmp_str_items[1]);

      atom_typ_pair_map.resize(n_pair_maps);
      atom_idx_pair_map.resize(n_pair_maps);

      atom_int_prpr_map.resize(n_pair_maps);

      if (rank == 0)
        cout << "chimesFF: " << "Built the following 2-body pair \"slow\" map:" << endl;

      for (int i = 0; i < n_pair_maps; i++) {
        line = get_next_line(param_file);
        split_line(line, tmp_str_items);

        atom_idx_pair_map[i] = stoi(tmp_str_items[0]);
        atom_typ_pair_map[i] = tmp_str_items[1];

        if (rank == 0)
          cout << "chimesFF: " << "\t" << atom_idx_pair_map[i] << " " << atom_typ_pair_map[i]
               << "(i: " << i << ")" << endl;
      }

      if (rank == 0)
        cout << "chimesFF: " << "Built the following 2-body pair \"fast\" map:" << endl;

      atom_int_pair_map.resize((natmtyps - 1) * natmtyps + (natmtyps - 1) +
                               1);    // Maximum possible pair value + a small buffer

      for (int i = 0; i < natmtyps; i++) {
        for (int j = 0; j < natmtyps; j++) {
          // Get the pair type name for the set of atoms

          tmp_str = atmtyps[i] + atmtyps[j];

          tmp_int = get_index(atom_typ_pair_map, tmp_str);

          atom_int_pair_map[i * natmtyps + j] = atom_idx_pair_map[tmp_int];

          tmp_int = get_proper_pair(atmtyps[i], atmtyps[j]);

          atom_int_prpr_map[i * natmtyps + j] =
              pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

          if (rank == 0)
            cout << "chimesFF: " << "\t" << tmp_str << ": " << i * natmtyps + j << " "
                 << atom_int_pair_map[i * natmtyps + j] << endl;
        }
      }
    }
  }

  // Rewind and read the 3-body Chebyshev pair parameters

  if (poly_orders[1] > 0) {
    int ntrips;
    int tmp_idx;

    // Read parameters

    param_file.seekg(0);

    found_end = false;

    while (!found_end) {
      line = get_next_line(param_file);

      if (line.find("ENDFILE") != string::npos) break;

      if (line.find("ATOM PAIR TRIPLETS:") != string::npos) {
        split_line(line, tmp_str_items);

        ntrips = stoi(tmp_str_items[3]);

        ncoeffs_3b.resize(ntrips);
        chimes_3b_powers.resize(ntrips);
        chimes_3b_params.resize(ntrips);
        chimes_3b_cutoff.resize(ntrips);

        trip_params_atm_chems.resize(ntrips);
        trip_params_pair_typs.resize(ntrips);
      }

      if (line.find("TRIPLETTYPE PARAMS:") != string::npos) {
        vector<int> tmp_int_vec(3);

        line = get_next_line(param_file);

        split_line(line, tmp_str_items);

        tmp_int = stoi(tmp_str_items[1]);

        trip_params_atm_chems[tmp_int].push_back(tmp_str_items[3]);
        trip_params_atm_chems[tmp_int].push_back(tmp_str_items[4]);
        trip_params_atm_chems[tmp_int].push_back(tmp_str_items[5]);

        if (rank == 0)
          cout << "chimesFF: " << "Read 3B parameters for triplet: " << tmp_int << " "
               << trip_params_atm_chems[tmp_int][0] << " " << trip_params_atm_chems[tmp_int][1]
               << " " << trip_params_atm_chems[tmp_int][2] << endl;

        line = get_next_line(param_file);

        split_line(line, tmp_str_items);

        trip_params_pair_typs[tmp_int].push_back(tmp_str_items[1]);
        trip_params_pair_typs[tmp_int].push_back(tmp_str_items[2]);
        trip_params_pair_typs[tmp_int].push_back(tmp_str_items[3]);

        ncoeffs_3b[tmp_int] = stoi(tmp_str_items[7]);

        get_next_line(param_file);
        get_next_line(param_file);

        for (int i = 0; i < ncoeffs_3b[tmp_int]; i++) {
          line = get_next_line(param_file);
          split_line(line, tmp_str_items);

          tmp_int_vec[0] = stoi(tmp_str_items[1]);
          tmp_int_vec[1] = stoi(tmp_str_items[2]);
          tmp_int_vec[2] = stoi(tmp_str_items[3]);

          chimes_3b_powers[tmp_int].push_back(tmp_int_vec);
          chimes_3b_params[tmp_int].push_back(stod(tmp_str_items[6]));

          if (rank == 0)
            cout << "chimesFF: " << "\t" << chimes_3b_powers[tmp_int][i][0] << " "
                 << chimes_3b_powers[tmp_int][i][1] << " " << chimes_3b_powers[tmp_int][i][2] << " "
                 << chimes_3b_params[tmp_int][i] << endl;
        }
        if (dense_coeffs) {
          densify_3B(ncoeffs_3b[tmp_int], chimes_3b_powers[tmp_int], chimes_3b_params[tmp_int]);
        }
      }

      if (line.find("TRIPMAPS:") != string::npos) {
        split_line(line, tmp_str_items);

        n_trip_maps = stoi(tmp_str_items[1]);

        atom_idx_trip_map.resize(n_trip_maps);
        atom_typ_trip_map.resize(n_trip_maps);

        if (rank == 0)
          cout << "chimesFF: " << "Built the following 3-body pair \"slow\" map:" << endl;

        for (int i = 0; i < n_trip_maps; i++) {
          line = get_next_line(param_file);
          split_line(line, tmp_str_items);

          atom_idx_trip_map[i] = stoi(tmp_str_items[0]);
          atom_typ_trip_map[i] = tmp_str_items[1];

          if (rank == 0)
            cout << "chimesFF: " << "\t" << atom_idx_trip_map[i] << " " << atom_typ_trip_map[i]
                 << endl;
        }

        if (rank == 0)
          cout << "chimesFF: " << "Built the following 3-body pair \"fast\" map:" << endl;

        atom_int_trip_map.resize(natmtyps * natmtyps * natmtyps);

        for (int i = 0; i < natmtyps; i++) {
          for (int j = 0; j < natmtyps; j++) {
            for (int k = 0; k < natmtyps; k++) {
              // Get the trip type name for the set of atoms

              tmp_str = "";

              tmp_int = get_proper_pair(atmtyps[i], atmtyps[j]);
              tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

              tmp_int = get_proper_pair(atmtyps[i], atmtyps[k]);
              tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

              tmp_int = get_proper_pair(atmtyps[j], atmtyps[k]);
              tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

              tmp_int = get_index(atom_typ_trip_map, tmp_str);

              tmp_idx = i * natmtyps * natmtyps + j * natmtyps + k;

              atom_int_trip_map[tmp_idx] = atom_idx_trip_map[tmp_int];

              if (rank == 0)
                cout << "chimesFF: " << "\t" << tmp_idx << " " << atom_int_trip_map[tmp_idx]
                     << endl;
            }
          }
        }
      }
    }

    // Set up cutoffs ... First set to match 2-body, then read special if they exist

    int atmtyp_1, atmtyp_2, atmtyp_3;
    int pairtyp_1, pairtyp_2, pairtyp_3;

    for (int i = 0; i < ntrips; i++) {
      // Figure out the atom type index for each atom in the triplet type

      atmtyp_1 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), trip_params_atm_chems[i][0]));
      atmtyp_2 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), trip_params_atm_chems[i][1]));
      atmtyp_3 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), trip_params_atm_chems[i][2]));

      // Figure out the corresponding 2-body pair type

      pairtyp_1 = atom_int_pair_map[atmtyp_1 * natmtyps + atmtyp_2];
      pairtyp_2 = atom_int_pair_map[atmtyp_1 * natmtyps + atmtyp_3];
      pairtyp_3 = atom_int_pair_map[atmtyp_2 * natmtyps + atmtyp_3];

      // Set the default inner/outer cutoffs to the corresponding 2-body value

      chimes_3b_cutoff[i].resize(2);

      chimes_3b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_1][0]);
      chimes_3b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_2][0]);
      chimes_3b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_3][0]);

      chimes_3b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_1][1]);
      chimes_3b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_2][1]);
      chimes_3b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_3][1]);
    }

    param_file.seekg(0);

    int nentries;
    double cutval;

    found_end = false;

    while (!found_end) {
      line = get_next_line(param_file);

      if (line.find("ENDFILE") != string::npos) break;

      if (line.find("SPECIAL 3B S_MAXIM:") != string::npos) {
        split_line(line, tmp_str_items);

        if (rank == 0)
          cout << "chimesFF: " << "Set the following special 3-body outer cutoffs: " << endl;

        if (tmp_str_items[3] == "ALL") {
          cutval = stod(tmp_str_items[4]);

          for (int i = 0; i < ntrips; i++) {
            chimes_3b_cutoff[i][1][0] = cutval;
            chimes_3b_cutoff[i][1][1] = cutval;
            chimes_3b_cutoff[i][1][2] = cutval;
          }
        } else {
          nentries = stoi(tmp_str_items[4]);

          vector<string> pair_name(3);
          vector<double> cutoffval(3);

          for (int i = 0; i < nentries; i++) {
            line = get_next_line(param_file);

            split_line(line, tmp_str_items);

            tmp_int = atom_idx_trip_map[distance(
                atom_typ_trip_map.begin(),
                find(atom_typ_trip_map.begin(), atom_typ_trip_map.end(), tmp_str_items[0]))];

            pair_name[0] = tmp_str_items[1];
            pair_name[1] = tmp_str_items[2];
            pair_name[2] = tmp_str_items[3];

            cutoffval[0] = stod(tmp_str_items[4]);
            cutoffval[1] = stod(tmp_str_items[5]);
            cutoffval[2] = stod(tmp_str_items[6]);

            vector<bool> disqualified(3, false);

            chimes_3b_cutoff[tmp_int][1][get_index_if(trip_params_pair_typs[tmp_int], pair_name[0],
                                                      disqualified)] = cutoffval[0];
            chimes_3b_cutoff[tmp_int][1][get_index_if(trip_params_pair_typs[tmp_int], pair_name[1],
                                                      disqualified)] = cutoffval[1];
            chimes_3b_cutoff[tmp_int][1][get_index_if(trip_params_pair_typs[tmp_int], pair_name[2],
                                                      disqualified)] = cutoffval[2];
          }
        }

        for (int i = 0; i < ntrips; i++)
          if (rank == 0)
            cout << "chimesFF: " << "\t" << i << " " << chimes_3b_cutoff[i][1][0] << " "
                 << chimes_3b_cutoff[i][1][1] << " " << chimes_3b_cutoff[i][1][2] << endl;
      }

      if (line.find("SPECIAL 3B S_MINIM:") != string::npos) {
        split_line(line, tmp_str_items);

        if (rank == 0)
          cout << "chimesFF: " << "Set the following special 3-body inner cutoffs: " << endl;

        if (tmp_str_items[3] == "ALL") {
          cutval = stod(tmp_str_items[4]);

          for (int i = 0; i < ntrips; i++) {
            chimes_3b_cutoff[i][0][0] = cutval;
            chimes_3b_cutoff[i][0][1] = cutval;
            chimes_3b_cutoff[i][0][2] = cutval;
          }
        } else {
          nentries = stoi(tmp_str_items[4]);

          vector<string> pair_name(3);
          vector<double> cutoffval(3);

          for (int i = 0; i < nentries; i++) {
            line = get_next_line(param_file);

            split_line(line, tmp_str_items);

            tmp_int = atom_idx_trip_map[distance(
                atom_typ_trip_map.begin(),
                find(atom_typ_trip_map.begin(), atom_typ_trip_map.end(), tmp_str_items[0]))];

            pair_name[0] = tmp_str_items[1];
            pair_name[1] = tmp_str_items[2];
            pair_name[2] = tmp_str_items[3];

            cutoffval[0] = stod(tmp_str_items[4]);
            cutoffval[1] = stod(tmp_str_items[5]);
            cutoffval[2] = stod(tmp_str_items[6]);

            vector<bool> disqualified(3, false);

            chimes_3b_cutoff[tmp_int][0][get_index_if(trip_params_pair_typs[tmp_int], pair_name[0],
                                                      disqualified)] = cutoffval[0];
            chimes_3b_cutoff[tmp_int][0][get_index_if(trip_params_pair_typs[tmp_int], pair_name[1],
                                                      disqualified)] = cutoffval[1];
            chimes_3b_cutoff[tmp_int][0][get_index_if(trip_params_pair_typs[tmp_int], pair_name[2],
                                                      disqualified)] = cutoffval[2];
          }
        }

        for (int i = 0; i < ntrips; i++)
          if (rank == 0)
            cout << "chimesFF: " << "\t" << i << " " << chimes_3b_cutoff[i][0][0] << " "
                 << chimes_3b_cutoff[i][0][1] << " " << chimes_3b_cutoff[i][0][2] << endl;
      }
    }
  }

  // Rewind and read the 4-body Chebyshev pair parameters

  if (poly_orders[2] > 0) {
    int nquads;
    int tmp_idx;

    // Read parameters

    param_file.seekg(0);

    found_end = false;

    while (!found_end) {
      line = get_next_line(param_file);

      if (line.find("ENDFILE") != string::npos) break;

      if (line.find("ATOM PAIR QUADRUPLETS:") != string::npos) {
        split_line(line, tmp_str_items);

        nquads = stoi(tmp_str_items[3]);

        ncoeffs_4b.resize(nquads);
        chimes_4b_powers.resize(nquads);
        chimes_4b_params.resize(nquads);
        chimes_4b_cutoff.resize(nquads);

        quad_params_atm_chems.resize(nquads);
        quad_params_pair_typs.resize(nquads);
      }

      if (line.find("QUADRUPLETYPE PARAMS:") != string::npos) {
        line = get_next_line(param_file);

        split_line(line, tmp_str_items);

        tmp_int = stoi(tmp_str_items[1]);

        quad_params_atm_chems[tmp_int].push_back(tmp_str_items[3]);
        quad_params_atm_chems[tmp_int].push_back(tmp_str_items[4]);
        quad_params_atm_chems[tmp_int].push_back(tmp_str_items[5]);
        quad_params_atm_chems[tmp_int].push_back(tmp_str_items[6]);

        if (rank == 0)
          cout << "chimesFF: " << "Read 4B parameters for quadruplets: " << tmp_int << " "
               << quad_params_atm_chems[tmp_int][0] << " " << quad_params_atm_chems[tmp_int][1]
               << " " << quad_params_atm_chems[tmp_int][2] << " "
               << quad_params_atm_chems[tmp_int][3] << endl;

        line = get_next_line(param_file);

        split_line(line, tmp_str_items);

        quad_params_pair_typs[tmp_int].push_back(tmp_str_items[1]);
        quad_params_pair_typs[tmp_int].push_back(tmp_str_items[2]);
        quad_params_pair_typs[tmp_int].push_back(tmp_str_items[3]);
        quad_params_pair_typs[tmp_int].push_back(tmp_str_items[4]);
        quad_params_pair_typs[tmp_int].push_back(tmp_str_items[5]);
        quad_params_pair_typs[tmp_int].push_back(tmp_str_items[6]);

        ncoeffs_4b[tmp_int] = stoi(tmp_str_items[10]);

        get_next_line(param_file);
        get_next_line(param_file);

        vector<int> tmp_int_vec(6);

        for (int i = 0; i < ncoeffs_4b[tmp_int]; i++) {
          line = get_next_line(param_file);
          split_line(line, tmp_str_items);

          tmp_int_vec[0] = stoi(tmp_str_items[1]);
          tmp_int_vec[1] = stoi(tmp_str_items[2]);
          tmp_int_vec[2] = stoi(tmp_str_items[3]);
          tmp_int_vec[3] = stoi(tmp_str_items[4]);
          tmp_int_vec[4] = stoi(tmp_str_items[5]);
          tmp_int_vec[5] = stoi(tmp_str_items[6]);

          chimes_4b_powers[tmp_int].push_back(tmp_int_vec);

          chimes_4b_params[tmp_int].push_back(stod(tmp_str_items[9]));

          if (rank == 0)
            cout << "chimesFF: " << "\t" << chimes_4b_powers[tmp_int][i][0] << " "
                 << chimes_4b_powers[tmp_int][i][1] << " " << chimes_4b_powers[tmp_int][i][2] << " "
                 << chimes_4b_powers[tmp_int][i][3] << " " << chimes_4b_powers[tmp_int][i][4] << " "
                 << chimes_4b_powers[tmp_int][i][5] << " " << chimes_4b_params[tmp_int][i] << endl;
        }
        //if (dense_coeffs) {
        //  densify_4B(ncoeffs_4b[tmp_int], chimes_4b_powers[tmp_int], chimes_4b_params[tmp_int]);
        //}
      }

      if (line.find("QUADMAPS:") != string::npos) {
        split_line(line, tmp_str_items);

        n_quad_maps = stoi(tmp_str_items[1]);

        atom_idx_quad_map.resize(n_quad_maps);
        atom_typ_quad_map.resize(n_quad_maps);

        if (rank == 0)
          cout << "chimesFF: " << "Built the following 4-body pair \"slow\" map:" << endl;

        for (int i = 0; i < n_quad_maps; i++) {
          line = get_next_line(param_file);
          split_line(line, tmp_str_items);

          atom_idx_quad_map[i] = stoi(tmp_str_items[0]);
          atom_typ_quad_map[i] = tmp_str_items[1];

          if (rank == 0)
            cout << "chimesFF: " << "\t" << atom_idx_quad_map[i] << " " << atom_typ_quad_map[i]
                 << endl;
        }

        if (rank == 0)
          cout << "chimesFF: " << "Built the following 4-body pair \"fast\" map:" << endl;

        atom_int_quad_map.resize(natmtyps * natmtyps * natmtyps * natmtyps);

        for (int i = 0; i < natmtyps; i++) {
          for (int j = 0; j < natmtyps; j++) {
            for (int k = 0; k < natmtyps; k++) {
              for (int l = 0; l < natmtyps; l++) {
                // Get the quad type name for the set of atoms

                tmp_str = "";

                tmp_int = get_proper_pair(atmtyps[i], atmtyps[j]);
                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                tmp_int = get_proper_pair(atmtyps[i], atmtyps[k]);
                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                tmp_int = get_proper_pair(atmtyps[i], atmtyps[l]);
                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                tmp_int = get_proper_pair(atmtyps[j], atmtyps[k]);
                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                tmp_int = get_proper_pair(atmtyps[j], atmtyps[l]);
                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                tmp_int = get_proper_pair(atmtyps[k], atmtyps[l]);
                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                tmp_int = get_index(atom_typ_quad_map, tmp_str);

                tmp_idx =
                    i * natmtyps * natmtyps * natmtyps + j * natmtyps * natmtyps + k * natmtyps + l;

                atom_int_quad_map[tmp_idx] = atom_idx_quad_map[tmp_int];

                if (rank == 0)
                  cout << "chimesFF: " << "\t" << tmp_idx << " " << atom_int_quad_map[tmp_idx]
                       << endl;
              }
            }
          }
        }
      }
    }

    // Set up cutoffs ... First set to match 2-body, then read special if they exist

    int atmtyp_1, atmtyp_2, atmtyp_3, atmtyp_4;
    int pairtyp_1, pairtyp_2, pairtyp_3, pairtyp_4, pairtyp_5, pairtyp_6;

    for (int i = 0; i < nquads; i++) {
      // Figure out the atom type index for each atom in the quadruplet type

      atmtyp_1 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][0]));
      atmtyp_2 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][1]));
      atmtyp_3 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][2]));
      atmtyp_4 = distance(atmtyps.begin(),
                          find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][3]));

      // Figure out the corresponding 2-body pair type

      pairtyp_1 = atom_int_pair_map[atmtyp_1 * natmtyps + atmtyp_2];
      pairtyp_2 = atom_int_pair_map[atmtyp_1 * natmtyps + atmtyp_3];
      pairtyp_3 = atom_int_pair_map[atmtyp_1 * natmtyps + atmtyp_4];
      pairtyp_4 = atom_int_pair_map[atmtyp_2 * natmtyps + atmtyp_3];
      pairtyp_5 = atom_int_pair_map[atmtyp_2 * natmtyps + atmtyp_4];
      pairtyp_6 = atom_int_pair_map[atmtyp_3 * natmtyps + atmtyp_4];

      // Set the default inner/outer cutoffs to the corresponding 2-body value

      chimes_4b_cutoff[i].resize(2);

      chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_1][0]);
      chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_2][0]);
      chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_3][0]);
      chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_4][0]);
      chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_5][0]);
      chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_6][0]);

      chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_1][1]);
      chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_2][1]);
      chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_3][1]);
      chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_4][1]);
      chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_5][1]);
      chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_6][1]);
    }

    param_file.seekg(0);

    int nentries;
    double cutval;

    found_end = false;

    while (!found_end) {
      line = get_next_line(param_file);

      if (line.find("ENDFILE") != string::npos) break;

      if (line.find("SPECIAL 4B S_MAXIM:") != string::npos) {
        split_line(line, tmp_str_items);

        if (rank == 0)
          cout << "chimesFF: " << "Set the following special 4-body outer cutoffs: " << endl;

        if (tmp_str_items[3] == "ALL") {
          cutval = stod(tmp_str_items[4]);

          for (int i = 0; i < nquads; i++) {
            chimes_4b_cutoff[i][1][0] = cutval;
            chimes_4b_cutoff[i][1][1] = cutval;
            chimes_4b_cutoff[i][1][2] = cutval;
            chimes_4b_cutoff[i][1][3] = cutval;
            chimes_4b_cutoff[i][1][4] = cutval;
            chimes_4b_cutoff[i][1][5] = cutval;
          }
        } else {
          nentries = stoi(tmp_str_items[4]);

          vector<string> pair_name(6);
          vector<double> cutoffval(6);

          for (int i = 0; i < nentries; i++) {
            line = get_next_line(param_file);

            split_line(line, tmp_str_items);

            tmp_int = atom_idx_quad_map[distance(
                atom_typ_quad_map.begin(),
                find(atom_typ_quad_map.begin(), atom_typ_quad_map.end(), tmp_str_items[0]))];

            pair_name[0] = tmp_str_items[1];
            pair_name[1] = tmp_str_items[2];
            pair_name[2] = tmp_str_items[3];
            pair_name[3] = tmp_str_items[4];
            pair_name[4] = tmp_str_items[5];
            pair_name[5] = tmp_str_items[6];

            cutoffval[0] = stod(tmp_str_items[7]);
            cutoffval[1] = stod(tmp_str_items[8]);
            cutoffval[2] = stod(tmp_str_items[9]);
            cutoffval[3] = stod(tmp_str_items[10]);
            cutoffval[4] = stod(tmp_str_items[11]);
            cutoffval[5] = stod(tmp_str_items[12]);

            vector<bool> disqualified(6, false);

            chimes_4b_cutoff[tmp_int][1][get_index_if(quad_params_pair_typs[tmp_int], pair_name[0],
                                                      disqualified)] = cutoffval[0];
            chimes_4b_cutoff[tmp_int][1][get_index_if(quad_params_pair_typs[tmp_int], pair_name[1],
                                                      disqualified)] = cutoffval[1];
            chimes_4b_cutoff[tmp_int][1][get_index_if(quad_params_pair_typs[tmp_int], pair_name[2],
                                                      disqualified)] = cutoffval[2];
            chimes_4b_cutoff[tmp_int][1][get_index_if(quad_params_pair_typs[tmp_int], pair_name[3],
                                                      disqualified)] = cutoffval[3];
            chimes_4b_cutoff[tmp_int][1][get_index_if(quad_params_pair_typs[tmp_int], pair_name[4],
                                                      disqualified)] = cutoffval[4];
            chimes_4b_cutoff[tmp_int][1][get_index_if(quad_params_pair_typs[tmp_int], pair_name[5],
                                                      disqualified)] = cutoffval[5];
          }
        }

        for (int i = 0; i < nquads; i++) {
          if (rank == 0)
            cout << "chimesFF: " << "\t" << i << " " << chimes_4b_cutoff[i][1][0] << " "
                 << chimes_4b_cutoff[i][1][1] << " " << chimes_4b_cutoff[i][1][2] << " "
                 << chimes_4b_cutoff[i][1][3] << " " << chimes_4b_cutoff[i][1][4] << " "
                 << chimes_4b_cutoff[i][1][5] << endl;
        }
      }

      if (line.find("SPECIAL 4B S_MINIM:") != string::npos) {
        split_line(line, tmp_str_items);

        if (rank == 0)
          cout << "chimesFF: " << "Set the following special 4-body inner cutoffs: " << endl;

        if (tmp_str_items[3] == "ALL") {
          cutval = stod(tmp_str_items[4]);

          for (int i = 0; i < nquads; i++) {
            chimes_4b_cutoff[i][0][0] = cutval;
            chimes_4b_cutoff[i][0][1] = cutval;
            chimes_4b_cutoff[i][0][2] = cutval;
            chimes_4b_cutoff[i][0][3] = cutval;
            chimes_4b_cutoff[i][0][4] = cutval;
            chimes_4b_cutoff[i][0][5] = cutval;
          }
        } else {
          nentries = stoi(tmp_str_items[4]);

          vector<string> pair_name(6);
          vector<double> cutoffval(6);

          for (int i = 0; i < nquads; i++) {
            chimes_4b_cutoff[i][0][0] = -1.0;
            chimes_4b_cutoff[i][0][1] = -1.0;
            chimes_4b_cutoff[i][0][2] = -1.0;
            chimes_4b_cutoff[i][0][3] = -1.0;
            chimes_4b_cutoff[i][0][4] = -1.0;
            chimes_4b_cutoff[i][0][5] = -1.0;
          }

          for (int i = 0; i < nentries; i++) {
            line = get_next_line(param_file);

            split_line(line, tmp_str_items);

            tmp_int = atom_idx_quad_map[distance(
                atom_typ_quad_map.begin(),
                find(atom_typ_quad_map.begin(), atom_typ_quad_map.end(), tmp_str_items[0]))];

            pair_name[0] = tmp_str_items[1];
            pair_name[1] = tmp_str_items[2];
            pair_name[2] = tmp_str_items[3];
            pair_name[3] = tmp_str_items[4];
            pair_name[4] = tmp_str_items[5];
            pair_name[5] = tmp_str_items[6];

            cutoffval[0] = stod(tmp_str_items[7]);
            cutoffval[1] = stod(tmp_str_items[8]);
            cutoffval[2] = stod(tmp_str_items[9]);
            cutoffval[3] = stod(tmp_str_items[10]);
            cutoffval[4] = stod(tmp_str_items[11]);
            cutoffval[5] = stod(tmp_str_items[12]);

            vector<bool> disqualified(6, false);

            chimes_4b_cutoff[tmp_int][0][get_index_if(quad_params_pair_typs[tmp_int], pair_name[0],
                                                      disqualified)] = cutoffval[0];
            chimes_4b_cutoff[tmp_int][0][get_index_if(quad_params_pair_typs[tmp_int], pair_name[1],
                                                      disqualified)] = cutoffval[1];
            chimes_4b_cutoff[tmp_int][0][get_index_if(quad_params_pair_typs[tmp_int], pair_name[2],
                                                      disqualified)] = cutoffval[2];
            chimes_4b_cutoff[tmp_int][0][get_index_if(quad_params_pair_typs[tmp_int], pair_name[3],
                                                      disqualified)] = cutoffval[3];
            chimes_4b_cutoff[tmp_int][0][get_index_if(quad_params_pair_typs[tmp_int], pair_name[4],
                                                      disqualified)] = cutoffval[4];
            chimes_4b_cutoff[tmp_int][0][get_index_if(quad_params_pair_typs[tmp_int], pair_name[5],
                                                      disqualified)] = cutoffval[5];
          }
        }

        for (int i = 0; i < nquads; i++) {
          if (rank == 0)
            cout << "chimesFF: " << "\t" << i << " " << chimes_4b_cutoff[i][1][0] << " "
                 << chimes_4b_cutoff[i][1][1] << " " << chimes_4b_cutoff[i][1][2] << " "
                 << chimes_4b_cutoff[i][1][3] << " " << chimes_4b_cutoff[i][1][4] << " "
                 << chimes_4b_cutoff[i][1][5] << endl;
        }
      }
    }
  }

  param_file.close();
}

void chimesFF::set_polys_out_of_range(vector<double> &Tn, vector<double> &Tnd, double dx, double x,
                                      int poly_order, double inner_cutoff, double exprlen,
                                      double dx_dr)
//	Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd) when dx is < inner_cutoff.
//	Tnd is the derivative with respect to the interatomic distance, not the transformed distance (x).
//
//	The derivative Tnd is continuously set to zero inside the cutoff.
//	The exponential smoothing distance is set to chimesFF::inner_smooth_distance.
//	x, exprlen, and dx_dr are evaluated at the inner cutoff.
//
//	dx is the pair distance, which is assumed to be less than inner_cutoff.
{
  Tn[0] = 1.0;
  Tn[1] = x;

  // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

  Tnd[0] = 1.0;
  Tnd[1] = 2.0 * x;

  // Use recursion to set up the higher n-value Tn and Tnd's
  for (int i = 2; i <= poly_order; i++) {
    Tn[i] = 2.0 * x * Tn[i - 1] - Tn[i - 2];
    Tnd[i] = 2.0 * x * Tnd[i - 1] - Tnd[i - 2];
  }

  // Now multiply by n to convert Tnd's to actual derivatives of Tn

  for (int i = poly_order; i >= 1; i--) Tnd[i] = i * dx_dr * Tnd[i - 1];

  Tnd[0] = 0.0;

  // Exponential damping of the derivative.
  double damp_fac = exp((dx - inner_cutoff) / inner_smooth_distance);

  // Correct Tn outside of the range using the damping factor.
  for (int i = 0; i <= poly_order; i++) {
    Tn[i] += inner_smooth_distance * (damp_fac - 1.0) * Tnd[i];
    Tnd[i] *= damp_fac;
  }
}

void chimesFF::compute_1B(const int typ_idx, double &energy)
{
  // Compute 1b (input: a single atom type index... outputs (updates) energy

  energy += energy_offsets[typ_idx];
}

void chimesFF::compute_2B(const double dx, const vector<double> &dr, const vector<int> typ_idxs,
                          vector<double> &force, vector<double> &stress, double &energy,
                          chimes2BTmp &tmp)
{
  // Compute 2b (input: 2 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx: Scalar (pair distance)
  // dr: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]  *note
  // Energy: Scalar; energy for interaction set
  // Tmp: Temporary storage for calculation.

  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force is a packed array of coordinates.

  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  int pair_idx;
  double fcut;
  double fcutderiv;

  // tmp.resize(poly_orders[0]+1) ;

  // Use references for readability.
  vector<double> &Tn = tmp.Tn;
  vector<double> &Tnd = tmp.Tnd;

  pair_idx = atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[1]];

  if (dx >= chimes_2b_cutoff[pair_idx][1]) return;

  set_cheby_polys(Tn, Tnd, dx, pair_idx, chimes_2b_cutoff[pair_idx][0],
                  chimes_2b_cutoff[pair_idx][1], 0);

  get_fcut(dx, chimes_2b_cutoff[pair_idx][1], fcut, fcutderiv);

  double poly, dpoly_dx;

  poly_2B(&poly, &dpoly_dx, ncoeffs_2b[pair_idx], chimes_2b_params[pair_idx],
          chimes_2b_pows[pair_idx], Tn, Tnd);

  double dx_inv = (dx > 0.0) ? 1.0 / dx : 1e20;

  energy += poly * fcut;
  double force_scalar = (fcut * dpoly_dx + fcutderiv * poly) / dx;

  force[0 * CHDIM + 0] += force_scalar * dr[0];
  force[0 * CHDIM + 1] += force_scalar * dr[1];
  force[0 * CHDIM + 2] += force_scalar * dr[2];

  force[1 * CHDIM + 0] -= force_scalar * dr[0];
  force[1 * CHDIM + 1] -= force_scalar * dr[1];
  force[1 * CHDIM + 2] -= force_scalar * dr[2];

  // xx xy xz yy yz zz
  // 0  1	 2	3  4  5

  // xx xy xz yx yy yz zx zy zz
  // 0  1	 2	3  4  5	 6	7  8
  // *		   *	   *

  stress[0] -= force_scalar * dr[0] * dr[0];    // xx tensor component
  stress[1] -= force_scalar * dr[0] * dr[1];    // xy tensor component
  stress[2] -= force_scalar * dr[0] * dr[2];    // xz tensor component
  stress[3] -= force_scalar * dr[1] * dr[1];    // yy tensor component
  stress[4] -= force_scalar * dr[1] * dr[2];    // yz tensor component
  stress[5] -= force_scalar * dr[2] * dr[2];    // zz tensor component

  double E_penalty = 0.0;
  get_penalty(dx, pair_idx, E_penalty, force_scalar);

  if (E_penalty > 0.0) {
    energy += E_penalty;

    force_scalar /= dx;

    // Note: force_scalar is negative (LEF) 7/30/21.
    force[0 * CHDIM + 0] += force_scalar * dr[0];
    force[0 * CHDIM + 1] += force_scalar * dr[1];
    force[0 * CHDIM + 2] += force_scalar * dr[2];

    force[1 * CHDIM + 0] -= force_scalar * dr[0];
    force[1 * CHDIM + 1] -= force_scalar * dr[1];
    force[1 * CHDIM + 2] -= force_scalar * dr[2];

    // Update stress according to penalty force. (LEF) 07/30/21
    stress[0] -= force_scalar * dr[0] * dr[0];    // xx tensor component
    stress[1] -= force_scalar * dr[0] * dr[1];    // xy tensor component
    stress[2] -= force_scalar * dr[0] * dr[2];    // xz tensor component
    stress[3] -= force_scalar * dr[1] * dr[1];    // yy tensor component
    stress[4] -= force_scalar * dr[1] * dr[2];    // yz tensor component
    stress[5] -= force_scalar * dr[2] * dr[2];    // zz tensor component
  }
}

inline double chimesFF::dr2_3B(const double *dr2, int i, int j, int k, int l)
// Access the dr2 distance tensor for a 3 body interaction.
{
  return (dr2[i * CHDIM * 3 * CHDIM + j * 3 * CHDIM + k * CHDIM + l]);
}

inline double chimesFF::dr2_4B(const double *dr2, int i, int j, int k, int l)
// Access the dr2 distance tensor for a 4 body interaction.
{
  return (dr2[i * CHDIM * 6 * CHDIM + j * 6 * CHDIM + k * CHDIM + l]);
}

inline void chimesFF::init_distance_tensor(double *dr2, const vector<double> &dr, int npairs)
{
  for (int i = 0; i < npairs; i++) {
    for (int j = 0; j < CHDIM; j++) {
      for (int k = 0; k < npairs; k++) {
        for (int l = 0; l < CHDIM; l++) {
          dr2[i * CHDIM * npairs * CHDIM + j * npairs * CHDIM + k * CHDIM + l] =
              dr[i * CHDIM + j] * dr[k * CHDIM + l];
        }
      }
    }
  }
}

void chimesFF::compute_3B(const vector<double> &dx, const vector<double> &dr,
                          const vector<int> &typ_idxs, vector<double> &force,
                          vector<double> &stress, double &energy, chimes3BTmp &tmp)
{
  // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx_ij: Scalar (pair distance)
  // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]
  // Energy: Scalar; energy for interaction set
  // Tmp: Temporary storage for 3-body interactions.

  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force and dr are packed vectors of coordinates.

  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  const int natoms = 3;                            // Number of atoms in an interaction set
  const int npairs = natoms * (natoms - 1) / 2;    // Number of pairs in an interaction set

  // tmp.resize(poly_orders[1]) ;

  vector<double> &Tn_ij = tmp.Tn_ij;
  vector<double> &Tn_ik = tmp.Tn_ik;
  vector<double> &Tn_jk = tmp.Tn_jk;    // The Chebyshev polymonials
  vector<double> &Tnd_ij = tmp.Tnd_ij;
  vector<double> &Tnd_ik = tmp.Tnd_ik;
  vector<double> &Tnd_jk = tmp.Tnd_jk;    // The Chebyshev polymonial derivatives

  // Avoid allocating std::vector quantities.  Heap memory allocation is slow on the GPU.
  // fixed-length C arrays are allocated on the stack.
  double fcut[npairs];
  double fcutderiv[npairs];
  double deriv[npairs];

#if DEBUG == 1
  if (dr.size() != 9) {
    cout << "Error: dr should have length = 9.  Current length = " << dr.size() << endl;
    exit(0);
  }
#endif

  int type_idx = typ_idxs[0] * natmtyps * natmtyps + typ_idxs[1] * natmtyps + typ_idxs[2];
  int tripidx = atom_int_trip_map[type_idx];

  if (tripidx < 0)    // Skipping an excluded interaction
    return;

  // Check whether cutoffs are within allowed ranges
  vector<int> &mapped_pair_idx = pair_int_trip_map[type_idx];

  if (dx[0] >= chimes_3b_cutoff[tripidx][1][mapped_pair_idx[0]])    // ij
    return;
  if (dx[1] >= chimes_3b_cutoff[tripidx][1][mapped_pair_idx[1]])    // ik
    return;
  if (dx[2] >= chimes_3b_cutoff[tripidx][1][mapped_pair_idx[2]])    // jk
    return;

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  // Set up the polynomials

  set_cheby_polys(Tn_ij, Tnd_ij, dx[0], atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[1]],
                  chimes_3b_cutoff[tripidx][0][mapped_pair_idx[0]],
                  chimes_3b_cutoff[tripidx][1][mapped_pair_idx[0]], 1);
  set_cheby_polys(Tn_ik, Tnd_ik, dx[1], atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[2]],
                  chimes_3b_cutoff[tripidx][0][mapped_pair_idx[1]],
                  chimes_3b_cutoff[tripidx][1][mapped_pair_idx[1]], 1);
  set_cheby_polys(Tn_jk, Tnd_jk, dx[2], atom_int_pair_map[typ_idxs[1] * natmtyps + typ_idxs[2]],
                  chimes_3b_cutoff[tripidx][0][mapped_pair_idx[2]],
                  chimes_3b_cutoff[tripidx][1][mapped_pair_idx[2]], 1);

  // Set up the smoothing functions

  get_fcut(dx[0], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[0]], fcut[0], fcutderiv[0]);
  get_fcut(dx[1], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[1]], fcut[1], fcutderiv[1]);
  get_fcut(dx[2], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[2]], fcut[2], fcutderiv[2]);
  double fcut_all = fcut[0] * fcut[1] * fcut[2];

  // Product of 2 fcuts divided by dx. Index i = product of all fcuts except i.
  double fcut_2[npairs];
  fcut_2[0] = fcut[1] * fcut[2] / dx[0];
  fcut_2[1] = fcut[0] * fcut[2] / dx[1];
  fcut_2[2] = fcut[0] * fcut[1] / dx[2];

  double poly, dpoly_dx[npairs];

  // Start the force/stress/energy calculation
  double coeff;
  int powers[npairs];
  double force_scalar[npairs];

  if (!dense_coeffs) {
    poly_3B(&poly, dpoly_dx, ncoeffs_3b[tripidx], chimes_3b_params[tripidx], mapped_pair_idx,
            chimes_3b_powers[tripidx], Tn_ij, Tn_ik, Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);
  } else {

    // JIT evaluation of the chebyshev polynomial and its derivatives
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++) { inv_mapped_pair[mapped_pair_idx[j]] = j; }

    vector<vector<double> *> Tn{npairs}, Tnd{npairs};

    for (int j = 0; j < npairs; j++) {
      switch (inv_mapped_pair[j]) {
        case 0:
          Tn[j] = &Tn_ij;
          Tnd[j] = &Tnd_ij;
          break;
        case 1:
          Tn[j] = &Tn_ik;
          Tnd[j] = &Tnd_ik;
          break;
        case 2:
          Tn[j] = &Tn_jk;
          Tnd[j] = &Tnd_jk;
          break;
        default:
          cout << "Bad inverse pair mapping found\n";
          exit(1);
      }
    }

    poly_3B_dense(poly, dpoly_dx[inv_mapped_pair[0]], dpoly_dx[inv_mapped_pair[1]],
                  dpoly_dx[inv_mapped_pair[2]], ncoeffs_3b[tripidx], chimes_3b_params[tripidx],
                  *Tn[0], *Tn[1], *Tn[2], *Tnd[0], *Tnd[1], *Tnd[2]);
  }

  energy += poly * fcut_all;

  force_scalar[0] = (fcut_all * dpoly_dx[0] + fcutderiv[0] * fcut[1] * fcut[2] * poly) / dx[0];
  force_scalar[1] = (fcut_all * dpoly_dx[1] + fcutderiv[1] * fcut[0] * fcut[2] * poly) / dx[1];
  force_scalar[2] = (fcut_all * dpoly_dx[2] + fcutderiv[2] * fcut[0] * fcut[1] * poly) / dx[2];

  // Accumulate forces/stresses on/from the ij pair

  force[0 * CHDIM + 0] += force_scalar[0] * dr[0 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[0] * dr[0 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[0] * dr[0 * CHDIM + 2];

  force[1 * CHDIM + 0] -= force_scalar[0] * dr[0 * CHDIM + 0];
  force[1 * CHDIM + 1] -= force_scalar[0] * dr[0 * CHDIM + 1];
  force[1 * CHDIM + 2] -= force_scalar[0] * dr[0 * CHDIM + 2];

  stress[0] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[0] * dr[0 * CHDIM + 2] * dr[0 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the ik pair

  force[0 * CHDIM + 0] += force_scalar[1] * dr[1 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[1] * dr[1 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[1] * dr[1 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[1] * dr[1 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[1] * dr[1 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[1] * dr[1 * CHDIM + 2];

  stress[0] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[1] * dr[1 * CHDIM + 2] * dr[1 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the jk pair

  force[1 * CHDIM + 0] += force_scalar[2] * dr[2 * CHDIM + 0];
  force[1 * CHDIM + 1] += force_scalar[2] * dr[2 * CHDIM + 1];
  force[1 * CHDIM + 2] += force_scalar[2] * dr[2 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[2] * dr[2 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[2] * dr[2 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[2] * dr[2 * CHDIM + 2];

  stress[0] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[2] * dr[2 * CHDIM + 2] * dr[2 * CHDIM + 2];    // zz tensor component

  return;
}

void chimesFF::compute_4B(const vector<double> &dx, const vector<double> &dr,
                          const vector<int> &typ_idxs, vector<double> &force,
                          vector<double> &stress, double &energy, chimes4BTmp &tmp)
{
  // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
  //
  // Input parameters:
  //
  // dx_ij: Scalar (pair distance)
  // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
  // Force: [natoms in interaction set][x,y, and z-component] *note
  // Stress [sxx, sxy, sxz, syy, syz, szz]
  // Energy: Scalar; energy for interaction set
  // Tmp: Structure containing temporary data.
  // Assumes atom indices start from zero
  // Assumes distances are atom_2 - atom_1
  //
  // *note: force and dr are packed vectors of coordinates.
  // Factored the Chebyshev polynomial and its derivatives from the cutoff function. (LEF 3/11/26)

  const int natoms = 4;                            // Number of atoms in an interaction set
  const int npairs = natoms * (natoms - 1) / 2;    // Number of pairs in an interaction set

  double fcut[npairs];
  double fcutderiv[npairs];
  double deriv[npairs];

#if DEBUG == 1
  if (force.size() != CHDIM * natoms) {
    cout << "Error: force vector had incorrect dimension of " << force.size() << endl;
    exit(1);
  }
#endif

  vector<double> &Tn_ij = tmp.Tn_ij;
  vector<double> &Tn_ik = tmp.Tn_ik;
  vector<double> &Tn_il = tmp.Tn_il;
  vector<double> &Tn_jk = tmp.Tn_jk;
  vector<double> &Tn_jl = tmp.Tn_jl;
  vector<double> &Tn_kl = tmp.Tn_kl;

  vector<double> &Tnd_ij = tmp.Tnd_ij;
  vector<double> &Tnd_ik = tmp.Tnd_ik;
  vector<double> &Tnd_il = tmp.Tnd_il;
  vector<double> &Tnd_jk = tmp.Tnd_jk;
  vector<double> &Tnd_jl = tmp.Tnd_jl;
  vector<double> &Tnd_kl = tmp.Tnd_kl;

  int idx = typ_idxs[0] * natmtyps * natmtyps * natmtyps + typ_idxs[1] * natmtyps * natmtyps +
      typ_idxs[2] * natmtyps + typ_idxs[3];

  int quadidx = atom_int_quad_map[idx];

  if (quadidx < 0)    // Skipping an excluded interaction
    return;

  vector<int> &mapped_pair_idx = pair_int_quad_map[idx];

  // Check whether cutoffs are within allowed ranges

  for (int i = 0; i < npairs; i++)
    if (dx[i] >= chimes_4b_cutoff[quadidx][1][mapped_pair_idx[i]]) return;

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  // Set up the polynomials

  set_cheby_polys(Tn_ij, Tnd_ij, dx[0], atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[1]],
                  chimes_4b_cutoff[quadidx][0][mapped_pair_idx[0]],
                  chimes_4b_cutoff[quadidx][1][mapped_pair_idx[0]], 2);

  set_cheby_polys(Tn_ik, Tnd_ik, dx[1], atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[2]],
                  chimes_4b_cutoff[quadidx][0][mapped_pair_idx[1]],
                  chimes_4b_cutoff[quadidx][1][mapped_pair_idx[1]], 2);

  set_cheby_polys(Tn_il, Tnd_il, dx[2], atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[3]],
                  chimes_4b_cutoff[quadidx][0][mapped_pair_idx[2]],
                  chimes_4b_cutoff[quadidx][1][mapped_pair_idx[2]], 2);

  set_cheby_polys(Tn_jk, Tnd_jk, dx[3], atom_int_pair_map[typ_idxs[1] * natmtyps + typ_idxs[2]],
                  chimes_4b_cutoff[quadidx][0][mapped_pair_idx[3]],
                  chimes_4b_cutoff[quadidx][1][mapped_pair_idx[3]], 2);

  set_cheby_polys(Tn_jl, Tnd_jl, dx[4], atom_int_pair_map[typ_idxs[1] * natmtyps + typ_idxs[3]],
                  chimes_4b_cutoff[quadidx][0][mapped_pair_idx[4]],
                  chimes_4b_cutoff[quadidx][1][mapped_pair_idx[4]], 2);

  set_cheby_polys(Tn_kl, Tnd_kl, dx[5], atom_int_pair_map[typ_idxs[2] * natmtyps + typ_idxs[3]],
                  chimes_4b_cutoff[quadidx][0][mapped_pair_idx[5]],
                  chimes_4b_cutoff[quadidx][1][mapped_pair_idx[5]], 2);

  // Set up the smoothing functions
  for (int i = 0; i < npairs; i++)
    get_fcut(dx[i], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[i]], fcut[i], fcutderiv[i]);

  // Product of all 6 fcuts.
  double fcut_all = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5];

  // Product of 5 fcuts
  double fcut_5[npairs];
  fcut_5[0] = fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[1] = fcut[0] * fcut[2] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[2] = fcut[0] * fcut[1] * fcut[3] * fcut[4] * fcut[5];
  fcut_5[3] = fcut[0] * fcut[1] * fcut[2] * fcut[4] * fcut[5];
  fcut_5[4] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[5];
  fcut_5[5] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4];

  double poly, dpoly_dx[npairs];

  if (!dense_coeffs) {
    poly_4B(&poly, dpoly_dx, ncoeffs_4b[quadidx], chimes_4b_params[quadidx], mapped_pair_idx,
            chimes_4b_powers[quadidx], Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik,
            Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else {
    // Dense evaluation of the chebyshev polynomial and its derivatives
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++) { inv_mapped_pair[mapped_pair_idx[j]] = j; }

    vector<vector<double> *> Tn{npairs}, Tnd{npairs};

    for (int j = 0; j < npairs; j++) {
      switch (inv_mapped_pair[j]) {
        case 0:
          Tn[j] = &Tn_ij;
          Tnd[j] = &Tnd_ij;
          break;
        case 1:
          Tn[j] = &Tn_ik;
          Tnd[j] = &Tnd_ik;
          break;
        case 2:
          Tn[j] = &Tn_il;
          Tnd[j] = &Tnd_il;
          break;
        case 3:
          Tn[j] = &Tn_jk;
          Tnd[j] = &Tnd_jk;
          break;
        case 4:
          Tn[j] = &Tn_jl;
          Tnd[j] = &Tnd_jl;
          break;
        case 5:
          Tn[j] = &Tn_kl;
          Tnd[j] = &Tnd_kl;
          break;
        default:
          cout << "Bad inverse pair mapping found\n";
          exit(1);
      }
    }

    poly_4B_dense(poly, dpoly_dx[inv_mapped_pair[0]], dpoly_dx[inv_mapped_pair[1]],
                  dpoly_dx[inv_mapped_pair[2]], dpoly_dx[inv_mapped_pair[3]],
                  dpoly_dx[inv_mapped_pair[4]], dpoly_dx[inv_mapped_pair[5]], ncoeffs_4b[quadidx],
                  chimes_4b_params[quadidx], *Tn[0], *Tn[1], *Tn[2], *Tn[3], *Tn[4], *Tn[5],
                  *Tnd[0], *Tnd[1], *Tnd[2], *Tnd[3], *Tnd[4], *Tnd[5]);
  }

  energy += poly * fcut_all;

  double force_scalar[npairs];
  for (int j = 0; j < npairs; j++) {
    force_scalar[j] = (fcut_all * dpoly_dx[j] + fcutderiv[j] * fcut_5[j] * poly) / dx[j];
  }

  // Accumulate forces/stresses on/from the ij pair

  force[0 * CHDIM + 0] += force_scalar[0] * dr[0 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[0] * dr[0 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[0] * dr[0 * CHDIM + 2];

  force[1 * CHDIM + 0] -= force_scalar[0] * dr[0 * CHDIM + 0];
  force[1 * CHDIM + 1] -= force_scalar[0] * dr[0 * CHDIM + 1];
  force[1 * CHDIM + 2] -= force_scalar[0] * dr[0 * CHDIM + 2];

  stress[0] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[0] * dr[0 * CHDIM + 2] * dr[0 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the ik pair

  force[0 * CHDIM + 0] += force_scalar[1] * dr[1 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[1] * dr[1 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[1] * dr[1 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[1] * dr[1 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[1] * dr[1 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[1] * dr[1 * CHDIM + 2];

  stress[0] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[1] * dr[1 * CHDIM + 2] * dr[1 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the il pair

  force[0 * CHDIM + 0] += force_scalar[2] * dr[2 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[2] * dr[2 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[2] * dr[2 * CHDIM + 2];

  force[3 * CHDIM + 0] -= force_scalar[2] * dr[2 * CHDIM + 0];
  force[3 * CHDIM + 1] -= force_scalar[2] * dr[2 * CHDIM + 1];
  force[3 * CHDIM + 2] -= force_scalar[2] * dr[2 * CHDIM + 2];

  stress[0] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[2] * dr[2 * CHDIM + 2] * dr[2 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the jk pair

  force[1 * CHDIM + 0] += force_scalar[3] * dr[3 * CHDIM + 0];
  force[1 * CHDIM + 1] += force_scalar[3] * dr[3 * CHDIM + 1];
  force[1 * CHDIM + 2] += force_scalar[3] * dr[3 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[3] * dr[3 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[3] * dr[3 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[3] * dr[3 * CHDIM + 2];

  stress[0] -= force_scalar[3] * dr[3 * CHDIM + 0] * dr[3 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[3] * dr[3 * CHDIM + 0] * dr[3 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[3] * dr[3 * CHDIM + 0] * dr[3 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[3] * dr[3 * CHDIM + 1] * dr[3 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[3] * dr[3 * CHDIM + 1] * dr[3 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[3] * dr[3 * CHDIM + 2] * dr[3 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the jl pair

  force[1 * CHDIM + 0] += force_scalar[4] * dr[4 * CHDIM + 0];
  force[1 * CHDIM + 1] += force_scalar[4] * dr[4 * CHDIM + 1];
  force[1 * CHDIM + 2] += force_scalar[4] * dr[4 * CHDIM + 2];

  force[3 * CHDIM + 0] -= force_scalar[4] * dr[4 * CHDIM + 0];
  force[3 * CHDIM + 1] -= force_scalar[4] * dr[4 * CHDIM + 1];
  force[3 * CHDIM + 2] -= force_scalar[4] * dr[4 * CHDIM + 2];

  stress[0] -= force_scalar[4] * dr[4 * CHDIM + 0] * dr[4 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[4] * dr[4 * CHDIM + 0] * dr[4 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[4] * dr[4 * CHDIM + 0] * dr[4 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[4] * dr[4 * CHDIM + 1] * dr[4 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[4] * dr[4 * CHDIM + 1] * dr[4 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[4] * dr[4 * CHDIM + 2] * dr[4 * CHDIM + 2];    // zz tensor component

  // Accumulate forces/stresses on/from the kl pair

  force[2 * CHDIM + 0] += force_scalar[5] * dr[5 * CHDIM + 0];
  force[2 * CHDIM + 1] += force_scalar[5] * dr[5 * CHDIM + 1];
  force[2 * CHDIM + 2] += force_scalar[5] * dr[5 * CHDIM + 2];

  force[3 * CHDIM + 0] -= force_scalar[5] * dr[5 * CHDIM + 0];
  force[3 * CHDIM + 1] -= force_scalar[5] * dr[5 * CHDIM + 1];
  force[3 * CHDIM + 2] -= force_scalar[5] * dr[5 * CHDIM + 2];

  stress[0] -= force_scalar[5] * dr[5 * CHDIM + 0] * dr[5 * CHDIM + 0];    // xx tensor component
  stress[1] -= force_scalar[5] * dr[5 * CHDIM + 0] * dr[5 * CHDIM + 1];    // xy tensor component
  stress[2] -= force_scalar[5] * dr[5 * CHDIM + 0] * dr[5 * CHDIM + 2];    // xz tensor component
  stress[3] -= force_scalar[5] * dr[5 * CHDIM + 1] * dr[5 * CHDIM + 1];    // yy tensor component
  stress[4] -= force_scalar[5] * dr[5 * CHDIM + 1] * dr[5 * CHDIM + 2];    // yz tensor component
  stress[5] -= force_scalar[5] * dr[5 * CHDIM + 2] * dr[5 * CHDIM + 2];    // zz tensor component

  return;
}

void chimesFF::get_cutoff_2B(vector<vector<double>> &cutoff_2b)
{
  int dim = chimes_2b_cutoff.size();

  cutoff_2b.resize(dim);

  for (int i = 0; i < dim; i++) {
    cutoff_2b[i].resize(0);

    for (int j = 0; j < chimes_2b_cutoff[i].size(); j++)

      cutoff_2b[i].push_back(chimes_2b_cutoff[i][j]);
  }
}

double chimesFF::max_cutoff(int ntypes, vector<vector<vector<double>>> &cutoff_list)
{
  double max = cutoff_list[0][1][0];

  for (int i = 0; i < ntypes; i++)
    for (int j = 0; j < cutoff_list[i][1].size(); j++)
      if (cutoff_list[i][1][j] > max) max = cutoff_list[i][1][j];

  return max;
}

double chimesFF::max_cutoff_2B(bool silent)
{
  double max = chimes_2b_cutoff[0][1];

  for (int i = 0; i < chimes_2b_cutoff.size(); i++)
    if (chimes_2b_cutoff[i][1] > max) max = chimes_2b_cutoff[i][1];

  if ((rank == 0) && (!silent))
    cout << "chimesFF: " << "\t" << "Setting 2-body max cutoff to: " << max << endl;

  return max;
}

double chimesFF::max_cutoff_3B(bool silent)
{

  if (poly_orders[1] == 0) return 0.0;

  double max = max_cutoff(chimes_3b_cutoff.size(), chimes_3b_cutoff);

  if ((rank == 0) && (!silent))
    cout << "chimesFF: " << "\t" << "Setting 3-body max cutoff to: " << max << endl;

  return max;
}

double chimesFF::max_cutoff_4B(bool silent)
{
  if (poly_orders[2] == 0) return 0.0;

  double max = max_cutoff(chimes_4b_cutoff.size(), chimes_4b_cutoff);

  if ((rank == 0) && (!silent))
    cout << "chimesFF: " << "\t" << "Setting 4-body max cutoff to: " << max << endl;

  return max;
}

void chimesFF::set_atomtypes(vector<string> &type_list)
{
  type_list.resize(natmtyps);

  for (int i = 0; i < natmtyps; i++) type_list[i] = atmtyps[i];
}

int chimesFF::get_atom_pair_index(int pair_id)
{
  return atom_idx_pair_map[pair_id];
}

void chimesFF::build_pair_int_quad_map()
// Build the pair maps for all possible quads.	Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
  const int natoms = 4;
  const int npairs = natoms * (natoms - 1) / 2;
  vector<int> pair_map(npairs);
  vector<int> typ_idxs(natoms);

  if (atom_int_quad_map.size() == 0) return;    // No quads !

  pair_int_quad_map.resize(natmtyps * natmtyps * natmtyps * natmtyps);

  for (int i = 0; i < natmtyps; i++) {
    typ_idxs[0] = i;
    for (int j = 0; j < natmtyps; j++) {
      typ_idxs[1] = j;
      for (int k = 0; k < natmtyps; k++) {
        typ_idxs[2] = k;
        for (int l = 0; l < natmtyps; l++) {
          typ_idxs[3] = l;
          int idx = i * natmtyps * natmtyps * natmtyps + j * natmtyps * natmtyps + k * natmtyps + l;
          int quadidx = atom_int_quad_map[idx];

          build_atom_and_pair_mappers(natoms, npairs, typ_idxs, quad_params_pair_typs[quadidx],
                                      pair_map);

          // Save for re-use in force evaluators.
          if (quadidx >= natmtyps * natmtyps * natmtyps * natmtyps) {
            cout << "Error: quadidx out of range\n";
            cout << "Quadidx = " << quadidx << endl;
            exit(1);
          }

          // Note: The entire vector<> is copied and stored.
          pair_int_quad_map[idx] = pair_map;
        }
      }
    }
  }
  for (int i = 0; i < pair_int_quad_map.size(); i++) {
    if (pair_int_quad_map[i].size() == 0) {
      cout << "Error: Did not initialize pair_int_quad_map entry " << i << endl;
    }
  }
}

void chimesFF::build_pair_int_trip_map()
// Build the pair maps for all possible triplets.  Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
  const int natoms = 3;
  const int npairs = natoms * (natoms - 1) / 2;
  vector<int> pair_map(npairs);
  vector<int> typ_idxs(natoms);

  if (atom_int_trip_map.size() == 0) return;    // No quads !

  pair_int_trip_map.resize(natmtyps * natmtyps * natmtyps);

  for (int i = 0; i < natmtyps; i++) {
    typ_idxs[0] = i;
    for (int j = 0; j < natmtyps; j++) {
      typ_idxs[1] = j;
      for (int k = 0; k < natmtyps; k++) {
        typ_idxs[2] = k;
        int tripidx = atom_int_trip_map[i * natmtyps * natmtyps + j * natmtyps + k];

        build_atom_and_pair_mappers(natoms, npairs, typ_idxs, trip_params_pair_typs[tripidx],
                                    pair_map);

        // Save for re-use in force evaluators.
        if (tripidx >= natmtyps * natmtyps * natmtyps * natmtyps) {
          cout << "Error: tripidx out of range\n";
          cout << "Tripidx = " << tripidx << endl;
          exit(1);
        }

        // Note: The entire vector<> is copied and stored.
        pair_int_trip_map[i * natmtyps * natmtyps + j * natmtyps + k] = pair_map;
      }
    }
  }
  for (int i = 0; i < pair_int_trip_map.size(); i++) {
    if (pair_int_trip_map[i].size() == 0) {
      cout << "Error: Did not initialize pair_int_trip_map entry " << i << endl;
    }
  }
}

void chimesFF::poly_2B(double *e, double *f0, int ncoeffs_2b, vector<double> &chimes_2b_params,
                       vector<int> &chimes_2b_pows, vector<double> &Tn, vector<double> &Tnd)
// Compute the 2 body polynomial (e) and derivatives with respect to the pair distance (f0)
// (LEF) 3/11/26
{
  *e = 0.0;
  *f0 = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs_2b; coeffs++) {
    double coeff_val = chimes_2b_params[coeffs];

    *e += coeff_val * Tn[chimes_2b_pows[coeffs] + 1];
    *f0 += coeff_val * Tnd[chimes_2b_pows[coeffs] + 1];
  }
}

void chimesFF::poly_3B(double *e, double *f, int ncoeffs_3b, vector<double> &chimes_3b_params,
                       vector<int> &mapped_pair_idx, vector<vector<int>> &chimes_3b_powers,
                       vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_jk,
                       vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_jk)
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  double coeff;
  int powers[3];
  double deriv[3];

  *e = 0.0;
  f[0] = 0.0;
  f[1] = 0.0;
  f[2] = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs_3b; coeffs++) {
    coeff = chimes_3b_params[coeffs];

    powers[0] = chimes_3b_powers[coeffs][mapped_pair_idx[0]];
    powers[1] = chimes_3b_powers[coeffs][mapped_pair_idx[1]];
    powers[2] = chimes_3b_powers[coeffs][mapped_pair_idx[2]];

    *e += coeff * Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk[powers[2]];

    deriv[0] = Tnd_ij[powers[0]];
    deriv[1] = Tnd_ik[powers[1]];
    deriv[2] = Tnd_jk[powers[2]];

    f[0] += coeff * deriv[0] * Tn_ik[powers[1]] * Tn_jk[powers[2]];
    f[1] += coeff * deriv[1] * Tn_ij[powers[0]] * Tn_jk[powers[2]];
    f[2] += coeff * deriv[2] * Tn_ij[powers[0]] * Tn_ik[powers[1]];
  }
}

void chimesFF::poly_3B_dense(double &e, double &f0, double &f1, double &f2, int ncoeffs_3b,
                             vector<double> &chimes_3b_params, vector<double> &Tn_ij,
                             vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                             vector<double> &Tnd_ik, vector<double> &Tnd_jk)
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f0, f1, f2)
// (LEF) 4/02/26
{
  double coeff;
  int powers[3];
  double deriv[3];
  const int loop_style = CHIMES_LOOP_STYLE;

  e = 0.0;
  f0 = 0.0;
  f1 = 0.0;
  f2 = 0.0;

  if (ncoeffs_3b == 0) return;

  int max_poly = 0;
  const int loop_max = 1000;
  int i = 0;
  for (; i < loop_max; i++) {
    if (i * i * i == ncoeffs_3b) {
      max_poly = i;
      break;
    }
  }
  if (i == loop_max) {
    cout << "Bad number of 3 body coefficients for dense evaluation\n";
    exit(1);
  }

  if (loop_style == 1) {
    poly_3B_dense_loop1(max_poly, e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  } else if (loop_style == 2) {
    poly_3B_dense_loop2(max_poly, e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  } else if (loop_style == 3) {
    poly_3B_dense_loop3(max_poly, e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                        Tnd_ij, Tnd_ik, Tnd_jk);
  } else {
    cout << "Error: bad 3 body dense loop style\n";
    exit(1);
  }
}

void chimesFF::poly_3B_dense_loop1(int max_poly, double &e, double &f0, double &f1, double &f2,
                                   int ncoeffs_3b, vector<double> &chimes_3b_params,
                                   vector<double> &Tn_ij, vector<double> &Tn_ik,
                                   vector<double> &Tn_jk, vector<double> &Tnd_ij,
                                   vector<double> &Tnd_ik, vector<double> &Tnd_jk)
{
  for (int count = 0; count < ncoeffs_3b; count++) {
    int l = count / (max_poly * max_poly);
    if (l >= max_poly) { cout << "Internal error: l > max_poly: " << l << "\n"; }
    int m = (count / max_poly) % max_poly;
    int n = count % max_poly;

    if (chimes_3b_params[count] != 0.0) {
      const double tn_ij = Tn_ij[l];
      const double tnd_ij = Tnd_ij[l];
      const double tn_ik = Tn_ik[m];
      const double tnd_ik = Tnd_ik[m];
      const double tn_jk = Tn_jk[n];
      const double tnd_jk = Tnd_jk[n];
      const double coeff = chimes_3b_params[count];

      e += coeff * tn_ij * tn_ik * tn_jk;
      f0 += coeff * tnd_ij * tn_ik * tn_jk;
      f1 += coeff * tnd_ik * tn_ij * tn_jk;
      f2 += coeff * tnd_jk * tn_ij * tn_ik;
    }
  }
}

void chimesFF::poly_3B_dense_loop2(int max_poly, double &e, double &f0, double &f1, double &f2,
                                   int ncoeffs_3b, vector<double> &chimes_3b_params,
                                   vector<double> &Tn_ij, vector<double> &Tn_ik,
                                   vector<double> &Tn_jk, vector<double> &Tnd_ij,
                                   vector<double> &Tnd_ik, vector<double> &Tnd_jk)
{
  int count = 0;
  for (int i = 0; i < max_poly; i++) {
    const double tn_ij = Tn_ij[i];
    const double tnd_ij = Tnd_ij[i];

    for (int j = 0; j < max_poly; j++) {
      const double tn_ik = Tn_ik[j];
      const double tnd_ik = Tnd_ik[j];
      const double tn_ij_ik = tn_ij * tn_ik;

      for (int k = 0; k < max_poly; k++) {
        if (chimes_3b_params[count] != 0.0) {
          const double tn_jk = Tn_jk[k];
          const double tnd_jk = Tnd_jk[k];
          const double coeff = chimes_3b_params[count];

          e += coeff * tn_ij_ik * tn_jk;
          f0 += coeff * tnd_ij * tn_ik * tn_jk;
          f1 += coeff * tnd_ik * tn_ij * tn_jk;
          f2 += coeff * tnd_jk * tn_ij_ik;
        }
        count++;
      }
    }
  }
}

void chimesFF::poly_3B_dense_loop3(int max_poly, double &e, double &f0, double &f1, double &f2,
                                   int ncoeffs_3b, vector<double> &chimes_3b_params,
                                   vector<double> &Tn_ij, vector<double> &Tn_ik,
                                   vector<double> &Tn_jk, vector<double> &Tnd_ij,
                                   vector<double> &Tnd_ik, vector<double> &Tnd_jk)
{
  switch (max_poly) {
    case 0:
      return;
    case 1:
      poly_3B_dense_template<1>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 2:
      poly_3B_dense_template<2>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 3:
      poly_3B_dense_template<3>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 4:
      poly_3B_dense_template<4>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 5:
      poly_3B_dense_template<5>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 6:
      poly_3B_dense_template<6>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 7:
      poly_3B_dense_template<7>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 8:
      poly_3B_dense_template<8>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 9:
      poly_3B_dense_template<9>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    case 10:
      poly_3B_dense_template<10>(e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik, Tn_jk,
                                 Tnd_ij, Tnd_ik, Tnd_jk);
      return;
    default:
      poly_3B_dense_loop2(max_poly, e, f0, f1, f2, ncoeffs_3b, chimes_3b_params, Tn_ij, Tn_ik,
                          Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);
      return;
  }
}

void chimesFF::poly_4B_dense_loop1(
    int max_poly, double &e, double &f0, double &f1, double &f2, double &f3, double &f4, double &f5,
    int ncoeffs_4b, vector<double> &params_4b, vector<double> &Tn_ij, vector<double> &Tn_ik,
    vector<double> &Tn_il, vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
    vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
    vector<double> &Tnd_jl, vector<double> &Tnd_kl)
{
  int max_poly_pow[6];
  double coeff;

  max_poly_pow[5] = 1;
  for (int l = 4; l >= 0; l--) { max_poly_pow[l] = max_poly_pow[l + 1] * max_poly; }
  for (int count = 0; count < ncoeffs_4b; count++) {
    if (params_4b[count] != 0.0) {
      int index[6];
      for (int i = 0; i < 6; i++) { index[i] = (count / max_poly_pow[i]) % max_poly; }
      const double tn_ij = Tn_ij[index[0]];
      const double tnd_ij = Tnd_ij[index[0]];
      const double tn_ik = Tn_ik[index[1]];
      const double tnd_ik = Tnd_ik[index[1]];
      const double tn_il = Tn_il[index[2]];
      const double tnd_il = Tnd_il[index[2]];
      const double tn_jk = Tn_jk[index[3]];
      const double tnd_jk = Tnd_jk[index[3]];
      const double tn_jl = Tn_jl[index[4]];
      const double tnd_jl = Tnd_jl[index[4]];
      const double tn_kl = Tn_kl[index[5]];
      const double tnd_kl = Tnd_kl[index[5]];
      const double coeff = params_4b[count];

      const double Tn_jk_jl = tn_jk * tn_jl;
      const double Tn_ij_ik_il = tn_ij * tn_ik * tn_il;

      e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

      // deriv[0] = tnd_ij ;
      // deriv[1] = tnd_ik ;
      // deriv[2] = tnd_il ;
      // deriv[3] = tnd_jk ;
      // deriv[4] = tnd_jl ;
      // deriv[5] = tnd_kl ;

      f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;

      f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;

      f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;

      f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;

      f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;

      f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
    }
  }
}

void chimesFF::poly_4B_dense_loop2(
    int max_poly, double &e, double &f0, double &f1, double &f2, double &f3, double &f4, double &f5,
    int ncoeffs_4b, vector<double> &params_4b, vector<double> &Tn_ij, vector<double> &Tn_ik,
    vector<double> &Tn_il, vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
    vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
    vector<double> &Tnd_jl, vector<double> &Tnd_kl)
{
  double coeff;

  int count = 0;
  for (int i = 0; i < max_poly; i++) {
    const double tn_ij = Tn_ij[i];
    const double tnd_ij = Tnd_ij[i];
    for (int j = 0; j < max_poly; j++) {
      const double tn_ik = Tn_ik[j];
      const double tnd_ik = Tnd_ik[j];
      for (int l = 0; l < max_poly; l++) {
        const double tn_il = Tn_il[l];
        const double tnd_il = Tnd_il[l];
        const double Tn_ij_ik_il = tn_ij * tn_ik * tn_il;
        for (int m = 0; m < max_poly; m++) {
          const double tn_jk = Tn_jk[m];
          const double tnd_jk = Tnd_jk[m];
          for (int n = 0; n < max_poly; n++) {
            const double tn_jl = Tn_jl[n];
            const double tnd_jl = Tnd_jl[n];
            const double Tn_jk_jl = tn_jk * tn_jl;
            for (int o = 0; o < max_poly; o++) {
              const double tn_kl = Tn_kl[o];
              const double tnd_kl = Tnd_kl[o];

              if (params_4b[count] != 0.0) {
                const double coeff = params_4b[count];

                e += coeff * Tn_ij_ik_il * Tn_jk_jl * tn_kl;

                f0 += coeff * tnd_ij * tn_ik * tn_il * Tn_jk_jl * tn_kl;

                f1 += coeff * tnd_ik * tn_ij * tn_il * Tn_jk_jl * tn_kl;

                f2 += coeff * tnd_il * tn_ij * tn_ik * Tn_jk_jl * tn_kl;

                f3 += coeff * tnd_jk * Tn_ij_ik_il * tn_jl * tn_kl;

                f4 += coeff * tnd_jl * Tn_ij_ik_il * tn_jk * tn_kl;

                f5 += coeff * tnd_kl * Tn_ij_ik_il * Tn_jk_jl;
              }
              count++;
            }
          }
        }
      }
    }
  }
}

void chimesFF::poly_4B_dense_loop3(
    int max_poly, double &e, double &f0, double &f1, double &f2, double &f3, double &f4, double &f5,
    int ncoeffs_4b, vector<double> &params_4b, vector<double> &Tn_ij, vector<double> &Tn_ik,
    vector<double> &Tn_il, vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
    vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
    vector<double> &Tnd_jl, vector<double> &Tnd_kl)
{
  switch (max_poly) {
    case 0:
      return;
    case 1:
      poly_4B_dense_template<1>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 2:
      poly_4B_dense_template<2>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 3:
      poly_4B_dense_template<3>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 4:
      poly_4B_dense_template<4>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 5:
      poly_4B_dense_template<5>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 6:
      poly_4B_dense_template<6>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 7:
      poly_4B_dense_template<7>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 8:
      poly_4B_dense_template<8>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 9:
      poly_4B_dense_template<9>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                Tnd_kl);
      return;
    case 10:
      poly_4B_dense_template<10>(e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                                 Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                                 Tnd_kl);
      return;
    default:
      poly_4B_dense_loop2(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                          Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl,
                          Tnd_kl);
      return;
  }
}

void chimesFF::poly_4B_dense(double &e, double &f0, double &f1, double &f2, double &f3, double &f4,
                             double &f5, int ncoeffs_4b, vector<double> &params_4b,
                             vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                             vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                             vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il,
                             vector<double> &Tnd_jk, vector<double> &Tnd_jl, vector<double> &Tnd_kl)
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f0, f1, f2)
// (LEF) 4/02/26
{
  double coeff;
  int powers[6];
  double deriv[6];
  const int loop_style = CHIMES_LOOP_STYLE;

  e = 0.0;
  f0 = 0.0;
  f1 = 0.0;
  f2 = 0.0;
  f3 = 0.0;
  f4 = 0.0;
  f5 = 0.0;

  if (ncoeffs_4b == 0) return;

  int max_poly = 0;
  const int loop_max = 100;
  int i = 0;
  for (; i < loop_max; i++) {
    if (i * i * i * i * i * i == ncoeffs_4b) {
      max_poly = i;
      break;
    }
  }
  if (i == loop_max) {
    cout << "Bad number of 4 body coefficients for dense evaluation\n";
    exit(1);
  }

  if (loop_style == 1) {
    poly_4B_dense_loop1(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                        Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else if (loop_style == 2) {
    poly_4B_dense_loop2(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                        Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else if (loop_style == 3) {
    poly_4B_dense_loop3(max_poly, e, f0, f1, f2, f3, f4, f5, ncoeffs_4b, params_4b, Tn_ij, Tn_ik,
                        Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  }
}

void chimesFF::poly_4B(double *e, double *f, int ncoeffs_4b, vector<double> &chimes_4b_params,
                       vector<int> &mapped_pair_idx, vector<vector<int>> &chimes_4b_powers,
                       vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                       vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                       vector<double> &Tnd_ij, vector<double> &Tnd_ik, vector<double> &Tnd_il,
                       vector<double> &Tnd_jk, vector<double> &Tnd_jl, vector<double> &Tnd_kl)
// Compute the 4 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  double coeff;
  const int npairs = 6;
  int powers[npairs];
  double deriv[npairs];

  *e = 0;
  for (int i = 0; i < 6; i++) f[i] = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs_4b; coeffs++) {
    coeff = chimes_4b_params[coeffs];

    for (int i = 0; i < npairs; i++) powers[i] = chimes_4b_powers[coeffs][mapped_pair_idx[i]];

    double Tn_ij_ik_il = Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_il[powers[2]];
    double Tn_jk_jl = Tn_jk[powers[3]] * Tn_jl[powers[4]];
    double Tn_kl_5 = Tn_kl[powers[5]];

    *e += coeff * Tn_ij_ik_il * Tn_jk_jl * Tn_kl_5;

    deriv[0] = Tnd_ij[powers[0]];
    deriv[1] = Tnd_ik[powers[1]];
    deriv[2] = Tnd_il[powers[2]];
    deriv[3] = Tnd_jk[powers[3]];
    deriv[4] = Tnd_jl[powers[4]];
    deriv[5] = Tnd_kl[powers[5]];

    f[0] += coeff * deriv[0] * Tn_ik[powers[1]] * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;

    f[1] += coeff * deriv[1] * Tn_ij[powers[0]] * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;

    f[2] += coeff * deriv[2] * Tn_ij[powers[0]] * Tn_ik[powers[1]] * Tn_jk_jl * Tn_kl_5;

    f[3] += coeff * deriv[3] * Tn_ij_ik_il * Tn_jl[powers[4]] * Tn_kl_5;

    f[4] += coeff * deriv[4] * Tn_ij_ik_il * Tn_jk[powers[3]] * Tn_kl_5;

    f[5] += coeff * deriv[5] * Tn_ij_ik_il * Tn_jk_jl;
  }
}

void chimesFF::densify_3B(int &ncoeffs3, vector<vector<int>> &powers_3b, vector<double> &params_3b)
// This converts the 3 body coefficients to "dense form" where all possible powers are used.  This form
// may be more efficient on GPUs or vectorized CPU architectures. (LEF 4/2/26)
{
  int max_pow3b = 0;

  for (int j = 0; j < powers_3b.size(); j++) {
    for (int k = 0; k < powers_3b[j].size(); k++) {
      if (powers_3b[j][k] > max_pow3b) { max_pow3b = powers_3b[j][k]; }
    }
  }

  cout << "Maximum 3-B power found = " << max_pow3b << endl;

  int dim1 = max_pow3b + 1;
  int dim = dim1 * dim1 * dim1;

  vector<double> dense_coeffs(dim, 0.0);

  for (int j = 0; j < ncoeffs3; j++) {
    int index = powers_3b[j][0] * dim1 * dim1 + powers_3b[j][1] * dim1 + powers_3b[j][2];
    dense_coeffs[index] = params_3b[j];
  }

  params_3b.resize(dim);
  for (int j = 0; j < dim; j++) { params_3b[j] = dense_coeffs[j]; }

  powers_3b.resize(dim);
  int count = 0;
  for (int j = 0; j < dim; j++) { powers_3b[j].resize(3); }

  for (int j = 0; j < dim1; j++) {
    for (int k = 0; k < dim1; k++) {
      for (int l = 0; l < dim1; l++) {
        powers_3b[count][0] = j;
        powers_3b[count][1] = k;
        powers_3b[count][2] = l;
        count++;
      }
    }
  }

  ncoeffs3 = dim;
}

void chimesFF::densify_4B(int &ncoeffs4, vector<vector<int>> &powers_4b, vector<double> &params_4b)
// This converts the 4 body coefficients to "dense form" where all possible powers are used.  This form
// may be more efficient on GPUs or vectorized CPU architectures. (LEF 4/2/26)
{
  int max_pow4b = 0;

  for (int j = 0; j < powers_4b.size(); j++) {
    for (int k = 0; k < powers_4b[j].size(); k++) {
      if (powers_4b[j][k] > max_pow4b) { max_pow4b = powers_4b[j][k]; }
    }
  }

  cout << "Maximum 4-B power found = " << max_pow4b << endl;

  int dim1 = max_pow4b + 1;
  int dim = dim1 * dim1 * dim1 * dim1 * dim1 * dim1;

  vector<double> dense_coeffs(dim, 0.0);

  for (int j = 0; j < ncoeffs4; j++) {
    int offset = 1;
    int index = 0;
    for (int l = 5; l >= 0; l--) {
      index += powers_4b[j][l] * offset;
      offset *= dim1;
    }
    if (index >= dim) {
      cout << "Error in calculating parameter index\n";
      exit(1);
    }
    dense_coeffs[index] = params_4b[j];
  }
  params_4b.resize(dim);
  for (int j = 0; j < dim; j++) { params_4b[j] = dense_coeffs[j]; }

  powers_4b.resize(dim);
  for (int j = 0; j < dim; j++) { powers_4b[j].resize(6); }

  int count = 0;
  for (int j = 0; j < dim1; j++) {
    for (int k = 0; k < dim1; k++) {
      for (int l = 0; l < dim1; l++) {
        for (int m = 0; m < dim1; m++) {
          for (int n = 0; n < dim1; n++) {
            for (int o = 0; o < dim1; o++) {

              if (count >= dim) {
                cout << "Count index overflow error\n";
                exit(1);
              }
              powers_4b[count][0] = j;
              powers_4b[count][1] = k;
              powers_4b[count][2] = l;
              powers_4b[count][3] = m;
              powers_4b[count][4] = n;
              powers_4b[count][5] = o;
              count++;
            }
          }
        }
      }
    }
  }
  ncoeffs4 = dim;
}
