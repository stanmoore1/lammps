/*
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author:  Rebecca K. Lindsey (2020)
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
            "0100010101101110110011101101001011011101100101     "
         << endl;
    cout << "chimesFF: " << endl;
    cout << "chimesFF: " << "       _____  _         _____    __    __    ______     _____     ______           _              "
         << endl;
    cout << "chimesFF: "
         << "      / ____|| |    |_     _||  \\/  ||  ____| / ____| |    ____|           (_)              " << endl;
    cout << "chimesFF: "
         << "     | |     | |__      | |  | \\     / || |__    | (___     | |__      _ __      __ _    _  _ __       ___     "
            " "
         << endl;
    cout << "chimesFF: "
         << "     | |     | '_ \\   | |    | |\\/| ||    __|      \\___ \\    |  __|    | '_ \\     / _` || || '_ \\  "
            "/ _ \\ "
         << endl;
    cout << "chimesFF: "
         << "     | |____ | | | | _| |_ | |    | || |____    ____) | | |____ | | | || (_| || || | | ||  "
            "__/      "
         << endl;
    cout << "chimesFF: "
         << "      \\_____||_| |_||_____||_|     |_||______||_____/     |______||_| |_| \\__, ||_||_| |_| "
            "\\___|      "
         << endl;
    cout << "chimesFF: " << "                                     __/ |              " << endl;
    cout << "chimesFF: " << "                                    |___/              " << endl;
    cout << "chimesFF: " << endl;
    cout << "chimesFF: " << "              Copyright (C) 2020 R.K. Lindsey, L.E. Fried, N. Goldman              "
         << endl;
    cout << "chimesFF: " << endl;
    cout << "chimesFF: "
         << "01000011011010001001001010011010100010101010011 "
            "0100010101101110110011101101001011011101100101      "
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
        cout << "chimesFF: " << "    ...Is this a ChIMES force field parameter file?" << endl;
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
//    Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd) when dx is < inner_cutoff.
//    Tnd is the derivative with respect to the interatomic distance, not the transformed distance (x).
//
//    The derivative Tnd is continuously set to zero inside the cutoff.
//    The exponential smoothing distance is set to chimesFF::inner_smooth_distance.
//    x, exprlen, and dx_dr are evaluated at the inner cutoff.
//
//    dx is the pair distance, which is assumed to be less than inner_cutoff.
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

CHIMES_VECTOR_CLONES
void chimesFF::compute_2B(const double dx, const vector<double> &dr, const vector<int> &typ_idxs,
                          vector<double> &force, vector<double> &stress, double &energy,
                          chimes2BTmp &tmp, const bool vflag)
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

  const chimesSlotConst &sc = slot_2b[typ_idxs[0] * natmtyps + typ_idxs[1]];

  if (dx >= sc.outer) return;

  pair_idx = atom_int_pair_map[typ_idxs[0] * natmtyps + typ_idxs[1]];

  get_fcut(dx, sc, fcut, fcutderiv);

  double poly, dpoly_dx;

  if (!mono_2b[pair_idx].empty() && (dx >= sc.inner)) {
    // The series in the monomial basis: one exponential, then Horner's rule
    // for the value and the derivative together.  The Chebyshev recurrence,
    // its arrays, and the coefficient gather all disappear.  A separation
    // inside the inner cutoff needs the damped form and takes the old path.

    const double exprlen = exp(dx * sc.neg_inv_morse);
    const double x = (exprlen - sc.x_avg) * sc.inv_x_diff;
    const double dx_dr = exprlen * sc.dxdr_scale;

    const vector<double> &row = mono_2b[pair_idx];

    double V = 0.0, D = 0.0;

    for (int k = (int) row.size() - 1; k >= 0; k--) {
      D = D * x + V;
      V = V * x + row[k];
    }

    poly = V;
    dpoly_dx = D * dx_dr;
  } else {
    set_cheby_polys(Tn, Tnd, dx, sc, 0);

    poly_2B(&poly, &dpoly_dx, ncoeffs_2b[pair_idx], chimes_2b_params[pair_idx],
            chimes_2b_pows[pair_idx], Tn, Tnd);
  }

  energy += poly * fcut;
  double force_scalar = (fcut * dpoly_dx + fcutderiv * poly) / dx;

  force[0 * CHDIM + 0] += force_scalar * dr[0];
  force[0 * CHDIM + 1] += force_scalar * dr[1];
  force[0 * CHDIM + 2] += force_scalar * dr[2];

  force[1 * CHDIM + 0] -= force_scalar * dr[0];
  force[1 * CHDIM + 1] -= force_scalar * dr[1];
  force[1 * CHDIM + 2] -= force_scalar * dr[2];

  // xx xy xz yy yz zz
  // 0  1     2    3  4  5

  // xx xy xz yx yy yz zx zy zz
  // 0  1     2    3  4  5     6    7  8
  // *           *       *

  if (vflag) {
    stress[0] -= force_scalar * dr[0] * dr[0];    // xx tensor component
    stress[1] -= force_scalar * dr[0] * dr[1];    // xy tensor component
    stress[2] -= force_scalar * dr[0] * dr[2];    // xz tensor component
    stress[3] -= force_scalar * dr[1] * dr[1];    // yy tensor component
    stress[4] -= force_scalar * dr[1] * dr[2];    // yz tensor component
    stress[5] -= force_scalar * dr[2] * dr[2];    // zz tensor component
  }

  double E_penalty = 0.0;
  get_penalty(dx, pair_idx, sc.inner, E_penalty, force_scalar);

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
    if (vflag) {
      stress[0] -= force_scalar * dr[0] * dr[0];    // xx tensor component
      stress[1] -= force_scalar * dr[0] * dr[1];    // xy tensor component
      stress[2] -= force_scalar * dr[0] * dr[2];    // xz tensor component
      stress[3] -= force_scalar * dr[1] * dr[1];    // yy tensor component
      stress[4] -= force_scalar * dr[1] * dr[2];    // yz tensor component
      stress[5] -= force_scalar * dr[2] * dr[2];    // zz tensor component
    }
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

CHIMES_VECTOR_CLONES
void chimesFF::compute_3B(const vector<double> &dx, const vector<double> &dr,
                          const vector<int> &typ_idxs, vector<double> &force,
                          vector<double> &stress, double &energy, chimes3BTmp &tmp,
                          const bool vflag)
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
  const chimesSlotConst *sc = &slot_3b[type_idx * 3];

  if (dx[0] >= sc[0].outer) return;    // ij
  if (dx[1] >= sc[1].outer) return;    // ik
  if (dx[2] >= sc[2].outer) return;    // jk

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  // Set up the polynomials and the smoothing functions, reusing whatever the
  // previous cluster left behind for a pair slot it shares with this one.
  // Neighboring triplets differ only in k, so the ij slot is almost always a
  // hit, and that is one of the three exp() and one of the three sincos().

  if (!pair_cached(tmp.cache[0], sc[0], dx[0])) {
    set_cheby_polys(Tn_ij, Tnd_ij, dx[0], sc[0], 1);
    get_fcut(dx[0], sc[0], tmp.cache[0].fcut, tmp.cache[0].fcutderiv);
  }
  if (!pair_cached(tmp.cache[1], sc[1], dx[1])) {
    set_cheby_polys(Tn_ik, Tnd_ik, dx[1], sc[1], 1);
    get_fcut(dx[1], sc[1], tmp.cache[1].fcut, tmp.cache[1].fcutderiv);
  }
  if (!pair_cached(tmp.cache[2], sc[2], dx[2])) {
    set_cheby_polys(Tn_jk, Tnd_jk, dx[2], sc[2], 1);
    get_fcut(dx[2], sc[2], tmp.cache[2].fcut, tmp.cache[2].fcutderiv);
  }

  for (int p = 0; p < npairs; p++) {
    fcut[p] = tmp.cache[p].fcut;
    fcutderiv[p] = tmp.cache[p].fcutderiv;
  }

  double fcut_all = fcut[0] * fcut[1] * fcut[2];

  double poly, dpoly_dx[npairs];

  // Start the force/stress/energy calculation
  double force_scalar[npairs];

  if (!dense_coeffs) {
    const chimesPolySet &ps = poly_3b_set[type_idx];

    if (ps.grouped)
      poly_3B_grouped(&poly, dpoly_dx, *ps.grouped, Tn_ij, Tn_ik, Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);
    else
      poly_3B(&poly, dpoly_dx, ps, Tn_ij, Tn_ik, Tn_jk, Tnd_ij, Tnd_ik, Tnd_jk);
  } else {

    // JIT evaluation of the chebyshev polynomial and its derivatives
    const vector<int> &mapped_pair_idx = pair_int_trip_map[type_idx];
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++) { inv_mapped_pair[mapped_pair_idx[j]] = j; }

    vector<double> *Tn[npairs], *Tnd[npairs];

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

  if (vflag) {
    stress[0] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[0] * dr[0 * CHDIM + 2] * dr[0 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the ik pair

  force[0 * CHDIM + 0] += force_scalar[1] * dr[1 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[1] * dr[1 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[1] * dr[1 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[1] * dr[1 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[1] * dr[1 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[1] * dr[1 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[1] * dr[1 * CHDIM + 2] * dr[1 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the jk pair

  force[1 * CHDIM + 0] += force_scalar[2] * dr[2 * CHDIM + 0];
  force[1 * CHDIM + 1] += force_scalar[2] * dr[2 * CHDIM + 1];
  force[1 * CHDIM + 2] += force_scalar[2] * dr[2 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[2] * dr[2 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[2] * dr[2 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[2] * dr[2 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[2] * dr[2 * CHDIM + 2] * dr[2 * CHDIM + 2];    // zz tensor component
  }

  return;
}

CHIMES_VECTOR_CLONES
void chimesFF::compute_4B(const vector<double> &dx, const vector<double> &dr,
                          const vector<int> &typ_idxs, vector<double> &force,
                          vector<double> &stress, double &energy, chimes4BTmp &tmp,
                          const bool vflag)
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

  const chimesSlotConst *sc = &slot_4b[idx * 6];

  // Check whether cutoffs are within allowed ranges

  for (int i = 0; i < npairs; i++)
    if (dx[i] >= sc[i].outer) return;

  // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

  // Set up the polynomials

  // Neighboring quadruplets differ only in l, so ij, ik and jk -- half the
  // slots -- are typically already set up from the previous cluster.

  vector<double> *const tn[6] = {&Tn_ij, &Tn_ik, &Tn_il, &Tn_jk, &Tn_jl, &Tn_kl};
  vector<double> *const tnd[6] = {&Tnd_ij, &Tnd_ik, &Tnd_il, &Tnd_jk, &Tnd_jl, &Tnd_kl};

  for (int p = 0; p < npairs; p++) {
    if (!pair_cached(tmp.cache[p], sc[p], dx[p])) {
      set_cheby_polys(*tn[p], *tnd[p], dx[p], sc[p], 2);
      get_fcut(dx[p], sc[p], tmp.cache[p].fcut, tmp.cache[p].fcutderiv);
    }

    fcut[p] = tmp.cache[p].fcut;
    fcutderiv[p] = tmp.cache[p].fcutderiv;
  }

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
    const chimesPolySet &ps = poly_4b_set[idx];

    if (ps.grouped)
      poly_4B_grouped(&poly, dpoly_dx, *ps.grouped, Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl,
                      Tnd_ij, Tnd_ik, Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
    else
      poly_4B(&poly, dpoly_dx, ps, Tn_ij, Tn_ik, Tn_il, Tn_jk, Tn_jl, Tn_kl, Tnd_ij, Tnd_ik,
              Tnd_il, Tnd_jk, Tnd_jl, Tnd_kl);
  } else {
    // Dense evaluation of the chebyshev polynomial and its derivatives
    const vector<int> &mapped_pair_idx = pair_int_quad_map[idx];
    int inv_mapped_pair[npairs];

    for (int j = 0; j < npairs; j++) { inv_mapped_pair[mapped_pair_idx[j]] = j; }

    vector<double> *Tn[npairs], *Tnd[npairs];

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

  if (vflag) {
    stress[0] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[0] * dr[0 * CHDIM + 0] * dr[0 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[0] * dr[0 * CHDIM + 1] * dr[0 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[0] * dr[0 * CHDIM + 2] * dr[0 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the ik pair

  force[0 * CHDIM + 0] += force_scalar[1] * dr[1 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[1] * dr[1 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[1] * dr[1 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[1] * dr[1 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[1] * dr[1 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[1] * dr[1 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[1] * dr[1 * CHDIM + 0] * dr[1 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[1] * dr[1 * CHDIM + 1] * dr[1 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[1] * dr[1 * CHDIM + 2] * dr[1 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the il pair

  force[0 * CHDIM + 0] += force_scalar[2] * dr[2 * CHDIM + 0];
  force[0 * CHDIM + 1] += force_scalar[2] * dr[2 * CHDIM + 1];
  force[0 * CHDIM + 2] += force_scalar[2] * dr[2 * CHDIM + 2];

  force[3 * CHDIM + 0] -= force_scalar[2] * dr[2 * CHDIM + 0];
  force[3 * CHDIM + 1] -= force_scalar[2] * dr[2 * CHDIM + 1];
  force[3 * CHDIM + 2] -= force_scalar[2] * dr[2 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[2] * dr[2 * CHDIM + 0] * dr[2 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[2] * dr[2 * CHDIM + 1] * dr[2 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[2] * dr[2 * CHDIM + 2] * dr[2 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the jk pair

  force[1 * CHDIM + 0] += force_scalar[3] * dr[3 * CHDIM + 0];
  force[1 * CHDIM + 1] += force_scalar[3] * dr[3 * CHDIM + 1];
  force[1 * CHDIM + 2] += force_scalar[3] * dr[3 * CHDIM + 2];

  force[2 * CHDIM + 0] -= force_scalar[3] * dr[3 * CHDIM + 0];
  force[2 * CHDIM + 1] -= force_scalar[3] * dr[3 * CHDIM + 1];
  force[2 * CHDIM + 2] -= force_scalar[3] * dr[3 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[3] * dr[3 * CHDIM + 0] * dr[3 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[3] * dr[3 * CHDIM + 0] * dr[3 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[3] * dr[3 * CHDIM + 0] * dr[3 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[3] * dr[3 * CHDIM + 1] * dr[3 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[3] * dr[3 * CHDIM + 1] * dr[3 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[3] * dr[3 * CHDIM + 2] * dr[3 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the jl pair

  force[1 * CHDIM + 0] += force_scalar[4] * dr[4 * CHDIM + 0];
  force[1 * CHDIM + 1] += force_scalar[4] * dr[4 * CHDIM + 1];
  force[1 * CHDIM + 2] += force_scalar[4] * dr[4 * CHDIM + 2];

  force[3 * CHDIM + 0] -= force_scalar[4] * dr[4 * CHDIM + 0];
  force[3 * CHDIM + 1] -= force_scalar[4] * dr[4 * CHDIM + 1];
  force[3 * CHDIM + 2] -= force_scalar[4] * dr[4 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[4] * dr[4 * CHDIM + 0] * dr[4 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[4] * dr[4 * CHDIM + 0] * dr[4 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[4] * dr[4 * CHDIM + 0] * dr[4 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[4] * dr[4 * CHDIM + 1] * dr[4 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[4] * dr[4 * CHDIM + 1] * dr[4 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[4] * dr[4 * CHDIM + 2] * dr[4 * CHDIM + 2];    // zz tensor component
  }

  // Accumulate forces/stresses on/from the kl pair

  force[2 * CHDIM + 0] += force_scalar[5] * dr[5 * CHDIM + 0];
  force[2 * CHDIM + 1] += force_scalar[5] * dr[5 * CHDIM + 1];
  force[2 * CHDIM + 2] += force_scalar[5] * dr[5 * CHDIM + 2];

  force[3 * CHDIM + 0] -= force_scalar[5] * dr[5 * CHDIM + 0];
  force[3 * CHDIM + 1] -= force_scalar[5] * dr[5 * CHDIM + 1];
  force[3 * CHDIM + 2] -= force_scalar[5] * dr[5 * CHDIM + 2];

  if (vflag) {
    stress[0] -= force_scalar[5] * dr[5 * CHDIM + 0] * dr[5 * CHDIM + 0];    // xx tensor component
    stress[1] -= force_scalar[5] * dr[5 * CHDIM + 0] * dr[5 * CHDIM + 1];    // xy tensor component
    stress[2] -= force_scalar[5] * dr[5 * CHDIM + 0] * dr[5 * CHDIM + 2];    // xz tensor component
    stress[3] -= force_scalar[5] * dr[5 * CHDIM + 1] * dr[5 * CHDIM + 1];    // yy tensor component
    stress[4] -= force_scalar[5] * dr[5 * CHDIM + 1] * dr[5 * CHDIM + 2];    // yz tensor component
    stress[5] -= force_scalar[5] * dr[5 * CHDIM + 2] * dr[5 * CHDIM + 2];    // zz tensor component
  }

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
// Build the pair maps for all possible quads.    Moved build_atom_and_pair_mappers out of the compute_XX routines
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

// Fill one slot record.  This is the only place the Morse transform bounds and
// the cutoff-function constants are derived; the expressions match what the
// compute routines used to evaluate inline, so the tabulated values are the
// same bits the old code produced on every call.

void chimesFF::fill_slot(chimesSlotConst &sc, int pair_idx, double inner, double outer)
{
  sc.morse = morse_var[pair_idx];
  sc.inner = inner;
  sc.outer = outer;
  sc.outer_sq = outer * outer;

  const double x_min = exp(-1 * inner / sc.morse);
  const double x_max = exp(-1 * outer / sc.morse);

  sc.x_avg = 0.5 * (x_max + x_min);
  sc.x_diff = 0.5 * (x_max - x_min);
  sc.x_diff *= -1.0;    // Special for Morse style

  // The per-lane setup divides by morse and by x_diff on every call; both are
  // fixed for the slot, so the reciprocals are taken once here instead.

  sc.neg_inv_morse = -1.0 / sc.morse;
  sc.inv_x_diff = 1.0 / sc.x_diff;
  sc.dxdr_scale = sc.neg_inv_morse * sc.inv_x_diff;

  sc.fcut_thresh = outer - fcut_var * outer;
  const double fcut_span = outer - sc.fcut_thresh;

  sc.fcut_mid = 0.5 * (sc.fcut_thresh + outer) - CHIMES_PI_PHASE * fcut_span;

  if (fcut_type == fcutType::CUBIC)
    sc.fcut_dscale = -1.0 * 3.0 / outer;
  else
    sc.fcut_dscale = 1.0 / fcut_span;
}

void chimesFF::build_interaction_tables()
{
  const int n = natmtyps;

  // A pre-permuted power block depends only on (parameter set, permutation),
  // not on the atom-type index, so the blocks are pooled and shared.  Pool
  // slots are recorded as indices first and turned into pointers in a second
  // pass, because growing the pool would invalidate any pointer taken early.

  map<pair<int, vector<int>>, int> pool_slot;
  vector<int> set_slot_3b, set_slot_4b;
  vector<int> grp_slot_3b, grp_slot_4b;
  vector<int> grouped_of_pool;    // powers_pool slot -> grouped_pool slot

  // 2-body: one slot per ordered atom type pair

  slot_2b.assign(n * n, chimesSlotConst());

  // The 2-body series in the monomial basis, one row per pair type.  Same
  // change of basis as the 3-body Horner leaves; same conditioning guard.

  mono_2b.assign(chimes_2b_params.size(), vector<double>());

  {
    const int dim = poly_orders[0] + 1;

    vector<double> M((size_t) dim * dim, 0.0);

    M[0] = 1.0;

    if (dim > 1) M[1 * dim + 1] = 1.0;

    for (int pp = 2; pp < dim; pp++) {
      for (int k = 1; k < dim; k++)
        M[(size_t) pp * dim + k] = 2.0 * M[(size_t) (pp - 1) * dim + k - 1];

      for (int k = 0; k < dim; k++) M[(size_t) pp * dim + k] -= M[(size_t) (pp - 2) * dim + k];
    }

    for (size_t pi = 0; pi < chimes_2b_params.size(); pi++) {
      vector<double> row(dim, 0.0);

      double amp_den = 0.0;

      for (size_t c = 0; c < chimes_2b_params[pi].size(); c++) {
        const double v = chimes_2b_params[pi][c];
        const int pp = chimes_2b_pows[pi][c] + 1;

        amp_den += fabs(v);

        for (int k = 0; k <= pp; k++) row[k] += v * M[(size_t) pp * dim + k];
      }

      double amp_num = 0.0;

      for (int k = 0; k < dim; k++) amp_num += fabs(row[k]);

      if (amp_num <= 1.0e5 * amp_den) mono_2b[pi].swap(row);
    }
  }

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      const int pair_idx = atom_int_pair_map[i * n + j];
      fill_slot(slot_2b[i * n + j], pair_idx, chimes_2b_cutoff[pair_idx][0],
                chimes_2b_cutoff[pair_idx][1]);
    }

  // 3-body: three slots per ordered atom type triple.  The Morse lambda comes
  // from the runtime pair's own type and the cutoffs from the parameter slot it
  // maps onto, exactly as compute_3B looked them up.

  if (poly_orders[1] > 0) {
    slot_3b.assign(n * n * n * 3, chimesSlotConst());
    poly_3b_set.assign(n * n * n, chimesPolySet{0, nullptr, nullptr, nullptr});
    set_slot_3b.assign(n * n * n, -1);
    grp_slot_3b.assign(n * n * n, -1);

    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++) {
          const int type_idx = (i * n + j) * n + k;
          const int tripidx = atom_int_trip_map[type_idx];

          if (tripidx < 0) continue;    // excluded interaction

          const vector<int> &map = pair_int_trip_map[type_idx];
          const int pidx[3] = {atom_int_pair_map[i * n + j], atom_int_pair_map[i * n + k],
                               atom_int_pair_map[j * n + k]};

          for (int p = 0; p < 3; p++)
            fill_slot(slot_3b[type_idx * 3 + p], pidx[p], chimes_3b_cutoff[tripidx][0][map[p]],
                      chimes_3b_cutoff[tripidx][1][map[p]]);

          poly_3b_set[type_idx].ncoeffs = ncoeffs_3b[tripidx];
          poly_3b_set[type_idx].params = chimes_3b_params[tripidx].data();
          const int ps = permuted_powers(pool_slot, tripidx, 3, map, chimes_3b_powers[tripidx],
                                         ncoeffs_3b[tripidx]);
          set_slot_3b[type_idx] = ps;

          if ((int) grouped_of_pool.size() <= ps) grouped_of_pool.resize(ps + 1, -1);

          if (grouped_of_pool[ps] < 0)
            grouped_of_pool[ps] = build_grouped(3, powers_pool[ps],
                                                chimes_3b_params[tripidx].data(),
                                                ncoeffs_3b[tripidx]);

          grp_slot_3b[type_idx] = grouped_of_pool[ps];
        }
  }

  // 4-body: six slots per ordered atom type quadruple

  if (poly_orders[2] > 0) {
    slot_4b.assign(n * n * n * n * 6, chimesSlotConst());
    poly_4b_set.assign(n * n * n * n, chimesPolySet{0, nullptr, nullptr, nullptr});
    set_slot_4b.assign(n * n * n * n, -1);
    grp_slot_4b.assign(n * n * n * n, -1);

    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
          for (int l = 0; l < n; l++) {
            const int type_idx = ((i * n + j) * n + k) * n + l;
            const int quadidx = atom_int_quad_map[type_idx];

            if (quadidx < 0) continue;    // excluded interaction

            const vector<int> &map = pair_int_quad_map[type_idx];
            const int pidx[6] = {atom_int_pair_map[i * n + j], atom_int_pair_map[i * n + k],
                                 atom_int_pair_map[i * n + l], atom_int_pair_map[j * n + k],
                                 atom_int_pair_map[j * n + l], atom_int_pair_map[k * n + l]};

            for (int p = 0; p < 6; p++)
              fill_slot(slot_4b[type_idx * 6 + p], pidx[p], chimes_4b_cutoff[quadidx][0][map[p]],
                        chimes_4b_cutoff[quadidx][1][map[p]]);

            poly_4b_set[type_idx].ncoeffs = ncoeffs_4b[quadidx];
            poly_4b_set[type_idx].params = chimes_4b_params[quadidx].data();
            const int ps = permuted_powers(pool_slot, quadidx, 6, map, chimes_4b_powers[quadidx],
                                           ncoeffs_4b[quadidx]);
            set_slot_4b[type_idx] = ps;

            if ((int) grouped_of_pool.size() <= ps) grouped_of_pool.resize(ps + 1, -1);

            if (grouped_of_pool[ps] < 0)
              grouped_of_pool[ps] = build_grouped(6, powers_pool[ps],
                                                  chimes_4b_params[quadidx].data(),
                                                  ncoeffs_4b[quadidx]);

            grp_slot_4b[type_idx] = grouped_of_pool[ps];
          }
  }

  // The pool is complete, so it is now safe to hand out pointers into it.

  for (size_t t = 0; t < set_slot_3b.size(); t++)
    if (set_slot_3b[t] >= 0) {
      poly_3b_set[t].powers = powers_pool[set_slot_3b[t]].data();
      if (grp_slot_3b[t] >= 0) poly_3b_set[t].grouped = &grouped_pool[grp_slot_3b[t]];
    }

  for (size_t t = 0; t < set_slot_4b.size(); t++)
    if (set_slot_4b[t] >= 0) {
      poly_4b_set[t].powers = powers_pool[set_slot_4b[t]].data();
      if (grp_slot_4b[t] >= 0) poly_4b_set[t].grouped = &grouped_pool[grp_slot_4b[t]];
    }
}

// Return the pool slot holding chimes_Xb_powers permuted into runtime pair
// order for this cluster type, creating it if this (parameter set,
// permutation) combination has not been seen yet.

int chimesFF::permuted_powers(map<pair<int, vector<int>>, int> &pool_slot, int cluster_idx,
                              int npairs, const vector<int> &map, const vector<vector<int>> &powers,
                              int ncoeffs)
{
  const pair<int, vector<int>> key(cluster_idx, map);
  auto it = pool_slot.find(key);

  if (it != pool_slot.end()) return it->second;

  const int slot = powers_pool.size();
  powers_pool.emplace_back();
  vector<int> &flat = powers_pool.back();
  flat.resize((size_t) ncoeffs * npairs);

  for (int c = 0; c < ncoeffs; c++)
    for (int p = 0; p < npairs; p++) flat[(size_t) c * npairs + p] = powers[c][map[p]];

  pool_slot[key] = slot;

  return slot;
}

// Arrange one coefficient set as a tree over its leading npairs-1 powers.
// Sorting the coefficients lexicographically by power tuple makes every subtree
// a contiguous run, so the tree is just a set of index ranges over the sorted
// order and the evaluator never chases a pointer.

int chimesFF::build_grouped(int npairs, const vector<int> &flatpow, const double *params,
                            int ncoeffs)
{
  const int nlevels = npairs - 1;

  vector<int> order(ncoeffs);

  for (int c = 0; c < ncoeffs; c++) order[c] = c;

  sort(order.begin(), order.end(), [&](int a, int b) {
    for (int p = 0; p < npairs; p++) {
      const int pa = flatpow[(size_t) a * npairs + p];
      const int pb = flatpow[(size_t) b * npairs + p];

      if (pa != pb) return pa < pb;
    }
    return false;
  });

  const int slot = grouped_pool.size();
  grouped_pool.emplace_back();
  chimesGroupedPoly &g = grouped_pool.back();
  g.nlevels = nlevels;

  g.leaf_pow.resize(ncoeffs);
  g.leaf_c.resize(ncoeffs);

  for (int c = 0; c < ncoeffs; c++) {
    g.leaf_pow[c] = flatpow[(size_t) order[c] * npairs + nlevels];
    g.leaf_c[c] = params[order[c]];
  }

  // Split the sorted range level by level.  ranges holds, for each node of the
  // level just built, the span of sorted coefficients underneath it.

  vector<pair<int, int>> ranges(1, make_pair(0, ncoeffs));

  for (int d = 0; d < nlevels; d++) {
    vector<pair<int, int>> child;

    g.level_pow[d].resize(0);
    g.level_start[d].resize(0);

    for (size_t r = 0; r < ranges.size(); r++) {
      int c = ranges[r].first;

      while (c < ranges[r].second) {
        const int v = flatpow[(size_t) order[c] * npairs + d];
        int c2 = c;

        while ((c2 < ranges[r].second) && (flatpow[(size_t) order[c2] * npairs + d] == v)) c2++;

        g.level_pow[d].push_back(v);
        child.push_back(make_pair(c, c2));

        c = c2;
      }
    }

    // Record where each parent's children begin.  For d > 0 that is an index
    // into this level's node array; the parent level was built last iteration.

    if (d > 0) {
      g.level_start[d - 1].resize(ranges.size() + 1);
      g.level_start[d - 1][0] = 0;

      size_t node = 0, done = 0;

      for (size_t r = 0; r < ranges.size(); r++) {
        while ((node < child.size()) && (child[node].first < ranges[r].second)) {
          node++;
          done++;
        }
        g.level_start[d - 1][r + 1] = done;
      }
    }

    ranges.swap(child);
  }

  // The deepest level's children are the leaf coefficients themselves.

  g.level_start[nlevels - 1].resize(ranges.size() + 1);

  for (size_t r = 0; r < ranges.size(); r++) g.level_start[nlevels - 1][r] = ranges[r].first;

  g.level_start[nlevels - 1][ranges.size()] = ncoeffs;

  // Re-express each deepest node's leaf series in the monomial basis, so the
  // evaluators can use Horner's rule: no basis arrays to load, only the node's
  // own coefficients and the transformed coordinate.  The change of basis is a
  // fixed integer matrix (the monomial coefficients of each Chebyshev
  // polynomial), applied once here.
  //
  // Two guards.  Monomial coefficients of a re-expressed Chebyshev series can
  // be large and alternating, and what the cancellation costs is the ratio of
  // their magnitudes; measured on real parameter files it stays near 1e3,
  // which is three digits of the sixteen available, but a pathological set is
  // refused rather than evaluated wrongly.  And a node holding one high power
  // would pay Horner's full walk up to it for a single term, so trees whose
  // rows are mostly empty keep their Chebyshev leaves.

  if (npairs == 3) {
    int maxpow = 0;

    for (int c = 0; c < ncoeffs; c++)
      if (g.leaf_pow[c] > maxpow) maxpow = g.leaf_pow[c];

    const int dim = maxpow + 1;

    vector<double> M((size_t) dim * dim, 0.0);

    M[0] = 1.0;

    if (dim > 1) M[1 * dim + 1] = 1.0;

    for (int pp = 2; pp < dim; pp++) {
      for (int k = 1; k < dim; k++)
        M[(size_t) pp * dim + k] = 2.0 * M[(size_t) (pp - 1) * dim + k - 1];

      for (int k = 0; k < dim; k++) M[(size_t) pp * dim + k] -= M[(size_t) (pp - 2) * dim + k];
    }

    const int ndeep = (int) ranges.size();

    g.mono_start.assign(ndeep + 1, 0);

    double amp_num = 0.0, amp_den = 0.0;
    long rowsum = 0;

    vector<double> rows;

    for (int r = 0; r < ndeep; r++) {
      const int c0 = g.level_start[nlevels - 1][r];
      const int c1 = g.level_start[nlevels - 1][r + 1];

      int rowlen = 0;

      for (int c = c0; c < c1; c++) rowlen = std::max(rowlen, g.leaf_pow[c] + 1);

      const size_t base = rows.size();

      rows.resize(base + rowlen, 0.0);

      for (int c = c0; c < c1; c++) {
        const double v = g.leaf_c[c];
        const int pp = g.leaf_pow[c];

        amp_den += fabs(v);

        for (int k = 0; k <= pp; k++) rows[base + k] += v * M[(size_t) pp * dim + k];
      }

      for (int k = 0; k < rowlen; k++) amp_num += fabs(rows[base + k]);

      rowsum += rowlen;
      g.mono_start[r + 1] = (int) rows.size();
    }

    // Horner's descent visits every power up to the row's highest whether a
    // coefficient sits there or not, and its multiply-add chain is serial where
    // the sparse leaf loop is not.  Measured across both benchmark models, the
    // crossover sits near rows that are mostly full: silicon's rows carry a
    // coefficient in almost every slot and gain twenty percent, while a
    // multi-element model at two-fifths fill loses ten.

    const bool dense_enough = rowsum <= (long) (1.3 * ncoeffs);
    const bool well_conditioned = amp_num <= 1.0e5 * amp_den;

    if (dense_enough && well_conditioned)
      g.mono_c.swap(rows);
    else
      g.mono_start.clear();
  }

  // The tree only pays off when its nodes actually have several children.  A
  // node costs roughly NODE_COST instructions of loop and index bookkeeping on
  // top of the two operations per accumulator it carries, so a tree that is
  // mostly a chain does strictly more work than the flat loop.  That is exactly
  // what happens to the 4-body term of a typical model: a few dozen sparse
  // coefficients spread over five levels leave almost every group a singleton.
  // Estimate both and keep the tree only when it is clearly ahead.

  // The flat loop spends about npairs multiplies on the energy term and another
  // npairs on each of the npairs derivatives, plus the accumulates: 12
  // operations per coefficient at npairs = 3 and 39 at npairs = 6, which
  // npairs^2 + 3 reproduces.

  const double NODE_COST = 6.0;
  const double flat_cost = (double) ncoeffs * (npairs * npairs + 3);

  double grouped_cost = 4.0 * ncoeffs;

  for (int d = nlevels - 1; d >= 0; d--) {
    const double nacc = nlevels - d + 1;    // accumulators carried out of level d

    grouped_cost += (double) g.level_pow[d].size() * (2.0 * nacc + NODE_COST);
  }

  if (grouped_cost * 1.25 > flat_cost) {
    grouped_pool.pop_back();
    return -1;
  }

  return slot;
}

CHIMES_VECTOR_CLONES
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

CHIMES_VECTOR_CLONES
void chimesFF::poly_3B(double *e, double *f, const chimesPolySet &ps, vector<double> &Tn_ij,
                       vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                       vector<double> &Tnd_ik, vector<double> &Tnd_jk)
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  const int ncoeffs = ps.ncoeffs;
  const double *const params = ps.params;
  const int *pow = ps.powers;

  const double *const tij = Tn_ij.data();
  const double *const tik = Tn_ik.data();
  const double *const tjk = Tn_jk.data();
  const double *const dij = Tnd_ij.data();
  const double *const dik = Tnd_ik.data();
  const double *const djk = Tnd_jk.data();

  *e = 0.0;
  f[0] = 0.0;
  f[1] = 0.0;
  f[2] = 0.0;

  for (int coeffs = 0; coeffs < ncoeffs; coeffs++, pow += 3) {
    const double coeff = params[coeffs];

    const double t0 = tij[pow[0]];
    const double t1 = tik[pow[1]];
    const double t2 = tjk[pow[2]];

    *e += coeff * t0 * t1 * t2;

    f[0] += coeff * dij[pow[0]] * t1 * t2;
    f[1] += coeff * dik[pow[1]] * t0 * t2;
    f[2] += coeff * djk[pow[2]] * t0 * t1;
  }
}

// exp() for a whole batch of lanes.
//
// The Morse transform needs one exponential per lane, and a call to libm's exp
// is a call: the compiler cannot vectorize across it, so a batch of eight pays
// eight scalar exps and they are the single largest item in the setup.  The
// arguments here are not general, though -- the argument is -r/lambda with r
// inside the outer cutoff -- so the textbook range reduction plus a Taylor
// polynomial is enough, and being straight-line arithmetic it goes through the
// vector unit with the rest of the lane loops.
//
// exp(y) = 2^k * exp(t), k = round(y*log2(e)), t = y - k*ln2 so |t| <= ln2/2.
// The remainder of the degree-12 Taylor series of exp(t) over that interval is
// below 1e-16 relative; measured against libm over 34 million points spanning
// the range this code actually asks about, the worst relative difference is
// 4.7e-16, or about four ulp, and over all of [-700,700] it is 3.2e-16.
// Anything outside the safe exponent range falls back to libm for the whole
// batch, which never happens for a physically sensible parameter file.

namespace {

// ln2 split so that k*LN2_HI is exact for every k this can produce

constexpr double CHEXP_LOG2E = 1.4426950408889634074;
constexpr double CHEXP_LN2_HI = 6.93147180369123816490e-01;
constexpr double CHEXP_LN2_LO = 1.90821492927058770002e-10;

// 2^52 + 2^51, the smallest value whose addition discards every fractional bit
// of a double in round-to-nearest

constexpr double CHEXP_ROUND_MAGIC = 6755399441055744.0;

constexpr double CHEXP_C[13] = {1.0,
                                1.0,
                                1.0 / 2.0,
                                1.0 / 6.0,
                                1.0 / 24.0,
                                1.0 / 120.0,
                                1.0 / 720.0,
                                1.0 / 5040.0,
                                1.0 / 40320.0,
                                1.0 / 362880.0,
                                1.0 / 3628800.0,
                                1.0 / 39916800.0,
                                1.0 / 479001600.0};

inline void chimes_exp_batch(double *out, const double *y)
{
  // The range guard is a reduction over the lanes, so writing it as a flag set
  // inside the loop makes the loop carry a dependency and stops it vectorizing.
  // A running maximum of the magnitudes carries no dependency the vector unit
  // cannot handle and settles the question with one comparison at the end.

  // Written as comparisons rather than fmax: fmax has NaN semantics the vector
  // maximum does not, so unless the build promises otherwise the compiler emits
  // a library call for it, which is exactly what this loop is trying to avoid.

  double amax = 0.0;

  for (int l = 0; l < CHIMES_VLEN; l++) {
    const double a = (y[l] < 0.0) ? -y[l] : y[l];

    amax = (a > amax) ? a : amax;
  }

  if (amax > 700.0) {
    for (int l = 0; l < CHIMES_VLEN; l++) out[l] = exp(y[l]);
    return;
  }

  int ki[CHIMES_VLEN];
  double t[CHIMES_VLEN], p[CHIMES_VLEN], scale[CHIMES_VLEN];
  uint64_t bits[CHIMES_VLEN];

  // Round to nearest without a branch or a select.  Adding a constant large
  // enough that the fractional bits fall off the end of the significand, then
  // subtracting it again, leaves the nearest integer; the guard above bounds
  // the argument far below the point where that breaks down.

  for (int l = 0; l < CHIMES_VLEN; l++) {
    const double s = y[l] * CHEXP_LOG2E;
    const double z = s + CHEXP_ROUND_MAGIC;

    ki[l] = (int) (z - CHEXP_ROUND_MAGIC);
  }

  for (int l = 0; l < CHIMES_VLEN; l++) {
    const double kf = (double) ki[l];

    t[l] = (y[l] - kf * CHEXP_LN2_HI) - kf * CHEXP_LN2_LO;
  }

  for (int l = 0; l < CHIMES_VLEN; l++) p[l] = CHEXP_C[12];

  for (int i = 11; i >= 0; i--)
    for (int l = 0; l < CHIMES_VLEN; l++) p[l] = p[l] * t[l] + CHEXP_C[i];

  // 2^k straight from the exponent field.  k is bounded by the guard above, so
  // the field never runs off either end.

  for (int l = 0; l < CHIMES_VLEN; l++)
    bits[l] = ((uint64_t) (uint32_t) (ki[l] + 1023)) << 52;

  memcpy(scale, bits, sizeof(bits));

  for (int l = 0; l < CHIMES_VLEN; l++) out[l] = scale[l] * p[l];
}

}    // namespace

// Chebyshev values for one pair slot across a batch, written lane-minor.  The
// recurrence is a plain lane loop, and so, now, is the exponential.

CHIMES_VECTOR_CLONES
void chimesFF::set_cheby_polys_batch(double *Tn, double *Tnd, const double *dx,
                                     const chimesSlotConst &sc, const int bodyness)
{
  const int order = poly_orders[bodyness];
  const int dim = order + 1;

  double x[CHIMES_VLEN], dx_dr[CHIMES_VLEN];
  double arg[CHIMES_VLEN], exprlen[CHIMES_VLEN];

  // As in the exponential: clamping, and asking afterwards whether any lane was
  // clamped, keeps this loop free of the flag that would serialize it.

  double dmin = dx[0];

  for (int l = 0; l < CHIMES_VLEN; l++) {
    dmin = (dx[l] < dmin) ? dx[l] : dmin;

    arg[l] = ((dx[l] > sc.inner) ? dx[l] : sc.inner) * sc.neg_inv_morse;
  }

  const bool any_short = (dmin < sc.inner);

  chimes_exp_batch(exprlen, arg);

  for (int l = 0; l < CHIMES_VLEN; l++) {
    x[l] = (exprlen[l] - sc.x_avg) * sc.inv_x_diff;
    dx_dr[l] = exprlen[l] * sc.dxdr_scale;
  }


  for (int l = 0; l < CHIMES_VLEN; l++) {
    Tn[l] = 1.0;
    Tnd[l] = 1.0;
  }
  for (int l = 0; l < CHIMES_VLEN; l++) {
    Tn[CHIMES_VLEN + l] = x[l];
    Tnd[CHIMES_VLEN + l] = 2.0 * x[l];
  }

  for (int i = 2; i < dim; i++)
    for (int l = 0; l < CHIMES_VLEN; l++) {
      const double x2 = 2.0 * x[l];

      Tn[i * CHIMES_VLEN + l] = x2 * Tn[(i - 1) * CHIMES_VLEN + l] - Tn[(i - 2) * CHIMES_VLEN + l];
      Tnd[i * CHIMES_VLEN + l] =
          x2 * Tnd[(i - 1) * CHIMES_VLEN + l] - Tnd[(i - 2) * CHIMES_VLEN + l];
    }

  for (int i = order; i >= 1; i--)
    for (int l = 0; l < CHIMES_VLEN; l++)
      Tnd[i * CHIMES_VLEN + l] = i * dx_dr[l] * Tnd[(i - 1) * CHIMES_VLEN + l];

  for (int l = 0; l < CHIMES_VLEN; l++) Tnd[l] = 0.0;

  // A separation inside the inner cutoff needs the damped form, which is rare
  // enough that those lanes are simply redone one at a time.

  if (any_short) {
    vector<double> tn(dim), tnd(dim);

    for (int l = 0; l < CHIMES_VLEN; l++) {
      if (dx[l] >= sc.inner) continue;

      set_cheby_polys(tn, tnd, dx[l], sc, bodyness);

      for (int i = 0; i < dim; i++) {
        Tn[i * CHIMES_VLEN + l] = tn[i];
        Tnd[i * CHIMES_VLEN + l] = tnd[i];
      }
    }
  }
}

CHIMES_VECTOR_CLONES
void chimesFF::compute_3B_batch(const int nlane, const int type_idx,
                                const double dx[3][CHIMES_VLEN], chimes3BBatch &b)
{
  const chimesSlotConst *sc = &slot_3b[type_idx * 3];

  // A lane of the last pair inside the inner cutoff carries the damped form,
  // which only the Chebyshev arrays represent; the Horner leaves must know.

  b.any_short = false;

  for (int l = 0; l < CHIMES_VLEN; l++)
    if (dx[2][l] < sc[2].inner) b.any_short = true;

  // The caller turns each cluster's polynomial derivative into a force by
  // dividing by the separation.  Done there it is one scalar division per pair
  // per cluster; done here it is a lane loop, so the whole block goes through
  // one vector divide.

  for (int p = 0; p < 3; p++) {
    set_cheby_polys_batch(b.Tn[p].data(), b.Tnd[p].data(), dx[p], sc[p], 1);

    for (int l = 0; l < CHIMES_VLEN; l++) get_fcut(dx[p][l], sc[p], b.fcut[p][l], b.fcutderiv[p][l]);

    for (int l = 0; l < CHIMES_VLEN; l++) b.inv_dx[p][l] = 1.0 / dx[p][l];
  }

  const chimesPolySet &ps = poly_3b_set[type_idx];

  if (ps.grouped) {
    if (!ps.grouped->mono_start.empty() && !b.any_short)
      poly_3B_horner_batch(*ps.grouped, b);
    else
      poly_3B_grouped_batch(*ps.grouped, b);
    return;
  }

  // No coefficient tree for this type: fall back to the flat kernel per lane,
  // reading the batch's Chebyshev values back out of their lane-minor layout.

  vector<double> tij(b.dim), tik(b.dim), tjk(b.dim), dij(b.dim), dik(b.dim), djk(b.dim);

  for (int l = 0; l < nlane; l++) {
    for (int i = 0; i < b.dim; i++) {
      tij[i] = b.Tn[0][i * CHIMES_VLEN + l];
      tik[i] = b.Tn[1][i * CHIMES_VLEN + l];
      tjk[i] = b.Tn[2][i * CHIMES_VLEN + l];
      dij[i] = b.Tnd[0][i * CHIMES_VLEN + l];
      dik[i] = b.Tnd[1][i * CHIMES_VLEN + l];
      djk[i] = b.Tnd[2][i * CHIMES_VLEN + l];
    }

    double f[3];

    poly_3B(&b.poly[l], f, ps, tij, tik, tjk, dij, dik, djk);

    for (int p = 0; p < 3; p++) b.dpoly[p][l] = f[p];
  }
}

// The coefficient tree, walked once for the whole batch.  The traversal is
// identical for every lane because they share a cluster type, so the tree
// indices stay scalar and only the arithmetic is per lane -- which is what
// makes the innermost load contiguous.

// The same walk as poly_3B_grouped_batch with the leaf level in the monomial
// basis: each deepest node's series over the last pair is a plain polynomial,
// taken by Horner's rule with its derivative in the same descent.  The leaf
// reads no basis arrays at all -- the transformed coordinate is row 1 of the
// batch (T_1(x) = x, and row 1 of the derivative array is dx/dr), copied to
// locals once and register-resident for the whole tree.  A separate function
// on purpose: a branch inside the shared evaluator's inner loop cost the
// models that never take it seven percent.

CHIMES_VECTOR_CLONES
void chimesFF::poly_3B_horner_batch(const chimesGroupedPoly &g, chimes3BBatch &b)
{
  const int *const l0_pow = g.level_pow[0].data();
  const int *const l0_start = g.level_start[0].data();
  const int *const l1_pow = g.level_pow[1].data();
  const int *const l1_start = g.level_start[1].data();
  const double *const mono_c = g.mono_c.data();
  const int *const mono_start = g.mono_start.data();

  const double *const tij = b.Tn[0].data();
  const double *const tik = b.Tn[1].data();
  const double *const dij = b.Tnd[0].data();
  const double *const dik = b.Tnd[1].data();

  const int n0 = g.level_pow[0].size();

  const int half = CHIMES_VLEN / 2;

  for (int lo = 0; lo < CHIMES_VLEN; lo += half) {
    double xv[half], xd[half];

    for (int l = 0; l < half; l++) {
      xv[l] = b.Tn[2][CHIMES_VLEN + lo + l];
      xd[l] = b.Tnd[2][CHIMES_VLEN + lo + l];
    }

    double E[half], F0[half], F1[half], F2[half];

    for (int l = 0; l < half; l++) E[l] = F0[l] = F1[l] = F2[l] = 0.0;

    for (int a = 0; a < n0; a++) {
      const double *const t0 = tij + (size_t) l0_pow[a] * CHIMES_VLEN + lo;
      const double *const d0 = dij + (size_t) l0_pow[a] * CHIMES_VLEN + lo;

      double A[half], A1[half], A2[half];

      for (int l = 0; l < half; l++) A[l] = A1[l] = A2[l] = 0.0;

      for (int bb = l0_start[a]; bb < l0_start[a + 1]; bb++) {
        const double *const t1 = tik + (size_t) l1_pow[bb] * CHIMES_VLEN + lo;
        const double *const d1 = dik + (size_t) l1_pow[bb] * CHIMES_VLEN + lo;

        double V[half], D[half];

        for (int l = 0; l < half; l++) V[l] = D[l] = 0.0;

        for (int k = mono_start[bb + 1] - 1; k >= mono_start[bb]; k--) {
          const double rk = mono_c[k];

          for (int l = 0; l < half; l++) {
            D[l] = D[l] * xv[l] + V[l];
            V[l] = V[l] * xv[l] + rk;
          }
        }

        for (int l = 0; l < half; l++) {
          const double S2 = D[l] * xd[l];

          A[l] += t1[l] * V[l];
          A1[l] += d1[l] * V[l];
          A2[l] += t1[l] * S2;
        }
      }

      for (int l = 0; l < half; l++) {
        E[l] += t0[l] * A[l];
        F0[l] += d0[l] * A[l];
        F1[l] += t0[l] * A1[l];
        F2[l] += t0[l] * A2[l];
      }
    }

    for (int l = 0; l < half; l++) {
      b.poly[lo + l] = E[l];
      b.dpoly[0][lo + l] = F0[l];
      b.dpoly[1][lo + l] = F1[l];
      b.dpoly[2][lo + l] = F2[l];
    }
  }
}

CHIMES_VECTOR_CLONES
void chimesFF::poly_3B_grouped_batch(const chimesGroupedPoly &g, chimes3BBatch &b)
{
  const int *const l0_pow = g.level_pow[0].data();
  const int *const l0_start = g.level_start[0].data();
  const int *const l1_pow = g.level_pow[1].data();
  const int *const l1_start = g.level_start[1].data();
  const int *const leaf_pow = g.leaf_pow.data();
  const double *const leaf_c = g.leaf_c.data();

  const double *const tij = b.Tn[0].data();
  const double *const tik = b.Tn[1].data();
  const double *const tjk = b.Tn[2].data();
  const double *const dij = b.Tnd[0].data();
  const double *const dik = b.Tnd[1].data();
  const double *const djk = b.Tnd[2].data();

  const int n0 = g.level_pow[0].size();

  const int half = CHIMES_VLEN / 2;

  for (int lo = 0; lo < CHIMES_VLEN; lo += half) {

    double E[half], F0[half], F1[half], F2[half];

    for (int l = 0; l < half; l++) E[l] = F0[l] = F1[l] = F2[l] = 0.0;

    for (int a = 0; a < n0; a++) {
      const double *const t0 = tij + (size_t) l0_pow[a] * CHIMES_VLEN + lo;
      const double *const d0 = dij + (size_t) l0_pow[a] * CHIMES_VLEN + lo;

      double A[half], A1[half], A2[half];

      for (int l = 0; l < half; l++) A[l] = A1[l] = A2[l] = 0.0;

      for (int bb = l0_start[a]; bb < l0_start[a + 1]; bb++) {
        const double *const t1 = tik + (size_t) l1_pow[bb] * CHIMES_VLEN + lo;
        const double *const d1 = dik + (size_t) l1_pow[bb] * CHIMES_VLEN + lo;

        double S[half], S2[half];

        for (int l = 0; l < half; l++) S[l] = S2[l] = 0.0;

        for (int c = l1_start[bb]; c < l1_start[bb + 1]; c++) {
          const double coeff = leaf_c[c];
          const double *const t2 = tjk + (size_t) leaf_pow[c] * CHIMES_VLEN + lo;
          const double *const d2 = djk + (size_t) leaf_pow[c] * CHIMES_VLEN + lo;

          for (int l = 0; l < half; l++) {
            S[l] += coeff * t2[l];
            S2[l] += coeff * d2[l];
          }
        }

        for (int l = 0; l < half; l++) {
          A[l] += t1[l] * S[l];
          A1[l] += d1[l] * S[l];
          A2[l] += t1[l] * S2[l];
        }
      }

      for (int l = 0; l < half; l++) {
        E[l] += t0[l] * A[l];
        F0[l] += d0[l] * A[l];
        F1[l] += t0[l] * A1[l];
        F2[l] += t0[l] * A2[l];
      }
    }

    for (int l = 0; l < half; l++) {
      b.poly[lo + l] = E[l];
      b.dpoly[0][lo + l] = F0[l];
      b.dpoly[1][lo + l] = F1[l];
      b.dpoly[2][lo + l] = F2[l];
    }
  }
}

CHIMES_VECTOR_CLONES
void chimesFF::poly_3B_grouped(double *e, double *f, const chimesGroupedPoly &g,
                               vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_jk,
                               vector<double> &Tnd_ij, vector<double> &Tnd_ik,
                               vector<double> &Tnd_jk)
// Same sum as poly_3B, evaluated over the coefficient tree.  Coefficients that
// share p_ij and p_ik share the products of those two Chebyshev factors, so the
// leaf costs two multiply-adds instead of the twelve multiplies the flat loop
// needed.  The summation order changes, so this is not bit-for-bit poly_3B.
{
  const int *const l0_pow = g.level_pow[0].data();
  const int *const l0_start = g.level_start[0].data();
  const int *const l1_pow = g.level_pow[1].data();
  const int *const l1_start = g.level_start[1].data();
  const int *const leaf_pow = g.leaf_pow.data();
  const double *const leaf_c = g.leaf_c.data();

  const double *const tij = Tn_ij.data();
  const double *const tik = Tn_ik.data();
  const double *const tjk = Tn_jk.data();
  const double *const dij = Tnd_ij.data();
  const double *const dik = Tnd_ik.data();
  const double *const djk = Tnd_jk.data();

  const int n0 = g.level_pow[0].size();

  double E = 0.0, F0 = 0.0, F1 = 0.0, F2 = 0.0;

  for (int a = 0; a < n0; a++) {
    const int p0 = l0_pow[a];
    const double t0 = tij[p0];
    const double d0 = dij[p0];

    double A = 0.0, A1 = 0.0, A2 = 0.0;

    for (int b = l0_start[a]; b < l0_start[a + 1]; b++) {
      const int p1 = l1_pow[b];
      const double t1 = tik[p1];
      const double d1 = dik[p1];

      double S = 0.0, S2 = 0.0;

      for (int c = l1_start[b]; c < l1_start[b + 1]; c++) {
        const double coeff = leaf_c[c];
        const int p2 = leaf_pow[c];

        S += coeff * tjk[p2];
        S2 += coeff * djk[p2];
      }

      A += t1 * S;
      A1 += d1 * S;
      A2 += t1 * S2;
    }

    E += t0 * A;
    F0 += d0 * A;
    F1 += t0 * A1;
    F2 += t0 * A2;
  }

  *e = E;
  f[0] = F0;
  f[1] = F1;
  f[2] = F2;
}

// The batched form of poly_4B_grouped.  Same six-level tree, evaluated for
// CHIMES_VLEN clusters at once: every accumulator becomes a lane vector and
// every Chebyshev value becomes a contiguous run of CHIMES_VLEN doubles, so
// each level's update is a flat lane loop rather than a scalar chain.  The
// scalar kernel is kept for the callers that hold one cluster.

CHIMES_VECTOR_CLONES
void chimesFF::poly_4B_grouped_batch(const chimesGroupedPoly &g, chimes4BBatch &b)
{
  const int *const lp0 = g.level_pow[0].data();
  const int *const lp1 = g.level_pow[1].data();
  const int *const lp2 = g.level_pow[2].data();
  const int *const lp3 = g.level_pow[3].data();
  const int *const lp4 = g.level_pow[4].data();

  const int *const ls0 = g.level_start[0].data();
  const int *const ls1 = g.level_start[1].data();
  const int *const ls2 = g.level_start[2].data();
  const int *const ls3 = g.level_start[3].data();
  const int *const ls4 = g.level_start[4].data();

  const int *const leaf_pow = g.leaf_pow.data();
  const double *const leaf_c = g.leaf_c.data();

  const double *const tij = b.Tn[0].data();
  const double *const tik = b.Tn[1].data();
  const double *const til = b.Tn[2].data();
  const double *const tjk = b.Tn[3].data();
  const double *const tjl = b.Tn[4].data();
  const double *const tkl = b.Tn[5].data();

  const double *const dij = b.Tnd[0].data();
  const double *const dik = b.Tnd[1].data();
  const double *const dil = b.Tnd[2].data();
  const double *const djk = b.Tnd[3].data();
  const double *const djl = b.Tnd[4].data();
  const double *const dkl = b.Tnd[5].data();

  const int n0 = g.level_pow[0].size();

  double E[CHIMES_VLEN], F[6][CHIMES_VLEN];

  for (int l = 0; l < CHIMES_VLEN; l++) {
    E[l] = 0.0;

    for (int p = 0; p < 6; p++) F[p][l] = 0.0;
  }

  for (int a = 0; a < n0; a++) {
    const double *const t0 = tij + (size_t) lp0[a] * CHIMES_VLEN;
    const double *const d0 = dij + (size_t) lp0[a] * CHIMES_VLEN;

    double D[CHIMES_VLEN], D1[CHIMES_VLEN], D2[CHIMES_VLEN];
    double D3[CHIMES_VLEN], D4[CHIMES_VLEN], D5[CHIMES_VLEN];

    for (int l = 0; l < CHIMES_VLEN; l++)
      D[l] = D1[l] = D2[l] = D3[l] = D4[l] = D5[l] = 0.0;

    for (int bb = ls0[a]; bb < ls0[a + 1]; bb++) {
      const double *const t1 = tik + (size_t) lp1[bb] * CHIMES_VLEN;
      const double *const d1 = dik + (size_t) lp1[bb] * CHIMES_VLEN;

      double C[CHIMES_VLEN], C2[CHIMES_VLEN], C3[CHIMES_VLEN];
      double C4[CHIMES_VLEN], C5[CHIMES_VLEN];

      for (int l = 0; l < CHIMES_VLEN; l++) C[l] = C2[l] = C3[l] = C4[l] = C5[l] = 0.0;

      for (int c = ls1[bb]; c < ls1[bb + 1]; c++) {
        const double *const t2 = til + (size_t) lp2[c] * CHIMES_VLEN;
        const double *const d2 = dil + (size_t) lp2[c] * CHIMES_VLEN;

        double B[CHIMES_VLEN], B3[CHIMES_VLEN], B4[CHIMES_VLEN], B5[CHIMES_VLEN];

        for (int l = 0; l < CHIMES_VLEN; l++) B[l] = B3[l] = B4[l] = B5[l] = 0.0;

        for (int m = ls2[c]; m < ls2[c + 1]; m++) {
          const double *const t3 = tjk + (size_t) lp3[m] * CHIMES_VLEN;
          const double *const d3 = djk + (size_t) lp3[m] * CHIMES_VLEN;

          double A[CHIMES_VLEN], A4[CHIMES_VLEN], A5[CHIMES_VLEN];

          for (int l = 0; l < CHIMES_VLEN; l++) A[l] = A4[l] = A5[l] = 0.0;

          for (int n = ls3[m]; n < ls3[m + 1]; n++) {
            const double *const t4 = tjl + (size_t) lp4[n] * CHIMES_VLEN;
            const double *const d4 = djl + (size_t) lp4[n] * CHIMES_VLEN;

            double S[CHIMES_VLEN], S5[CHIMES_VLEN];

            for (int l = 0; l < CHIMES_VLEN; l++) S[l] = S5[l] = 0.0;

            for (int q = ls4[n]; q < ls4[n + 1]; q++) {
              const double coeff = leaf_c[q];
              const double *const t5 = tkl + (size_t) leaf_pow[q] * CHIMES_VLEN;
              const double *const d5 = dkl + (size_t) leaf_pow[q] * CHIMES_VLEN;

              for (int l = 0; l < CHIMES_VLEN; l++) {
                S[l] += coeff * t5[l];
                S5[l] += coeff * d5[l];
              }
            }

            for (int l = 0; l < CHIMES_VLEN; l++) {
              A[l] += t4[l] * S[l];
              A4[l] += d4[l] * S[l];
              A5[l] += t4[l] * S5[l];
            }
          }

          for (int l = 0; l < CHIMES_VLEN; l++) {
            B[l] += t3[l] * A[l];
            B3[l] += d3[l] * A[l];
            B4[l] += t3[l] * A4[l];
            B5[l] += t3[l] * A5[l];
          }
        }

        for (int l = 0; l < CHIMES_VLEN; l++) {
          C[l] += t2[l] * B[l];
          C2[l] += d2[l] * B[l];
          C3[l] += t2[l] * B3[l];
          C4[l] += t2[l] * B4[l];
          C5[l] += t2[l] * B5[l];
        }
      }

      for (int l = 0; l < CHIMES_VLEN; l++) {
        D[l] += t1[l] * C[l];
        D1[l] += d1[l] * C[l];
        D2[l] += t1[l] * C2[l];
        D3[l] += t1[l] * C3[l];
        D4[l] += t1[l] * C4[l];
        D5[l] += t1[l] * C5[l];
      }
    }

    for (int l = 0; l < CHIMES_VLEN; l++) {
      E[l] += t0[l] * D[l];
      F[0][l] += d0[l] * D[l];
      F[1][l] += t0[l] * D1[l];
      F[2][l] += t0[l] * D2[l];
      F[3][l] += t0[l] * D3[l];
      F[4][l] += t0[l] * D4[l];
      F[5][l] += t0[l] * D5[l];
    }
  }

  for (int l = 0; l < CHIMES_VLEN; l++) {
    b.poly[l] = E[l];

    for (int p = 0; p < 6; p++) b.dpoly[p][l] = F[p][l];
  }
}

CHIMES_VECTOR_CLONES
void chimesFF::poly_4B_grouped(double *e, double *f, const chimesGroupedPoly &g,
                               vector<double> &Tn_ij, vector<double> &Tn_ik, vector<double> &Tn_il,
                               vector<double> &Tn_jk, vector<double> &Tn_jl, vector<double> &Tn_kl,
                               vector<double> &Tnd_ij, vector<double> &Tnd_ik,
                               vector<double> &Tnd_il, vector<double> &Tnd_jk,
                               vector<double> &Tnd_jl, vector<double> &Tnd_kl)
// The 4-body analogue: five nested group levels over (ij, ik, il, jk, jl) and a
// leaf over kl.  Each level multiplies the accumulators coming from below by
// its own Chebyshev value and adds one more for its own derivative.
{
  const double *const tij = Tn_ij.data();
  const double *const tik = Tn_ik.data();
  const double *const til = Tn_il.data();
  const double *const tjk = Tn_jk.data();
  const double *const tjl = Tn_jl.data();
  const double *const tkl = Tn_kl.data();
  const double *const dij = Tnd_ij.data();
  const double *const dik = Tnd_ik.data();
  const double *const dil = Tnd_il.data();
  const double *const djk = Tnd_jk.data();
  const double *const djl = Tnd_jl.data();
  const double *const dkl = Tnd_kl.data();

  const int *const lp0 = g.level_pow[0].data();
  const int *const lp1 = g.level_pow[1].data();
  const int *const lp2 = g.level_pow[2].data();
  const int *const lp3 = g.level_pow[3].data();
  const int *const lp4 = g.level_pow[4].data();
  const int *const ls0 = g.level_start[0].data();
  const int *const ls1 = g.level_start[1].data();
  const int *const ls2 = g.level_start[2].data();
  const int *const ls3 = g.level_start[3].data();
  const int *const ls4 = g.level_start[4].data();
  const int *const leaf_pow = g.leaf_pow.data();
  const double *const leaf_c = g.leaf_c.data();

  const int n0 = g.level_pow[0].size();

  double E = 0.0, F[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int a = 0; a < n0; a++) {
    const int p0 = lp0[a];
    const double t0 = tij[p0];
    const double d0 = dij[p0];

    double D = 0.0, D1 = 0.0, D2 = 0.0, D3 = 0.0, D4 = 0.0, D5 = 0.0;

    for (int b = ls0[a]; b < ls0[a + 1]; b++) {
      const int p1 = lp1[b];
      const double t1 = tik[p1];
      const double d1 = dik[p1];

      double C = 0.0, C2 = 0.0, C3 = 0.0, C4 = 0.0, C5 = 0.0;

      for (int c = ls1[b]; c < ls1[b + 1]; c++) {
        const int p2 = lp2[c];
        const double t2 = til[p2];
        const double d2 = dil[p2];

        double B = 0.0, B3 = 0.0, B4 = 0.0, B5 = 0.0;

        for (int m = ls2[c]; m < ls2[c + 1]; m++) {
          const int p3 = lp3[m];
          const double t3 = tjk[p3];
          const double d3 = djk[p3];

          double A = 0.0, A4 = 0.0, A5 = 0.0;

          for (int n = ls3[m]; n < ls3[m + 1]; n++) {
            const int p4 = lp4[n];
            const double t4 = tjl[p4];
            const double d4 = djl[p4];

            double S = 0.0, S5 = 0.0;

            for (int q = ls4[n]; q < ls4[n + 1]; q++) {
              const double coeff = leaf_c[q];
              const int p5 = leaf_pow[q];

              S += coeff * tkl[p5];
              S5 += coeff * dkl[p5];
            }

            A += t4 * S;
            A4 += d4 * S;
            A5 += t4 * S5;
          }

          B += t3 * A;
          B3 += d3 * A;
          B4 += t3 * A4;
          B5 += t3 * A5;
        }

        C += t2 * B;
        C2 += d2 * B;
        C3 += t2 * B3;
        C4 += t2 * B4;
        C5 += t2 * B5;
      }

      D += t1 * C;
      D1 += d1 * C;
      D2 += t1 * C2;
      D3 += t1 * C3;
      D4 += t1 * C4;
      D5 += t1 * C5;
    }

    E += t0 * D;
    F[0] += d0 * D;
    F[1] += t0 * D1;
    F[2] += t0 * D2;
    F[3] += t0 * D3;
    F[4] += t0 * D4;
    F[5] += t0 * D5;
  }

  *e = E;
  for (int p = 0; p < 6; p++) f[p] = F[p];
}

void chimesFF::poly_3B_dense(double &e, double &f0, double &f1, double &f2, int ncoeffs_3b,
                             vector<double> &chimes_3b_params, vector<double> &Tn_ij,
                             vector<double> &Tn_ik, vector<double> &Tn_jk, vector<double> &Tnd_ij,
                             vector<double> &Tnd_ik, vector<double> &Tnd_jk)
// Compute the 3 body polynomial (e) and derivatives with respect to each pair distance (f0, f1, f2)
// (LEF) 4/02/26
{
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

// The batched form of poly_4B.  Same arithmetic, six pairs at a time across
// CHIMES_VLEN clusters of one type: because the batch stores its Chebyshev
// values lane-minor, a coefficient's power selects one contiguous run of
// doubles per pair rather than one scalar, so the lane loop is the shape a
// vector unit wants and the coefficient's power indices are loaded once for
// the whole batch instead of once per cluster.

CHIMES_VECTOR_CLONES
void chimesFF::poly_4B_batch(const chimesPolySet &ps, chimes4BBatch &b)
{
  const int ncoeffs = ps.ncoeffs;
  const double *const params = ps.params;

  // Seven accumulators -- the energy and one derivative per pair -- are live
  // across the whole coefficient loop, and at CHIMES_VLEN lanes that is fourteen
  // vector registers before anything else is counted.  The batch is therefore
  // worked in half-width groups, so the accumulators stay in registers and only
  // the Chebyshev values come from memory.  Coefficients are broadcast either
  // way, and the second group finds them all in L1.

  const int half = CHIMES_VLEN / 2;

  for (int lo = 0; lo < CHIMES_VLEN; lo += half) {
    const int *pw = ps.powers;

    double E[half], F[6][half];

    for (int l = 0; l < half; l++) {
      E[l] = 0.0;

      for (int p = 0; p < 6; p++) F[p][l] = 0.0;
    }

    for (int c = 0; c < ncoeffs; c++, pw += 6) {
      const double coeff = params[c];

      const double *const t0 = b.Tn[0].data() + (size_t) pw[0] * CHIMES_VLEN + lo;
      const double *const t1 = b.Tn[1].data() + (size_t) pw[1] * CHIMES_VLEN + lo;
      const double *const t2 = b.Tn[2].data() + (size_t) pw[2] * CHIMES_VLEN + lo;
      const double *const t3 = b.Tn[3].data() + (size_t) pw[3] * CHIMES_VLEN + lo;
      const double *const t4 = b.Tn[4].data() + (size_t) pw[4] * CHIMES_VLEN + lo;
      const double *const t5 = b.Tn[5].data() + (size_t) pw[5] * CHIMES_VLEN + lo;

      const double *const d0 = b.Tnd[0].data() + (size_t) pw[0] * CHIMES_VLEN + lo;
      const double *const d1 = b.Tnd[1].data() + (size_t) pw[1] * CHIMES_VLEN + lo;
      const double *const d2 = b.Tnd[2].data() + (size_t) pw[2] * CHIMES_VLEN + lo;
      const double *const d3 = b.Tnd[3].data() + (size_t) pw[3] * CHIMES_VLEN + lo;
      const double *const d4 = b.Tnd[4].data() + (size_t) pw[4] * CHIMES_VLEN + lo;
      const double *const d5 = b.Tnd[5].data() + (size_t) pw[5] * CHIMES_VLEN + lo;

      for (int l = 0; l < half; l++) {
        const double g0 = t0[l];
        const double g1 = t1[l];
        const double g2 = t2[l];
        const double g3 = t3[l];
        const double g4 = t4[l];
        const double g5 = t5[l];

        const double p1 = coeff * g0;
        const double p2 = p1 * g1;
        const double p3 = p2 * g2;
        const double p4 = p3 * g3;
        const double p5 = p4 * g4;

        const double s4 = g5;
        const double s3 = s4 * g4;
        const double s2 = s3 * g3;
        const double s1 = s2 * g2;
        const double s0 = s1 * g1;

        E[l] += p5 * g5;

        F[0][l] += coeff * d0[l] * s0;
        F[1][l] += p1 * d1[l] * s1;
        F[2][l] += p2 * d2[l] * s2;
        F[3][l] += p3 * d3[l] * s3;
        F[4][l] += p4 * d4[l] * s4;
        F[5][l] += p5 * d5[l];
      }
    }

    for (int l = 0; l < half; l++) {
      b.poly[lo + l] = E[l];

      for (int p = 0; p < 6; p++) b.dpoly[p][lo + l] = F[p][l];
    }
  }
}

// Set up and evaluate one batch of same-typed 4-body clusters.  Mirrors
// compute_3B_batch: the caller has already grouped the clusters by type and
// filled dx lane-minor, with unused lanes padded from lane 0.

CHIMES_VECTOR_CLONES
void chimesFF::compute_4B_batch(const int nlane, const int type_idx,
                                const double dx[6][CHIMES_VLEN], chimes4BBatch &b)
{
  const chimesSlotConst *sc = &slot_4b[type_idx * 6];

  for (int p = 0; p < 6; p++) {
    set_cheby_polys_batch(b.Tn[p].data(), b.Tnd[p].data(), dx[p], sc[p], 2);

    for (int l = 0; l < CHIMES_VLEN; l++) get_fcut(dx[p][l], sc[p], b.fcut[p][l], b.fcutderiv[p][l]);

    for (int l = 0; l < CHIMES_VLEN; l++) b.inv_dx[p][l] = 1.0 / dx[p][l];
  }

  const chimesPolySet &ps = poly_4b_set[type_idx];

  if (ps.grouped)
    poly_4B_grouped_batch(*ps.grouped, b);
  else
    poly_4B_batch(ps, b);
}

CHIMES_VECTOR_CLONES
void chimesFF::poly_4B(double *e, double *f, const chimesPolySet &ps, vector<double> &Tn_ij,
                       vector<double> &Tn_ik, vector<double> &Tn_il, vector<double> &Tn_jk,
                       vector<double> &Tn_jl, vector<double> &Tn_kl, vector<double> &Tnd_ij,
                       vector<double> &Tnd_ik, vector<double> &Tnd_il, vector<double> &Tnd_jk,
                       vector<double> &Tnd_jl, vector<double> &Tnd_kl)
// Compute the 4 body polynomial (e) and derivatives with respect to each pair distance (f)
// (LEF) 3/11/26
{
  const int ncoeffs = ps.ncoeffs;
  const double *const params = ps.params;
  const int *pow = ps.powers;

  const double *const tij = Tn_ij.data();
  const double *const tik = Tn_ik.data();
  const double *const til = Tn_il.data();
  const double *const tjk = Tn_jk.data();
  const double *const tjl = Tn_jl.data();
  const double *const tkl = Tn_kl.data();
  const double *const dij = Tnd_ij.data();
  const double *const dik = Tnd_ik.data();
  const double *const dil = Tnd_il.data();
  const double *const djk = Tnd_jk.data();
  const double *const djl = Tnd_jl.data();
  const double *const dkl = Tnd_kl.data();

  *e = 0;
  for (int i = 0; i < 6; i++) f[i] = 0.0;

  // Each term needs the product of all six Chebyshev factors for the energy,
  // and for each pair the product of the other five for its derivative.  Six
  // separate five-way products repeat almost all of the same multiplications,
  // so build them from one pass forward and one back: with the running
  // products p (everything before a pair, the coefficient folded in) and s
  // (everything after it), the derivative term for that pair is p*s times its
  // own derivative factor.  That is twenty-one multiplications per term where
  // the direct form takes thirty-two.

  for (int coeffs = 0; coeffs < ncoeffs; coeffs++, pow += 6) {
    const double t0 = tij[pow[0]];
    const double t1 = tik[pow[1]];
    const double t2 = til[pow[2]];
    const double t3 = tjk[pow[3]];
    const double t4 = tjl[pow[4]];
    const double t5 = tkl[pow[5]];

    const double p0 = params[coeffs];    // coeff
    const double p1 = p0 * t0;           // coeff * t0
    const double p2 = p1 * t1;
    const double p3 = p2 * t2;
    const double p4 = p3 * t3;
    const double p5 = p4 * t4;           // coeff * t0..t4

    const double s4 = t5;    // t5
    const double s3 = s4 * t4;
    const double s2 = s3 * t3;
    const double s1 = s2 * t2;
    const double s0 = s1 * t1;    // t1..t5

    *e += p5 * t5;

    f[0] += p0 * dij[pow[0]] * s0;
    f[1] += p1 * dik[pow[1]] * s1;
    f[2] += p2 * dil[pow[2]] * s2;
    f[3] += p3 * djk[pow[3]] * s3;
    f[4] += p4 * djl[pow[4]] * s4;
    f[5] += p5 * dkl[pow[5]];
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

  if (rank == 0) cout << "chimesFF: Maximum 3-B power found = " << max_pow3b << endl;

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

  if (rank == 0) cout << "chimesFF: Maximum 4-B power found = " << max_pow4b << endl;

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
