/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_cluster_switch_kokkos.h"
#include "modify.h"
#include "atom_kokkos.h"
#include "memory_kokkos.h"
#include "update.h"
#include "force.h"
#include "random_park.h" //random number generator
#include "atom_masks.h"

#include "error.h" //error
#include <string.h> //strcmp()
#include <math.h>
#include <utility>

#include "respa.h"
#include "neighbor.h" //neigh->build
#include "neigh_list_kokkos.h" //class NeighList
#include "neigh_request.h" //neigh->request
#include "atom.h" //per-atom variables
#include "pair.h" //force->pair
#include "bond.h" //force->bond
#include "group.h"//temperature compute
#include "compute.h" //temperature->compute_scalar
#include "domain.h"
#include "comm.h"
#include "utils.h"
#include <fstream> //read input files 
#include <string> //read input files 

using namespace LAMMPS_NS;
using namespace FixConst; //in fix.h, defines POST_FORCE, etc.

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixClusterSwitchKokkos<DeviceType>::FixClusterSwitchKokkos(LAMMPS *lmp, int narg, char **arg) : FixClusterSwitch(lmp,narg,arg)
{
  kokkosable = 1;
  //forward_comm_device = 1;
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixClusterSwitchKokkos<DeviceType>::~FixClusterSwitchKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::init()
{
  FixClusterSwitch::init();

  // adjust neighbor list request for KOKKOS

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::allocate()
{
  if (allocate_flag == 1) {
    memory->create(mol_restrict,maxmol+1,"fix:mol_restrict");
    memory->create(mol_state,maxmol+1,"fix:mol_state");
    memory->create(mol_accept,maxmol+1,"fix:mol_accept");
    memory->create(mol_cluster,maxmol+1,"fix:mol_cluster");
    //      memory->create(mol_atoms,maxmol+1,nSwitchTypes,"fix:mol_atoms");
    memory->create(mol_atoms,maxmol+1,nSwitchPerMol,"fix:mol_atoms");

    // no memory; initialize array such that every state is -1 to start

    for (int i = 0; i <= maxmol; i++) {
      mol_restrict[i] = -1;
      mol_state[i] = -1;
      mol_accept[i] = -1;
      //        for (int j = 0; j < nSwitchTypes; j++) {
      for (int j = 0; j < nSwitchPerMol; j++) {
        mol_atoms[i][j] = -1;
      }
    }
  } else if (allocate_flag == 2) {
    memory->create(atomtypesON,nSwitchTypes,"fix:atomtypesON");
    memory->create(atomtypesOFF,nSwitchTypes,"fix:atomtypesOFF");
  } else if (allocate_flag == 3) {
    memory->create(contactMap,nContactTypes,nAtomsPerContact,2,"fix:contactMap");
  }
}

/* ---------------------------------------------------------------------- */

//void FixClusterSwitchKokkos<DeviceType>::pre_force(int /*vflag*/)

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::pre_exchange()
{
  if (switchFreq == 0) return; 
  //if (update->ntimestep % switchFreq) return;
  if (next_reneighbor != update->ntimestep) return;

  domain->pbc();
  comm->exchange();
  comm->borders();
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  check_cluster();
  attempt_switch();

  next_reneighbor = update->ntimestep + switchFreq;
}

/* ---------------------------------------------------------------------- */

// Sends local atom data toward neighboring procs (into ghost atom data)
// Reverse sends ghost data to local data (e.g., forces)

template<class DeviceType>
int FixClusterSwitchKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int m;

  for (m = 0; m < n; m++) buf[m] = atom->type[list[m]];

  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  for (m = 0, i = first; m < n; m++, i++) atom->type[i] = static_cast<int> (buf[m]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::check_cluster()
{
  //  modify->clearstep_compute();

  mask = atomKK->k_mask.view<DeviceType>();
  molecule = atomKK->k_molecule.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  x = atomKK->k_x.view<DeviceType>();
  int nlocal = atom->nlocal;

  // invoke neighbor list (full)

  neighbor->build_one(list);
  int inum = list->inum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  int *mol_cluster_local = new int[maxmol+1];
  for (int i = 0; i <= maxmol; i++) {
    mol_cluster_local[i] = -1;
    mol_cluster[i] = -1;
  }

  //preset mol_seed and mol_offset
  mol_cluster_local[mol_seed] = mol_cluster[mol_seed] = mol_seed;
  mol_cluster_local[mol_seed-mol_offset] = mol_cluster[mol_seed-mol_offset] = mol_seed;

  for (int ii = 0; ii < inum; ii++) { // parallel loop
    int i = d_ilist[ii];
    if (mask[i] & groupbit) {
      int molID = molecule[i];
      mol_cluster_local[molID] = molID;
    }
  }

  //make sure mol_offset criteria is also accounted for
  for (int ii = 0; ii < inum; ii++) { // parallel loop
    int i = d_ilist[ii];
    if (mask[i] & groupbit) {
      int molID = molecule[i];
      if (mol_state[molID] == 0 || mol_state[molID] == 1) {
        mol_cluster_local[molID-mol_offset] = molID;
      }
    }
  }

  // use MPI_MAX here since we start with -1
  MPI_Allreduce(mol_cluster_local,mol_cluster,maxmol+1,MPI_INT,MPI_MAX,world);

  // loop until no more changes to mol_cluster (local copy on mol_cluster_local) on any proc

  int change,done,anychange;

  while (1) {

    //update local copy
    for (int i = 0; i <= maxmol; i++) mol_cluster_local[i] = mol_cluster[i];

    change = 0;
    while (1) {
      done = 1;
      for (int ii = 0; ii < inum; ii++) {
        const int i = d_ilist[ii];
        if (!(mask[i] & groupbit)) continue;
        const int i_molID = molecule[i];
        const int itype = type[i];

        const double xtmp = x(i,0);
        const double ytmp = x(i,1);
        const double ztmp = x(i,2);
        const int jnum = d_numneigh[i];

        for (int jj = 0; jj < jnum; jj++) {
          const int j = d_neighbors(i,jj) & NEIGHMASK;
          if (!(mask[j] & groupbit)) continue;          
          const int j_molID = molecule[j];
          const int jtype = type[j];

          if (mol_cluster_local[i_molID] == mol_cluster_local[j_molID]) continue;

          // first check to see if valid contact types

          int contactFlag = -1;
          for (int m = 0; m < nContactTypes; m++) {
            if (contactFlag > -1) break;
            for (int n = 0; n < nAtomsPerContact; n++) {
              if (contactFlag > -1) break;
              if (contactMap[m][n][0] == itype && contactMap[m][n][1] == jtype) contactFlag = 1;
            }
          }

          if (contactFlag == 1) {

            const double delx = xtmp - x(j,0);
            const double dely = ytmp - x(j,1);
            const double delz = ztmp - x(j,2);
            const double rsq = delx*delx + dely*dely + delz*delz;
            //then check if distance criteria is valid and update mols and offset mols
            //next version needs to incorporate explicit contact maps
            if (rsq < cutsq) {
              int newID_v1 = MIN(mol_cluster_local[i_molID],mol_cluster_local[j_molID]);
              int newID_v2 = newID_v1;
              int newID_v3 = newID_v2;
              //if mol is a switchable mol, check -mol_offset; else, check mol_offset
              if (mol_state[i_molID] == 0 || mol_state[i_molID] == 1) newID_v2 = MIN(mol_cluster_local[i_molID-mol_offset],newID_v1);
              else newID_v2 = MIN(mol_cluster_local[i_molID+mol_offset],newID_v1);

              if (mol_state[j_molID] == 0 || mol_state[j_molID] == 1) newID_v3 = MIN(mol_cluster_local[j_molID-mol_offset],newID_v2);
              else newID_v3 = MIN(mol_cluster_local[j_molID+mol_offset],newID_v2);
            
              //now update cluster ID of i, j
              mol_cluster_local[i_molID] = mol_cluster_local[j_molID] = newID_v3;

              //and update cluster ID of offset mols to be consistent
              if (mol_state[i_molID] == 0 || mol_state[i_molID] == 1) mol_cluster_local[i_molID-mol_offset] = newID_v3;
              else mol_cluster_local[i_molID+mol_offset] = newID_v3;

              if (mol_state[j_molID] == 0 || mol_state[j_molID] == 1) mol_cluster_local[j_molID-mol_offset] = newID_v3;
              else mol_cluster_local[j_molID+mol_offset] = newID_v3;

              done = 0;
            }
          }
        }
      }
      if (!done) change = 1;
      if (done) break;
    }

    MPI_Allreduce(&change,&anychange,1,MPI_INT,MPI_MAX,world);
    MPI_Allreduce(mol_cluster_local,mol_cluster,maxmol+1,MPI_INT,MPI_MIN,world); // send min cluster ID to global array
    if (!anychange) break;
  }

  // now switch mol_restrict flags based on cluster ID of mol_seed (mol_cluster should be copied beforehand)
  int clusterID = mol_cluster[mol_seed];
  nCluster = 0.0;
  for (int i = 0; i <= maxmol; i++) {
    if (mol_cluster[i] != -1) {
      // if this is a switchable mol, mol_restrict should be updated
      if (mol_state[i] == 0 || mol_state[i] == 1) {
        if (mol_cluster[i] == clusterID) {
          mol_restrict[i] = -1;
          mol_state[i] = 1;
        }
        else mol_restrict[i] = 1;
      }
      if (mol_cluster[i] == clusterID) nCluster += 1.0;
    }
  }

  // print cluster stats here
  if (comm->me == 0) {

    //    printf("Debugging 1\n");
    int currtime = update->ntimestep;
    //    printf("Debugging 2\n");
    fprintf(fp1,"%d ",currtime);
    fprintf(fp2,"%d ",currtime);
    //    printf("Debugging 3\n");
    for (int i = 0; i <= maxmol; i++) {
      int clusterflag = 0;
      if (mol_cluster[i] == clusterID) clusterflag = 1;
      fprintf(fp1, "%d ",clusterflag);
      fprintf(fp2, "%d ",mol_state[i]);
    }
    fprintf(fp1,"\n");
    fprintf(fp2,"\n");

    fflush(fp1);
    fflush(fp2);
  }
  
  delete [] mol_cluster_local;

  //  modify->addstep_compute(update->ntimestep + nevery);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::attempt_switch()
{
  mask = atomKK->k_mask.view<DeviceType>();
  molecule = atomKK->k_molecule.view<DeviceType>(); // molecule[i] = mol IDs for atom tag i
  type = atomKK->k_type.view<DeviceType>(); // molecule[i] = mol IDs for atom tag i
  int nlocal = atom->nlocal;

  //first gather unique molIDs on this processor
  hash = new std::map<int,int>();

  typedef hash_type::size_type size_type;    // uint32_t
  typedef hash_type::key_type key_type;      // int
  typedef hash_type::value_type value_type;  // int

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (hash->find(molecule[i]) == hash->end()) {
        (*hash)[molecule[i]] = 1;
      }
    }
  }
  int nmol_local = hash->size();

  //no memory; initialize array such that every state is -1 to start
  for (int i = 0; i <= nmol; i++) {
    //    for (int j = 0; j < nSwitchTypes; j++) {
    for (int j = 0; j < nSwitchPerMol; j++) {
      mol_atoms[i][j] = -1;
    }
  }

  //local copy of mol_accept (this should be zero'd at every attempt
  int *mol_accept_local = new int[maxmol+1];
  for (int i = 0; i <= maxmol; i++) {
    mol_accept_local[i] = -1;
    mol_accept[i] = -1;
    for (int j = 0; j < nSwitchPerMol; j++) mol_atoms[i][j] = -1;
  }

  std::map<int,int>::iterator pos;
  for (pos = hash->begin(); pos != hash->end(); ++pos) {
    int mID = pos->first;
    int confirmflag;
    //    mol_restrict[n] = mID;
    //    if (mol_restrict[mID] != 1) error->all(FLERR,"Current molID in fix cluster_switch is not valid!");
    if (mol_restrict[mID] == 1) confirmflag = confirm_molecule(mID); //checks if this proc should be decision-maker
    else confirmflag = 0;

    if (mol_accept_local[mID] == -1 && confirmflag != 0)  mol_accept_local[mID] = switch_flag(mID);
  }

  // communicate accept flags across processors...
  MPI_Allreduce(mol_accept_local, mol_accept, maxmol+1, MPI_INT, MPI_MAX, world);
  gather_statistics(); //keep track of MC statistics
  check_arrays();

  //now perform switchings on each proc
  for (int i = 0; i <= maxmol; i++) {
    //if acceptance flag is turned on
    if (mol_accept[i] == 1) {
      for (int j = 0; j < nSwitchPerMol; j++) {
        int tagi = mol_atoms[i][j];
        //if originally in OFF state
        if (mol_state[i] == 0 && tagi > -1) {
          for (int k = 0; k < nSwitchTypes; k++) {
            if (type[tagi] == atomtypesOFF[k]) type[tagi] = atomtypesON[k];
          }
        }
        //if originally in ON state
        else if (mol_state[i] == 1 && tagi > -1) {        
          for (int k = 0; k < nSwitchTypes; k++) {
            if (type[tagi] == atomtypesON[k]) type[tagi] = atomtypesOFF[k];
          }
        }
      }
      //update mol_state
      if (mol_state[i] == 0) mol_state[i] = 1;
      else if (mol_state[i] == 1) mol_state[i] = 0;
    }
  } 

  //communicate changed types
  comm->forward_comm(this);

  delete [] mol_accept_local;
  delete hash;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixClusterSwitchKokkos<DeviceType>::confirm_molecule( int molID )
{
  molecule = atomKK->k_molecule.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();

  double sumState = 0.0;
  double decisionBuffer = (double)nSwitchPerMol/2.0 - 1.0 + 0.01; // just to ensure that the proc with the majority of the switching types makes the switching decision

  for (int i = 0; i < atom->nlocal; i++) {

    if (molecule[i] == molID) {

      int itype = type[i];
      for (int k = 0; k < nSwitchTypes; k++) {

        if (itype == atomtypesON[k]) {

          for (int j = 0; j < nSwitchPerMol; j++) {

            if (mol_atoms[molID][j] == -1) {
              mol_atoms[molID][j] = i; //tag[i]; //i;
              sumState += 1.0;          
              break;
            }
          }
        } else if (itype == atomtypesOFF[k]) {

          for (int j = 0; j < nSwitchPerMol; j++) {

            if (mol_atoms[molID][j] == -1) {
              mol_atoms[molID][j] = i; //tag[i]; //i;
              sumState -= 1.0;
              break;
            }

          }
        }
          
      }
    }
  }
    
  if (sumState < (decisionBuffer * -1)) return -1;
  else if (sumState > decisionBuffer) return 1;
  else return 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixClusterSwitchKokkos<DeviceType>::switch_flag( int molID )
{
  int state = mol_state[molID];
  double checkProb;

  // if current state is OFF (0), then use probability to turn ON
  if (state == 0) {
    checkProb = probON;
  }
  else {
    checkProb = probOFF;
  }
  
  double rand = random_unequal->uniform();
  if (rand < checkProb) {
    return 1;
  }
  else
    return 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixClusterSwitchKokkos<DeviceType>::compute_vector(int n)
{
  if (n == 0) return nAttemptsTotal;
  if (n == 1) return nSuccessTotal;
  if (n == 2) return nAttemptsON;
  if (n == 3) return nAttemptsOFF;
  if (n == 4) return nSuccessON;
  if (n == 5) return nSuccessOFF;
  if (n == 6) return nCluster;
  return 0.0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::gather_statistics()
{
  //gather these stats before mol_state is updated
  double dt_AttemptsTotal = 0.0, dt_AttemptsON = 0.0, dt_AttemptsOFF = 0.0;
  double dt_SuccessTotal = 0.0, dt_SuccessON = 0.0, dt_SuccessOFF = 0.0;
  for (int i = 0; i <= maxmol; i++) {
    if (mol_restrict[i] == 1) {
      dt_AttemptsTotal += 1.0;
      if (mol_state[i] == 0) {
        dt_AttemptsON += 1.0;
        if (mol_accept[i] == 1) {
          dt_SuccessTotal += 1.0;
          dt_SuccessON += 1.0;
        }
      } else if (mol_state[i] == 1) {
        dt_AttemptsOFF += 1.0;
        if (mol_accept[i] == 1) {
          dt_SuccessTotal += 1.0;
          dt_SuccessOFF += 1.0;
        }
      }
    }
  }

  //now update
  nAttemptsTotal += dt_AttemptsTotal;
  nAttemptsON += dt_AttemptsON;
  nAttemptsOFF += dt_AttemptsOFF;
  nSuccessTotal += dt_SuccessTotal;
  nSuccessON += dt_SuccessON;
  nSuccessOFF += dt_SuccessOFF;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixClusterSwitchKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixClusterSwitchKokkos<LMPHostType>;
#endif
}
