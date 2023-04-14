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
#include "kokkos.h"

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

  d_done = DAT::t_int_scalar("fix_cluster_switch:done");
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
void FixClusterSwitchKokkos<DeviceType>::allocate(int flag)
{
  if (flag == 1) {
    memoryKK->create_kokkos(k_mol_restrict,mol_restrict,maxmol+1,"fix:mol_restrict");
    memoryKK->create_kokkos(k_mol_state,mol_state,maxmol+1,"fix:mol_state");
    memoryKK->create_kokkos(k_mol_accept,mol_accept,maxmol+1,"fix:mol_accept");
    memoryKK->create_kokkos(k_mol_cluster,mol_cluster,maxmol+1,"fix:mol_cluster");
    memoryKK->create_kokkos(k_mol_atoms,mol_atoms,maxmol+1,nSwitchPerMol,"fix:mol_atoms");

    d_mol_restrict = k_mol_restrict.template view<DeviceType>();
    d_mol_state = k_mol_state.template view<DeviceType>();
    d_mol_accept = k_mol_accept.template view<DeviceType>();
    d_mol_cluster = k_mol_cluster.template view<DeviceType>();
    d_mol_atoms = k_mol_atoms.template view<DeviceType>();

    d_mol_cluster_local = DAT::t_int_1d("fix:mol_cluster_local",maxmol+1);
    d_mol_accept_local = DAT::t_int_1d("fix:mol_accept_local",maxmol+1);

    // initialize arrays such that every state is -1 to start

    {
      // local variables for lambda capture

      auto l_mol_restrict = d_mol_restrict;
      auto l_mol_state = d_mol_state;
      auto l_mol_accept = d_mol_accept;
      auto l_mol_atoms = d_mol_atoms;

      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,maxmol), LAMMPS_LAMBDA(int i) {
        l_mol_restrict[i] = -1;
        l_mol_state[i] = -1;
        l_mol_accept[i] = -1;
        for (int j = 0; j < nSwitchPerMol; j++)
          l_mol_atoms(i,j) = -1;
      });
    }
  } else if (flag == 2) {
    memoryKK->create_kokkos(k_atomtypesON,atomtypesON,nSwitchTypes,"fix:atomtypesON");
    memoryKK->create_kokkos(k_atomtypesOFF,atomtypesOFF,nSwitchTypes,"fix:atomtypesOFF");

    d_atomtypesON = k_atomtypesON.template view<DeviceType>();
    d_atomtypesOFF = k_atomtypesOFF.template view<DeviceType>();
  } else if (flag == 3) {
    memoryKK->create_kokkos(k_contactMap,contactMap,nContactTypes,nAtomsPerContact,2,"fix:contactMap");

    d_contactMap = k_contactMap.template view<DeviceType>();
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
  atomKK->sync(Host, TYPE_MASK);

  int m;

  for (m = 0; m < n; m++) buf[m] = atom->type[list[m]];

  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  atomKK->sync(Host, TYPE_MASK);

  int i, m;

  for (m = 0, i = first; m < n; m++, i++) atom->type[i] = static_cast<int> (buf[m]);

  atomKK->modified(Host, TYPE_MASK);
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

  atomKK->sync(execution_space, MASK_MASK|MOLECULE_MASK|TYPE_MASK|X_MASK);

  // invoke neighbor list (full)

  neighbor->build_one(list);
  int inum = list->inum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  {
    // local variables for lambda capture

    auto l_mol_cluster_local = d_mol_cluster_local;
    auto l_mol_cluster = d_mol_cluster;

    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,maxmol), LAMMPS_LAMBDA(int i) {
      l_mol_cluster_local[i] = -1;
      l_mol_cluster[i] = -1;

      //preset mol_seed and mol_offset
      l_mol_cluster_local[mol_seed] = d_mol_cluster[mol_seed] = mol_seed;
      l_mol_cluster_local[mol_seed-mol_offset] = d_mol_cluster[mol_seed-mol_offset] = mol_seed;
    });
  }

  { 
    // local variables for lambda capture
    
    auto l_mask = mask;
    auto l_molecule = molecule;
    auto l_mol_cluster_local = d_mol_cluster_local;
    auto l_mol_state = d_mol_state;
    
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,inum), LAMMPS_LAMBDA(int ii) {
      const int i = d_ilist[ii];
      if (l_mask[i] & groupbit) {
        const int molID = l_molecule[i];
        l_mol_cluster_local[molID] = molID;
  
        // make sure mol_offset criteria is also accounted for

        if (l_mol_state[molID] == 0 || l_mol_state[molID] == 1) {
          l_mol_cluster_local[molID-mol_offset] = molID;
        }
      }
    });
  }

  // use MPI_MAX here since we start with -1

  if (lmp->kokkos->gpu_aware_flag)
    MPI_Allreduce(d_mol_cluster_local.data(),d_mol_cluster.data(),maxmol+1,MPI_INT,MPI_MAX,world);
  else {
    auto h_mol_cluster_local = Kokkos::create_mirror_view_and_copy(LMPHostType(),d_mol_cluster_local);
    auto h_mol_cluster = k_mol_cluster.h_view;

    MPI_Allreduce(h_mol_cluster_local.data(),h_mol_cluster.data(),maxmol+1,MPI_INT,MPI_MAX,world);

    Kokkos::deep_copy(d_mol_cluster,h_mol_cluster);
  }

  // loop until no more changes to mol_cluster (local copy on mol_cluster_local) on any proc

  int change,done,anychange;

  while (1) {

    //update local copy
    Kokkos::deep_copy(d_mol_cluster_local,d_mol_cluster);

    change = 0;
    while (1) {
      Kokkos::deep_copy(d_done,1);

      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagFixClusterSwitchCheckCluster>(0,inum),*this);

      auto h_done = Kokkos::create_mirror_view_and_copy(LMPHostType(),d_done);
      done = h_done();

      if (!done) change = 1;
      if (done) break;
    }

    MPI_Allreduce(&change,&anychange,1,MPI_INT,MPI_MAX,world);

    if (lmp->kokkos->gpu_aware_flag)
      MPI_Allreduce(d_mol_cluster_local.data(),d_mol_cluster.data(),maxmol+1,MPI_INT,MPI_MIN,world);
    else {
      auto h_mol_cluster_local = Kokkos::create_mirror_view_and_copy(LMPHostType(),d_mol_cluster_local);
      auto h_mol_cluster = k_mol_cluster.h_view;

      MPI_Allreduce(h_mol_cluster_local.data(),h_mol_cluster.data(),maxmol+1,MPI_INT,MPI_MIN,world);

      Kokkos::deep_copy(d_mol_cluster,h_mol_cluster);
    }

    if (!anychange) break;
  }

  {
    // local variables for lambda capture

    auto l_mask = mask;
    auto l_molecule = molecule;
    auto l_mol_cluster_local = d_mol_cluster_local;
    auto l_mol_state = d_mol_state;

    nCluster = 0.0;

    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,maxmol+1), LAMMPS_LAMBDA(int i, double &nCluster) {
      // now switch mol_restrict flags based on cluster ID of mol_seed (mol_cluster should be copied beforehand)
      if (d_mol_cluster[i] != -1) {
        const int clusterID = d_mol_cluster[mol_seed];
        // if this is a switchable mol, mol_restrict should be updated
        if (d_mol_state[i] == 0 || d_mol_state[i] == 1) {
          if (d_mol_cluster[i] == clusterID) {
            d_mol_restrict[i] = -1;
            d_mol_state[i] = 1;
          }
          else d_mol_restrict[i] = 1;
        }
        if (d_mol_cluster[i] == clusterID) nCluster += 1.0;
      }
    },nCluster);
  }

  // print cluster stats here
  if (comm->me == 0) {

    k_mol_cluster.template modify<DeviceType>();
    k_mol_state.template modify<DeviceType>();

    k_mol_cluster.sync_host();
    k_mol_state.sync_host();

    int clusterID = mol_cluster[mol_seed];

    int currtime = update->ntimestep;
    fprintf(fp1,"%d ",currtime);
    fprintf(fp2,"%d ",currtime);
    for (int i = 0; i <= maxmol; i++) {
      int clusterflag = 0;
      if (d_mol_cluster[i] == clusterID) clusterflag = 1;
      fprintf(fp1, "%d ",clusterflag);
      fprintf(fp2, "%d ",mol_state[i]);
    }
    fprintf(fp1,"\n");
    fprintf(fp2,"\n");

    fflush(fp1);
    fflush(fp2);
  }
  
  //  modify->addstep_compute(update->ntimestep + nevery);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixClusterSwitchKokkos<DeviceType>::operator()(TagFixClusterSwitchCheckCluster, const int &ii) const
{
  const int i = d_ilist[ii];
  if (!(mask[i] & groupbit)) return;
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

    if (d_mol_cluster_local[i_molID] == d_mol_cluster_local[j_molID]) continue;

    // first check to see if valid contact types

    int contactFlag = -1;
    for (int m = 0; m < nContactTypes; m++) {
      if (contactFlag > -1) break;
      for (int n = 0; n < nAtomsPerContact; n++) {
        if (contactFlag > -1) break;
        if (d_contactMap(m,n,0) == itype && d_contactMap(m,n,1) == jtype) contactFlag = 1;
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
        int newID_v1 = MIN(d_mol_cluster_local[i_molID],d_mol_cluster_local[j_molID]);
        int newID_v2 = newID_v1;
        int newID_v3 = newID_v2;
        //if mol is a switchable mol, check -mol_offset; else, check mol_offset
        if (d_mol_state[i_molID] == 0 || d_mol_state[i_molID] == 1) newID_v2 = MIN(d_mol_cluster_local[i_molID-mol_offset],newID_v1);
        else newID_v2 = MIN(d_mol_cluster_local[i_molID+mol_offset],newID_v1);

        if (d_mol_state[j_molID] == 0 || d_mol_state[j_molID] == 1) newID_v3 = MIN(d_mol_cluster_local[j_molID-mol_offset],newID_v2);
        else newID_v3 = MIN(d_mol_cluster_local[j_molID+mol_offset],newID_v2);
      
        //now update cluster ID of i, j
        d_mol_cluster_local[i_molID] = d_mol_cluster_local[j_molID] = newID_v3;

        //and update cluster ID of offset mols to be consistent
        if (d_mol_state[i_molID] == 0 || d_mol_state[i_molID] == 1) d_mol_cluster_local[i_molID-mol_offset] = newID_v3;
        else d_mol_cluster_local[i_molID+mol_offset] = newID_v3;

        if (d_mol_state[j_molID] == 0 || d_mol_state[j_molID] == 1) d_mol_cluster_local[j_molID-mol_offset] = newID_v3;
        else d_mol_cluster_local[j_molID+mol_offset] = newID_v3;

        d_done() = 0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixClusterSwitchKokkos<DeviceType>::attempt_switch()
{
  mask = atomKK->k_mask.view<DeviceType>();
  molecule = atomKK->k_molecule.view<DeviceType>(); // molecule[i] = mol IDs for atom tag i
  type = atomKK->k_type.view<DeviceType>(); // molecule[i] = mol IDs for atom tag i
  tag = atomKK->k_tag.view<DeviceType>();
  int nlocal = atom->nlocal;

  atomKK->sync(execution_space, MASK_MASK|MOLECULE_MASK|TYPE_MASK|TAG_MASK);

  //first gather unique molIDs on this processor

  typedef hash_type::size_type size_type;    // uint32_t
  typedef hash_type::key_type key_type;      // int
  typedef hash_type::value_type value_type;  // int

  { 
    // local variables for lambda capture
    
    auto l_mask = mask;
    auto l_molecule = molecule;
    auto l_hash = d_hash;
    
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nlocal), LAMMPS_LAMBDA(int i) {
      if (l_mask[i] & groupbit) {

      /*size_type h_index = hash_kk.find(molecule[i]);
      if (hash_kk.valid_at(h_index))

      auto insert_result = h_map_hash.insert(global, local);
      if (insert_result.failed()) error->one(FLERR, "Kokkos::UnorderedMap insertion failed");*/

        if (hash->find(molecule[i]) == hash->end()) {
          (*hash)[l_molecule[i]] = 1;
        }
      }    
    });
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (hash->find(molecule[i]) == hash->end()) {
        (*hash)[molecule[i]] = 1;
      }
    }
  }
  int nmol_local = hash->size();

  // initialize arrays such that every state is -1 to start
  {
    auto l_mol_atoms = d_mol_atoms;
    auto l_nSwitchPerMol = nSwitchPerMol;

    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nmol+1), LAMMPS_LAMBDA(int i) {
      for (int j = 0; j < l_nSwitchPerMol; j++)
        l_mol_atoms(i,j) = -1;
    });
  }

  //local copy of mol_accept (this should be zero'd at every attempt
  {
    auto l_nSwitchPerMol = nSwitchPerMol;
    auto l_mol_accept_local = d_mol_accept_local;
    auto l_mol_accept = d_mol_accept;
    auto l_mol_atoms = d_mol_atoms;

    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,maxmol+1), LAMMPS_LAMBDA(int i) {
      l_mol_accept_local[i] = -1;
      l_mol_accept[i] = -1;
      for (int j = 0; j < l_nSwitchPerMol; j++) l_mol_atoms(i,j) = -1;
    });
  }


  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagFixClusterSwitchAttemptSwitch>(0,d_hash.capacity()),*this);
  copymode = 0;

  // communicate accept flags across processors
  if (lmp->kokkos->gpu_aware_flag)
    MPI_Allreduce(d_mol_accept_local.data(),d_mol_accept.data(),maxmol+1,MPI_INT,MPI_MAX,world);
  else {
    auto h_mol_accept_local = Kokkos::create_mirror_view_and_copy(LMPHostType(),d_mol_accept_local);
    auto h_mol_accept = k_mol_accept.h_view;

    MPI_Allreduce(h_mol_accept_local.data(),h_mol_accept.data(),maxmol+1,MPI_INT,MPI_MAX,world);

    Kokkos::deep_copy(d_mol_accept,h_mol_accept);
  }

  gather_statistics(); //keep track of MC statistics
  check_arrays();

  //now perform switchings on each proc
  {
    auto l_nSwitchPerMol = nSwitchPerMol;
    auto l_nSwitchTypes = nSwitchTypes;
    auto l_mol_accept_local = d_mol_accept_local;
    auto l_mol_accept = d_mol_accept;
    auto l_mol_atoms = d_mol_atoms;
    auto l_mol_state = d_mol_state;
    auto l_type = type;
    auto l_atomtypesOFF = d_atomtypesOFF;
    auto l_atomtypesON = d_atomtypesON;

    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,maxmol+1), LAMMPS_LAMBDA(int i) {
      //if acceptance flag is turned on
      if (l_mol_accept[i] == 1) {
        for (int j = 0; j < l_nSwitchPerMol; j++) {
          int tagi = l_mol_atoms(i,j);
          //if originally in OFF state
          if (l_mol_state[i] == 0 && tagi > -1) {
            for (int k = 0; k < l_nSwitchTypes; k++) {
              if (l_type[tagi] == l_atomtypesOFF[k]) l_type[tagi] = l_atomtypesON[k];
            }
          }
          //if originally in ON state
          else if (l_mol_state[i] == 1 && tagi > -1) {        
            for (int k = 0; k < l_nSwitchTypes; k++) {
              if (l_type[tagi] == l_atomtypesON[k]) l_type[tagi] = l_atomtypesOFF[k];
            }
          }
        }
        //update mol_state
        if (l_mol_state[i] == 0) l_mol_state[i] = 1;
        else if (l_mol_state[i] == 1) l_mol_state[i] = 0;
      }
    });
  }

  //communicate changed types

  atomKK->modified(execution_space, TYPE_MASK);
  comm->forward_comm(this);
  atomKK->sync(execution_space, TYPE_MASK);

  delete hash;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int FixClusterSwitchKokkos<DeviceType>::confirm_molecule( int molID ) const
{
 int *molecule = atom->molecule;
  int *atype = atom->type;
  int *atag = atom->tag;
  double sumState = 0.0;
  double decisionBuffer = (double)nSwitchPerMol/2.0 - 1.0 + 0.01; // just to ensure that the proc with the majority of the switching types makes the switching decision
  for(int i = 0; i < atom->nlocal; i++){

    //printf("Checking molecule[i] with tagid %d against molID %d on proc %d\n",molecule[i],molID,comm->me);
    if(molecule[i] == molID){

      int itype = atype[i];
      //      printf("Found a matching molecule for local id %d! Now checking against nSwitchPerMol %d and nSwitchType %d using current itype %d on proc %d\n",i,nSwitchPerMol,nSwitchTypes,itype,comm->me);
      for(int k = 0; k < nSwitchTypes; k++){

        if(itype == atomtypesON[k]){

          for(int j = 0; j < nSwitchPerMol; j++){

            if(mol_atoms[molID][j] == -1) {
              mol_atoms[molID][j] = i; //atag[i]; //i;
              sumState += 1.0;    
              //              printf("Adding sumState using atag %d\n", atag[i]);
              break;
            }

          }

        }
        else if(itype == atomtypesOFF[k]){

          for(int j = 0; j < nSwitchPerMol; j++){

            if(mol_atoms[molID][j] == -1) {
              mol_atoms[molID][j] = i; //atag[i]; //i;
              sumState -= 1.0;
              //      printf("Subtracting sumState using atag %d\n", atag[i]);
              break;
            }

          }
        }
          
      }
    }
  }
    
  //printf("Current sumState is %f with decisionbuffer %f and molID %d and comm %d\n", sumState, decisionBuffer, molID, comm->me);
  if(sumState < (decisionBuffer * -1)) return -1;
  else if(sumState > decisionBuffer) return 1;
  else return 0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixClusterSwitchKokkos<DeviceType>::operator()(TagFixClusterSwitchAttemptSwitch, const int &i) const
{
  int mID;// = pos->first; ///////
  int confirmflag;
  if (d_mol_restrict[mID] == 1) confirmflag = confirm_molecule(mID); //checks if this proc should be decision-maker
  else confirmflag = 0;

  rand_type rand_gen = rand_pool.get_state();

  if (d_mol_accept_local[mID] == -1 && confirmflag != 0)  d_mol_accept_local[mID] = switch_flag(mID, rand_gen);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixClusterSwitchKokkos<DeviceType>::operator()(TagFixClusterSwitchConfirmMolecule, const int &i) const
{
  int molID = molecule[i];

  int itype = type[i];
  for (int k = 0; k < nSwitchTypes; k++) {

    if (itype == d_atomtypesON[k]) {

      for (int j = 0; j < nSwitchPerMol; j++) {

        if (d_mol_atoms(molID,j) == -1) {
          d_mol_atoms(molID,j) = i; //tag[i]; //i;
          d_sumState(molID) += 1.0;          
          break;
        }
      }
    } else if (itype == d_atomtypesOFF[k]) {

      for (int j = 0; j < nSwitchPerMol; j++) {

        if (d_mol_atoms(molID,j) == -1) {
          d_mol_atoms(molID,j) = i; //tag[i]; //i;
          d_sumState(molID) -= 1.0;
          break;
        }
      }
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int FixClusterSwitchKokkos<DeviceType>::decide()
{
  int molID;
  double decisionBuffer = (double)nSwitchPerMol/2.0 - 1.0 + 0.01; // just to ensure that the proc with the majority of the switching types makes the switching decision

  if (d_sumState(molID) < (decisionBuffer * -1)) return -1;
  else if (d_sumState(molID) > decisionBuffer) return 1;
  else return 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int FixClusterSwitchKokkos<DeviceType>::switch_flag( int molID, rand_type &rand_gen ) const
{
  int state = d_mol_state[molID];
  double checkProb;

  // if current state is OFF (0), then use probability to turn ON
  if (state == 0)
    checkProb = probON;
  else
    checkProb = probOFF;

  double rand = rand_gen.drand();
  if (rand < checkProb)
    return 1;
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
  //// need parallel_reduce
  //gather these stats before mol_state is updated
  double dt_AttemptsTotal = 0.0, dt_AttemptsON = 0.0, dt_AttemptsOFF = 0.0;
  double dt_SuccessTotal = 0.0, dt_SuccessON = 0.0, dt_SuccessOFF = 0.0;
  for (int i = 0; i <= maxmol; i++) {
    if (d_mol_restrict[i] == 1) {
      dt_AttemptsTotal += 1.0;
      if (d_mol_state[i] == 0) {
        dt_AttemptsON += 1.0;
        if (d_mol_accept[i] == 1) {
          dt_SuccessTotal += 1.0;
          dt_SuccessON += 1.0;
        }
      } else if (d_mol_state[i] == 1) {
        dt_AttemptsOFF += 1.0;
        if (d_mol_accept[i] == 1) {
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
