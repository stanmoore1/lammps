/* ---------------------------------------------------------------------- */
#include "fix_cluster_switch.h"
#include "mpi.h"
#include "modify.h"
#include "memory.h"
#include "update.h"
#include "force.h"
#include "random_park.h" //random number generator

#include "error.h" //error
#include <string.h> //strcmp()
#include <math.h>
#include <utility>

#include "respa.h"
#include "neighbor.h" //neigh->build
#include "neigh_list.h" //class NeighList
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

#define MYDEBUG
#define MAXLINE 1024
//#define MYDEBUG1

/* ---------------------------------------------------------------------- */

FixClusterSwitch::FixClusterSwitch(LAMMPS *lmp, int narg, char **arg) : Fix(lmp,narg,arg)
{
  /*
  USAGE:
     fix  1   all  cluster_switch molID_seed  mol_offset cutoff  15123(seed) rateFreq   1000  rateFile  rates.txt  contactFile  contacts.txt
     arg:[0]  [1]     [2]            [3]        [4]        [5]        [6]       [7]     [8]      [9]        [10]       [11]        [12]
  Checks for largest contiguous cluster connected to molID_seed (molID_seed - mol_offset)
  Switches population of all molecules that are NOT part of the contiguous cluster
  */
  if (narg < 8) error->all(FLERR,"Illegal cluster_switch command");

  mol_seed = utils::inumeric(FLERR, arg[3], false, lmp);
  mol_offset = utils::inumeric(FLERR, arg[4], false, lmp); // molIDs that are mol_offset less than molIDs-counted-based-on-rates.txt should be counted toward clustering
  double cutoff = utils::numeric(FLERR, arg[5], false, lmp); //cutoff distance for contact mapping
  
  int seed = utils::inumeric(FLERR, arg[6], false, lmp);

  switchFreq = utils::inumeric(FLERR, arg[8], false, lmp); // how often attempts to switch are made, equivalent to "nevery"
  random_equal = new RanPark(lmp, seed);
  random_unequal = new RanPark(lmp, seed);

  // ERROR CHECKPOINT: make sure pair style has cutoff attribute
  if (force->pair == nullptr) error->all(FLERR,"fix cluster_switch requires a pair style");
  if (force->pair->cutsq == nullptr) error->all(FLERR,"fix cluster_switch is incompatible with pair style");
  cutsq = cutoff * cutoff; //force->pair->cutsq;
  
  // INITIALIZE FLAGS
  force_reneighbor = 1; //fix requires reneighboring
  next_reneighbor = update->ntimestep + 1; //reneighboring flag, will set later on
  //next_reneighbor = update->ntimestep; //reneighboring flag, will set later on
  comm_forward = 1; //Flag for forward communication (send local atom data to other procs as ghost particles)
  pack_flag = 0; //This may or may not be used later (to set which information is transmitted to adjacent nodes)

  fp1 = nullptr;
  fp2 = nullptr;
  atomtypesON = nullptr;
  atomtypesOFF = nullptr;
  contactMap = nullptr;
  mol_restrict = nullptr;
  mol_atoms = nullptr;
  mol_state = nullptr;
  mol_cluster = nullptr;
  mol_accept = nullptr;
  
  vector_flag = 1;
  size_vector = 7; 
  global_freq = 1;
  extvector = 0;
  time_depend = 1;
  
  // INITIALIZE COUNTERS FOR STATISTICS
  nAttemptsTotal = 0.0;
  nAttemptsON = 0.0;
  nAttemptsOFF = 0.0;
  nSuccessTotal = 0.0;
  nSuccessON = 0.0;
  nSuccessOFF = 0.0;
  nCluster = 0.0;

  // check molecule flag
  if(atom->molecule_flag == 0) error->all(FLERR,"fix cluster_switch requires that atoms have molecule attributes");

 rate_file = std::string(arg[10]);
 contact_file = std::string(arg[12]); 
}

/* ---------------------------------------------------------------------- */
FixClusterSwitch::~FixClusterSwitch()
{
  if (copymode) return;

  memory->destroy(atomtypesON);
  memory->destroy(atomtypesOFF);
  memory->destroy(contactMap);

  memory->destroy(mol_restrict);
  memory->destroy(mol_atoms);
  memory->destroy(mol_state);
  memory->destroy(mol_cluster);
  memory->destroy(mol_accept);

  if (fp1 && comm->me == 0) fclose(fp1);
  if (fp2 && comm->me == 0) fclose(fp2);

  delete random_equal;
  delete random_unequal;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::read_file(const char *file)
{
  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = utils::open_potential(file, lmp, nullptr);
    if (fp == nullptr) {
      char str[128];
      sprintf(str,"Cannot open rates file (fix cluster_switch) called %s",file);
      error->one(FLERR,str);
    }
  }

  // read each line out of file, skipping blank lines or leading '#'
  // store line of params if all 3 element tags are in element list

  int n,nwords,lineNum,eof;
  char line[MAXLINE],*ptr;
  char **words = new char*[100];

  eof = lineNum = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    lineNum++;

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = utils::trim_and_count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(nullptr," \t\n\r\f"))) continue;

    if(lineNum == 1) {
      probON  = atof(words[0]);
      if(probON > 1.0) {
	char str[128];
	sprintf(str,"Incorrect probability in rates.txt files (fix cluster_switch)");
	error->one(FLERR,str);
      }
      probOFF = 1.0 - probON;
    }
    else if (lineNum == 2) {
      nSwitchTypes = atoi(words[0]);
      if(nSwitchTypes > atom->ntypes) {
	char str[128];
	sprintf(str,"Incorrect number of atom switching types (fix cluster_switch)");
	error->one(FLERR,str);
      }
      allocate(2);
    }
    else if (lineNum == 3) {
      for(int i = 0; i < nSwitchTypes; i++) atomtypesON[i] = atoi(words[i]);
    }
    else if (lineNum == 4) {
      for(int i = 0; i < nSwitchTypes; i++) atomtypesOFF[i] = atoi(words[i]);
    }
  }
  delete [] words;


}


/* ---------------------------------------------------------------------- */
void FixClusterSwitch::read_contacts(const char *file)
{
  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = utils::open_potential(file, lmp, nullptr);
    if (fp == nullptr) {
      char str[128];
      sprintf(str,"Cannot open contact file (fix cluster_switch) called %s",file);
      error->one(FLERR,str);
    }
  }

  // read each line out of file, skipping blank lines or leading '#'
  // store line of params if all 3 element tags are in element list

  int n,nwords,lineNum,eof;
  char line[MAXLINE],*ptr;
  char **words = new char*[100];

  eof = lineNum = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    lineNum++;

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = utils::trim_and_count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(nullptr," \t\n\r\f"))) continue;

    if(lineNum == 1) {
      nContactTypes  = atoi(words[1]);
      //      printf("Num Contacts %d\n",nContactTypes);
    }
    else if (lineNum == 2) {
      nAtomsPerContact = atoi(words[1]);
      //      printf("Atoms per Contact %d\n",nAtomsPerContact);
      allocate(3);
    }
    else {
      int lineOffset = lineNum - 3;
      int i = lineOffset / nAtomsPerContact; // take advantage of integer division
      int j = lineOffset - i*nAtomsPerContact;
      contactMap[i][j][0] = atoi(words[0]);
      contactMap[i][j][1] = atoi(words[1]);
      //      printf("Contacts [%d][%d] are %d and %d\n",i,j,contactMap[i][j][0],contactMap[i][j][1]);
    }
  }
  delete [] words;


}

/* ---------------------------------------------------------------------- */
int FixClusterSwitch::setmask()
{
  int mask=0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::init()
{
  // READ INPUT FILE THAT CONTAINS RATES AND ATOM TYPE INFO
  read_file(rate_file.c_str()); // reads the rate file and initializes/populates arrays
  read_contacts(contact_file.c_str()); // reads contact map file

  int nmolatoms = 0;
  int maxmol_local = -1;
  int *atypes = atom->type;
  int *molecule = atom->molecule;
  int nSwitchPerMol_local = 0;
  for(int i = 0; i < atom->nlocal; i++){
    if(atom->mask[i] & groupbit){
      if(molecule[i] > maxmol_local) maxmol_local = molecule[i];
      for(int j = 0; j < nSwitchTypes; j++){
	if(atypes[i] == atomtypesON[j] || atypes[i] == atomtypesOFF[j]) {
	  nmolatoms++; // the number of atoms that are viable for switching
	  if(molecule[i] == mol_seed) {
	    nSwitchPerMol_local++; // the number of switchable atoms per molecule
	  }
	}
      }
    }
  }
  MPI_Allreduce(&nmolatoms, &nmol, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&maxmol_local, &maxmol, 1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(&nSwitchPerMol_local, &nSwitchPerMol, 1, MPI_INT, MPI_SUM, world);
  //printf("nmol init: %d\n", nmol);
  //nmol = nmol / nSwitchTypes; //this logic doesn't work when nSwitchTypes != switchable_atoms_per_molecule
  //this is now corrected to find the total number of molecules that are switchable
  nmol = nmol / nSwitchPerMol;
  //printf("nmol reduce: %d\n", nmol);
  //printf("Maximum mol number is %d\n",maxmol);

  if(maxmol < 0) error->all(FLERR,"Selected group does not have any mols (fix cluster_switch)");

  allocate(1);

  //populate mol_restrict list with -1 if restricted and 1 if molecule is switch enabled, then share data across processors
  int *mol_restrict_local = new int[maxmol+1];
  int *mol_state_local = new int[maxmol+1];
  for(int i = 0; i <= maxmol; i++){
    mol_restrict_local[i] = -1;
    mol_state_local[i] = -1;
  }
  
  for(int i = 0; i < atom->nlocal; i++){
    if(atom->mask[i] & groupbit){
      int molID = molecule[i];

      for(int j = 0; j < nSwitchTypes; j++){
	if(atypes[i] == atomtypesON[j] && mol_state_local[molID] == -1) {
	  mol_state_local[molID] = 1;
	  if(molID != mol_seed && molID != (mol_seed - mol_offset)){
	    mol_restrict_local[molID] = 1;
	  }
	}
	else if(atypes[i] == atomtypesOFF[j] && mol_state_local[molID] == -1) {
	  mol_state_local[molID] = 0;
	  if(molID != mol_seed && molID != (mol_seed - mol_offset)){
	    mol_restrict_local[molID] = 1;
	  }
	}
      }
    }
  }

  MPI_Allreduce(mol_restrict_local, mol_restrict, maxmol+1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(mol_state_local, mol_state, maxmol+1, MPI_INT, MPI_MAX, world); 

  delete [] mol_restrict_local;
  delete [] mol_state_local;
  
  check_arrays();

  // open file for debugging 
  if(comm->me == 0) {
    fp1 = fopen("cluster_assignment.log","w");
    if(fp1 == nullptr) {
      error->one(FLERR,"File cluster_assignment.log in cluster_switch not open!\n");
    }     
    fp2 = fopen("state_assignment.log","w");
    if(fp2 == nullptr) {
      error->one(FLERR,"File state_assignment.log in cluster_switch not open!\n");
    }     
  }
  
  //  // init cluster
  //  check_cluster();
  //  attempt_switch();

  ngroup = group->count(igroup);
  //groupbit = group->bitmask[igroup];
    
  if(strstr(update->integrate_style,"respa")) nlevels_respa = ((Respa *) update->integrate)->nlevels;
  
  //need a full neighbor list, built whenever re-neighboring occurs//
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

  //  printf("FixClusterSwitch successfully initialized.\n");
}

void FixClusterSwitch::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::allocate(int flag)
{

  //printf("Debug Flag: Allocation\n");

  if (flag == 1)
    {
      memory->create(mol_restrict,maxmol+1,"fix:mol_restrict");
      memory->create(mol_state,maxmol+1,"fix:mol_state");
      memory->create(mol_accept,maxmol+1,"fix:mol_accept");
      memory->create(mol_cluster,maxmol+1,"fix:mol_cluster");
      //      memory->create(mol_atoms,maxmol+1,nSwitchTypes,"fix:mol_atoms");
      memory->create(mol_atoms,maxmol+1,nSwitchPerMol,"fix:mol_atoms");

      //no memory; initialize array such that every state is -1 to start
      for(int i = 0; i <= maxmol; i++){
	mol_restrict[i] = -1;
	mol_state[i] = -1;
	mol_accept[i] = -1;
	//	for(int j = 0; j < nSwitchTypes; j++){
	for(int j = 0; j < nSwitchPerMol; j++){
	  mol_atoms[i][j] = -1;
	}
      }

    }

  else if(flag == 2)
    {
      memory->create(atomtypesON,nSwitchTypes,"fix:atomtypesON");
      memory->create(atomtypesOFF,nSwitchTypes,"fix:atomtypesOFF");
    }

  else if(flag == 3)
    {
      memory->create(contactMap,nContactTypes,nAtomsPerContact,2,"fix:contactMap");
    }

  //  printf("FixClusterSwitch allocate(=%d) finished.\n",flag);
}


/* ---------------------------------------------------------------------- */
//void FixClusterSwitch::pre_force(int /*vflag*/)
void FixClusterSwitch::pre_exchange()
{
  // printf("The current timestep is %d, switchFreq is %d, mod of the two is %d\n", int(update->ntimestep), switchFreq, int(update->ntimestep) % switchFreq);
  if (switchFreq == 0) return; 
  //if (update->ntimestep % switchFreq) return;
  if (next_reneighbor != update->ntimestep) return;

  domain->pbc();
  comm->exchange();
  comm->borders();
  if(modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  check_cluster();
  attempt_switch();

  next_reneighbor = update->ntimestep + switchFreq;
}

void FixClusterSwitch::normalize(double *a)
{
  double mag=0.0;
  for(int k=0;k<3;k++) mag += a[k]*a[k];
  if(mag<=0) error->all(FLERR,"Normalization error in cluster_switch");
  mag=sqrt(mag);
  for(int k=0;k<3;k++) a[k] /= mag;
}

void FixClusterSwitch::computecross(double *a, double *b1, double *b2)
{
  a[0] = b1[1]*b2[2] - b1[2]*b2[1];
  a[1] = b1[2]*b2[0] - b1[0]*b2[2];
  a[2] = b1[0]*b2[1] - b1[1]*b2[0];
}
 

void FixClusterSwitch::check_arrays()
{
  for(int i = 0; i <= maxmol; i++){
    //printf("For mol %d, restrict flag is %d and state flag is %d on proc %d\n", i, mol_restrict[i], mol_state[i], comm->me);
    if(mol_restrict[i] == 1 && !(mol_state[i] == 1 || mol_state[i] == 0)) error->all(FLERR,"Communication of mol_state inconsistent: fix cluster_switch");
  }
}

/* ---------------------------------------------------------------------- */
// Sends local atom data toward neighboring procs (into ghost atom data)
// Reverse sends ghost data to local data (e.g., forces)
int FixClusterSwitch::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int m;

  //  if(pack_flag == 1)
  //    for(m = 0; m < n; m++) buf[m] = mol_restrict[list[m]];
  //  if(pack_flag == 2)
  //    for(m = 0; m < n; m++) buf[m] = mol_state[list[m]];
  //  if(pack_flag == 3)
  //    for(m = 0; m < n; m++) buf[m] = mol_accept[list[m]];
  //  if(pack_flag == 4)
  for(m = 0; m < n; m++) buf[m] = atom->type[list[m]];

  return m;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  //  if(pack_flag == 1)
  //    for(m = 0, i = first; m < n; m++, i++) mol_restrict[i] = buf[m];
  //  if(pack_flag == 2)
  //    for(m = 0, i = first; m < n; m++, i++) mol_state[i] = buf[m];
  //  if(pack_flag == 1)
  //    for(m = 0, i = first; m < n; m++, i++) mol_accept[i] = buf[m];
  //  if(pack_flag == 2)
  for(m = 0, i = first; m < n; m++, i++) atom->type[i] = static_cast<int> (buf[m]);
}

/* ---------------------------------------------------------------------- */
double FixClusterSwitch::memory_usage()
{
  return 0;
}


void FixClusterSwitch::check_cluster()
{

  //  modify->clearstep_compute();

  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // invoke neighbor list (full)
  neighbor->build_one(list);
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int *mol_cluster_local = new int[maxmol+1];
  for(int i = 0; i <= maxmol; i++){
    mol_cluster_local[i] = -1;
    mol_cluster[i] = -1;

  }

  //preset mol_seed and mol_offset
  mol_cluster_local[mol_seed] = mol_cluster[mol_seed] = mol_seed;
  mol_cluster_local[mol_seed-mol_offset] = mol_cluster[mol_seed-mol_offset] = mol_seed;

  for(int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if(mask[i] & groupbit) {
      int molID = molecule[i];
      mol_cluster_local[molID] = molID;
    }
  }

  //make sure mol_offset criteria is also accounted for
  for(int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if(mask[i] & groupbit) {
      int molID = molecule[i];
      if(mol_state[molID] == 0 || mol_state[molID] == 1){
	mol_cluster_local[molID-mol_offset] = molID;
      }
    }
  }

  // use MPI_MAX here since we start with -1
  MPI_Allreduce(mol_cluster_local,mol_cluster,maxmol+1,MPI_INT,MPI_MAX,world);

  // loop until no more changes to mol_cluster (local copy on mol_cluster_local) on any proc
  double **x = atom->x;

  int change,done,anychange;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *jlist, jnum;

  while (1) {

    //update local copy
    for(int i = 0; i <= maxmol; i++) mol_cluster_local[i] = mol_cluster[i];

    change = 0;
    while (1) {
      done = 1;
      for(int ii = 0; ii < inum; ii++){
	int i = ilist[ii];
	if(!(mask[i] & groupbit)) continue;
	int i_molID = molecule[i];
	int itype = type[i];

	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	jlist = firstneigh[i];
	jnum = numneigh[i];

	for(int jj = 0; jj < jnum; jj++){
	  int j = jlist[jj];
	  j &= NEIGHMASK;
	  if(!(mask[j] & groupbit)) continue;	  
	  int j_molID = molecule[j];
	  int jtype = type[j];

	  if(mol_cluster_local[i_molID] == mol_cluster_local[j_molID]) continue;

	  // first check to see if valid contact types
	  //	  printf("Checking types %d and %d now\n",itype,jtype);

	  int contactFlag = -1;
	  for(int m = 0; m < nContactTypes; m++){
	    if(contactFlag > -1) break;
	    for(int n = 0; n < nAtomsPerContact; n++){
	      if(contactFlag > -1) break;
	      if(contactMap[m][n][0] == itype && contactMap[m][n][1] == jtype) contactFlag = 1;
	    }
	  }

	  if(contactFlag == 1) {

	    //printf("Made it to contactFlag=1 area\n");

	    delx = xtmp - x[j][0];
	    dely = ytmp - x[j][1];
	    delz = ztmp - x[j][2];
	    rsq = delx*delx + dely*dely + delz*delz;
	    //printf("Computed rsq distance %f for itype %d and jtype %d\n", rsq, itype, jtype);
	    //then check if distance criteria is valid and update mols and offset mols
	    //next version needs to incorporate explicit contact maps
	    if(rsq < cutsq) {
	      int newID_v1 = MIN(mol_cluster_local[i_molID],mol_cluster_local[j_molID]);
	      int newID_v2 = newID_v1;
	      int newID_v3 = newID_v2;
	      //printf("Made it inside rsq %f < cutsq %f\n", rsq, cutsq);
	      //printf("Mol IDs are %d and %d\n", i_molID, j_molID);
	      //if mol is a switchable mol, check -mol_offset; else, check mol_offset
	      if(mol_state[i_molID] == 0 || mol_state[i_molID] == 1) newID_v2 = MIN(mol_cluster_local[i_molID-mol_offset],newID_v1);
	      else newID_v2 = MIN(mol_cluster_local[i_molID+mol_offset],newID_v1);

	      if(mol_state[j_molID] == 0 || mol_state[j_molID] == 1) newID_v3 = MIN(mol_cluster_local[j_molID-mol_offset],newID_v2);
	      else newID_v3 = MIN(mol_cluster_local[j_molID+mol_offset],newID_v2);
	    
	      //now update cluster ID of i, j
	      mol_cluster_local[i_molID] = mol_cluster_local[j_molID] = newID_v3;

	      //and update cluster ID of offset mols to be consistent
	      if(mol_state[i_molID] == 0 || mol_state[i_molID] == 1) mol_cluster_local[i_molID-mol_offset] = newID_v3;
	      else mol_cluster_local[i_molID+mol_offset] = newID_v3;

	      if(mol_state[j_molID] == 0 || mol_state[j_molID] == 1) mol_cluster_local[j_molID-mol_offset] = newID_v3;
	      else mol_cluster_local[j_molID+mol_offset] = newID_v3;

	      done = 0;

	      //printf("Mol cluster IDs are i: %d and j: %d\n", mol_cluster_local[i_molID], mol_cluster_local[j_molID]);
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
  for(int i = 0; i <= maxmol; i++){
    if(mol_cluster[i] != -1){
      // if this is a switchable mol, mol_restrict should be updated
      if(mol_state[i] == 0 || mol_state[i] == 1) {
	if(mol_cluster[i] == clusterID) {
	  mol_restrict[i] = -1;
	  mol_state[i] = 1;
	} else
          mol_restrict[i] = 1;
      }
      if(mol_cluster[i] == clusterID) nCluster += 1.0;
    }
  }

  // print cluster stats here
  if(comm->me == 0) {

    //    printf("Debugging 1\n");
    int currtime = update->ntimestep;
    //    printf("Debugging 2\n");
    fprintf(fp1,"%d ",currtime);
    fprintf(fp2,"%d ",currtime);
    //    printf("Debugging 3\n");
    for(int i = 0; i <= maxmol; i++) {
      int clusterflag = 0;
      if(mol_cluster[i] == clusterID) clusterflag = 1;
      fprintf(fp1, "%d ",clusterflag);
      fprintf(fp2, "%d ",mol_state[i]);
    }
    fprintf(fp1,"\n");
    fprintf(fp2,"\n");
    //    printf("Debugging 4\n");

    fflush(fp1);
    fflush(fp2);
  }
  
  delete [] mol_cluster_local;

  //  modify->addstep_compute(update->ntimestep + nevery);

}

void FixClusterSwitch::attempt_switch()
{
  int *mask = atom->mask;
  int *molecule = atom->molecule; // molecule[i] = mol IDs for atom tag i
  int nlocal = atom->nlocal;

  //first gather unique molIDs on this processor
  hash = new std::map<int,int>();

  //printf("Debug Section 1A\n");

  for(int i = 0; i < nlocal; i++){
    if(mask[i] & groupbit){
      if(hash->find(molecule[i]) == hash->end()) {
	(*hash)[molecule[i]] = 1;
	//printf("molecule[i] for i = %d prints out molID %d\n",i,molecule[i]);
      }
    }
  }
  int nmol_local = hash->size();
  //printf("Total number of mols on proc %d is %d\n",comm->me,nmol_local);

  //printf("Debug Section 2A\n");
  //printf("nmol is: %d\n", nmol);
  //no memory; initialize array such that every state is -1 to start
  for(int i = 0; i <= nmol; i++){
    //    for(int j = 0; j < nSwitchTypes; j++){
    for(int j = 0; j < nSwitchPerMol; j++){
      mol_atoms[i][j] = -1;
    }
  }

  //printf("Debug Section 3A\n");
  //local copy of mol_accept (this should be zero'd at every attempt
  int *mol_accept_local = new int[maxmol+1];
  for(int i = 0; i <= maxmol; i++){
    mol_accept_local[i] = -1;
    mol_accept[i] = -1;
    for(int j = 0; j < nSwitchPerMol; j++) mol_atoms[i][j] = -1;
  }

  //printf("Debug Section 4A\n");
  std::map<int,int>::iterator pos;
  for (pos = hash->begin(); pos != hash->end(); ++pos){
    int mID = pos->first;
    int confirmflag;
    //    printf("molID is %d on proc %d\n",mID,comm->me);
    //    mol_restrict[n] = mID;
    //    if(mol_restrict[mID] != 1) error->all(FLERR,"Current molID in fix cluster_switch is not valid!");
    if(mol_restrict[mID] == 1) confirmflag = confirm_molecule(mID); //checks if this proc should be decision-maker
    else confirmflag = 0;

    //printf("For molID %d, we have confirmflag %d\n", mID, confirmflag);

    if(mol_accept_local[mID] == -1 && confirmflag != 0)
      mol_accept_local[mID] = switch_flag(mID);

    //printf("For molID %d, we have mol_accept_local[mID] = %d\n", mID, mol_accept_local[mID]);
  }

  // communicate accept flags across processors...
  MPI_Allreduce(mol_accept_local, mol_accept, maxmol+1, MPI_INT, MPI_MAX, world);
  gather_statistics(); //keep track of MC statistics
  check_arrays();

  //printf("Debug Section 5A\n");
  //now perform switchings on each proc
  int *atype = atom->type;
  //  tagint *atag = atom->tag;
  for(int i = 0; i <= maxmol; i++){
    //if acceptance flag is turned on
    if(mol_accept[i] == 1){
      for(int j = 0; j < nSwitchPerMol; j++){
	int tagi = mol_atoms[i][j];
	//if originally in OFF state
	if(mol_state[i] == 0 && tagi > -1){
	  for(int k = 0; k < nSwitchTypes; k++){
	    if(atype[tagi] == atomtypesOFF[k]) atype[tagi] = atomtypesON[k];
	  }
	}
	//if originally in ON state
	else if(mol_state[i] == 1 && tagi > -1){	
	  for(int k = 0; k < nSwitchTypes; k++){
	    if(atype[tagi] == atomtypesON[k]) atype[tagi] = atomtypesOFF[k];
	  }
	}
      }
      //update mol_state
      if(mol_state[i] == 0) mol_state[i] = 1;
      else if(mol_state[i] == 1) mol_state[i] = 0;
    }
  } 

  //printf("Debug Section 6A\n");
  //communicate changed types
  //  pack_flag = 4;
  comm->forward_comm(this);
  //domain->pbc();
  //comm->exchange();
  //comm->borders();

  //printf("Debug Section 7A\n");

  delete [] mol_accept_local;
  delete hash;

  //printf("Debug Section 8A\n");
}

int FixClusterSwitch::confirm_molecule( int molID )
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
	      //	      printf("Adding sumState using atag %d\n", atag[i]);
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


int FixClusterSwitch::switch_flag( int molID )
{
  //  printf("Check 1\n");

  int state = mol_state[molID];
  double checkProb;

  //  printf("Check 2\n");

  // if current state is OFF (0), then use probability to turn ON
  if(state == 0) {
    checkProb = probON;
  }
  else {
    checkProb = probOFF;
  }
  
  //  printf("Check 3\n");

  double rand = random_unequal->uniform();
  if(rand < checkProb){
    return 1;
  }
  else
    return 0;
}

double FixClusterSwitch::compute_vector(int n)
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

void FixClusterSwitch::gather_statistics()
{
  //gather these stats before mol_state is updated
  double dt_AttemptsTotal = 0.0, dt_AttemptsON = 0.0, dt_AttemptsOFF = 0.0;
  double dt_SuccessTotal = 0.0, dt_SuccessON = 0.0, dt_SuccessOFF = 0.0;
  for(int i = 0; i <= maxmol; i++) {
    if(mol_restrict[i] == 1){
      dt_AttemptsTotal += 1.0;
      if(mol_state[i] == 0) {
	dt_AttemptsON += 1.0;
	if(mol_accept[i] == 1){
	  dt_SuccessTotal += 1.0;
	  dt_SuccessON += 1.0;
	}
      }
      else if(mol_state[i] == 1){
	dt_AttemptsOFF += 1.0;
	if(mol_accept[i] == 1){
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
