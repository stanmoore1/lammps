
#include "compute_mass_chunk.h"
#include <mpi.h>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "compute_chunk_atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMASSChunk::ComputeMASSChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  idchunk(NULL), massproc(NULL)
{
  if (narg != 4) error->all(FLERR,"Illegal compute mass/chunk command");

  vector_flag = 1;
  size_vector = 0;
  size_vector_variable = 1;
  extvector = 1;

  // ID of compute chunk/atom

  int n = strlen(arg[3]) + 1;
  idchunk = new char[n];
  strcpy(idchunk,arg[3]);

  init();

  // chunk-based data

  nchunk = 1;
  maxchunk = 0;
  allocate();
}

/* ---------------------------------------------------------------------- */

ComputeMASSChunk::~ComputeMASSChunk()
{
  delete [] idchunk;
  delete [] vector;
  memory->destroy(massproc);
}

/* ---------------------------------------------------------------------- */

void ComputeMASSChunk::init()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for compute mass/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Compute mass/chunk does not use chunk/atom compute");
}

/* ---------------------------------------------------------------------- */

void ComputeMASSChunk::setup()
{
    compute_vector();
}

/* ---------------------------------------------------------------------- */

void ComputeMASSChunk::compute_vector()
{
  int index;
  double massone;

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) allocate();
  size_vector = nchunk;
  vector = new double[size_vector];

  // zero local per-chunk values

  for (int i = 0; i < nchunk; i++) massproc[i] = 0.0;

  // compute MASS for each chunk

  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      massproc[index] += massone;
    }

  MPI_Allreduce(massproc,vector,nchunk,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   lock methods: called by fix ave/time
   these methods insure vector size is locked for Nfreq epoch
     by passing lock info along to compute chunk/atom
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   increment lock counter
------------------------------------------------------------------------- */

void ComputeMASSChunk::lock_enable()
{
  cchunk->lockcount++;
}

/* ----------------------------------------------------------------------
   decrement lock counter in compute chunk/atom, it if still exists
------------------------------------------------------------------------- */

void ComputeMASSChunk::lock_disable()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute >= 0) {
    cchunk = (ComputeChunkAtom *) modify->compute[icompute];
    cchunk->lockcount--;
  }
}

/* ----------------------------------------------------------------------
   calculate and return # of chunks = length of vector
------------------------------------------------------------------------- */

int ComputeMASSChunk::lock_length()
{
  nchunk = cchunk->setup_chunks();
  return nchunk;
}

/* ----------------------------------------------------------------------
   set the lock from startstep to stopstep
------------------------------------------------------------------------- */

void ComputeMASSChunk::lock(Fix *fixptr, bigint startstep, bigint stopstep)
{
  cchunk->lock(fixptr,startstep,stopstep);
}

/* ----------------------------------------------------------------------
   unset the lock
------------------------------------------------------------------------- */

void ComputeMASSChunk::unlock(Fix *fixptr)
{
  cchunk->unlock(fixptr);
}

/* ----------------------------------------------------------------------
   free and reallocate per-chunk vectors
------------------------------------------------------------------------- */

void ComputeMASSChunk::allocate()
{
  memory->destroy(massproc);
  maxchunk = nchunk;
  memory->create(massproc,maxchunk,"mass/chunk:massproc");
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeMASSChunk::memory_usage()
{
  double bytes = (bigint) maxchunk * 2 * sizeof(double);
  bytes += (bigint) maxchunk * 2*3 * sizeof(double);
  return bytes;
}
