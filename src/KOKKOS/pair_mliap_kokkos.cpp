/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matt Bettencourt (NVIDIA)
------------------------------------------------------------------------- */

#include "pair_mliap_kokkos.h"
#include "memory_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "mliap_data_kokkos.h"
#include "mliap_descriptor_so3_kokkos.h"
#include "mliap_model_linear_kokkos.h"
#ifdef MLIAP_PYTHON
#include "mliap_model_python_kokkos.h"
#include "mliap_unified_kokkos.h"
#endif
#include "comm_kokkos.h"
#include "comm_tiled_kokkos.h"
#include "error.h"
#include "neigh_request.h"
#include "lammps.h"
#include "kokkos.h"
#include "pointers.h"

#include <climits>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {
// true for the instance that runs on the GPU in a KOKKOS GPU build
template <class DeviceType>
constexpr bool is_device_instance =
    std::is_same_v<DeviceType, LMPDeviceType> && !std::is_same_v<LMPDeviceType, LMPHostType>;
}    // namespace

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMLIAPKokkos<DeviceType>::PairMLIAPKokkos(class LAMMPS *lmp) : PairMLIAP(lmp)
{
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_modify = 0;
  is_child=true;
  reverse_comm_device = 1;
  comm_type=COMM_TYPE::UNSET;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMLIAPKokkos<DeviceType>::~PairMLIAPKokkos()
{
  memoryKK->destroy_kokkos(k_map, map);
  memoryKK->destroy_kokkos(k_cutsq, cutsq);
  memoryKK->destroy_kokkos(k_setflag, setflag);
  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
  if (ghostneigh) memoryKK->destroy_kokkos(k_cutghost, cutghost);
  delete model;
  delete descriptor;
  model=nullptr;
  descriptor=nullptr;
  allocated = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::compute(int eflag, int vflag)
{
  atomKK->sync(execution_space,X_MASK | TYPE_MASK );
  MLIAPDataKokkos<DeviceType> *k_data = (MLIAPDataKokkos<DeviceType>*)(data);

  int is_kokkos_model = (dynamic_cast<MLIAPModelKokkos<DeviceType>*>(model)) != nullptr;
  int is_kokkos_descriptor = (dynamic_cast<MLIAPDescriptorKokkos<DeviceType>*>(descriptor)) != nullptr;
  auto model_space = is_kokkos_model ? execution_space : Host;
  auto descriptor_space = is_kokkos_descriptor? execution_space : Host;
  // consistency checks
  if (data->ndescriptors != model->ndescriptors)
    error->all(FLERR, "Incompatible model and descriptor descriptor count");

  if (data->nelements != model->nelements)
    error->all(FLERR, "Incompatible model and descriptor element count");

  ev_init(eflag, vflag, 0);
  if (eflag_atom) {
    if ((int)k_eatom.view_host().extent(0) < maxeatom) {
      memoryKK->destroy_kokkos(k_eatom,eatom);
      memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    } else {
      Kokkos::deep_copy(k_eatom.template view<DeviceType>(),0);
      k_eatom.modify<DeviceType>();
      k_eatom.sync_host();
    }
  }

  if (vflag_atom) {
    if ((int)k_vatom.view_host().extent(0) < maxvatom) {
      memoryKK->destroy_kokkos(k_vatom,vatom);
      memoryKK->create_kokkos(k_vatom,vatom,maxvatom,6,"pair:vatom");
    } else {
      Kokkos::deep_copy(k_vatom.template view<DeviceType>(),0);
      k_vatom.modify<DeviceType>();
      k_vatom.sync_host();
    }
  }

  data->generate_neighdata(list, eflag, vflag);

  // compute descriptors, if needed
  if (model->nonlinearflag || eflag)  {
    k_data->sync(descriptor_space, NUMNEIGHS_MASK | IATOMS_MASK | IELEMS_MASK | ELEMS_MASK | JATOMS_MASK | PAIR_I_MASK | JELEMS_MASK | RIJ_MASK );
    descriptor->compute_descriptors(data);
    if (!is_kokkos_descriptor)
      k_data->modified(descriptor_space, DESCRIPTORS_MASK);
  }

  // compute E_i and beta_i = dE_i/dB_i for all i in list
  k_data->sync(model_space, IELEMS_MASK | DESCRIPTORS_MASK);
  model->compute_gradients(data);
  k_data->modified(model_space, BETAS_MASK);
  if (eflag_atom) {
    k_data->modified(model_space, EATOMS_MASK);
  }

  // calculate force contributions beta_i*dB_i/dR_j
  atomKK->sync(descriptor_space,F_MASK);
  k_data->sync(descriptor_space, NUMNEIGHS_MASK | IATOMS_MASK | IELEMS_MASK | ELEMS_MASK | BETAS_MASK | JATOMS_MASK | PAIR_I_MASK | JELEMS_MASK | RIJ_MASK );
  descriptor->compute_forces(data);

  e_tally(data);

  if (evflag) {
    atomKK->modified(descriptor_space,F_MASK | ENERGY_MASK | VIRIAL_MASK);
    atomKK->sync(execution_space,F_MASK | ENERGY_MASK | VIRIAL_MASK);
  } else {
    atomKK->modified(descriptor_space,F_MASK);
    atomKK->sync(execution_space,F_MASK);
  }

  // calculate stress
  if (vflag_fdotr) {
    pair_virial_fdotr_compute(this);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::allocate()
{
  int n = atom->ntypes;

  memoryKK->destroy_kokkos(k_map, map);
  memoryKK->destroy_kokkos(k_cutsq, cutsq);
  memoryKK->destroy_kokkos(k_setflag, setflag);
  memoryKK->create_kokkos(k_map, map, n+1, "pair_mliap:map");
  memoryKK->create_kokkos(k_cutsq, cutsq, n+1, n+1, "pair_mliap:cutsq");
  memoryKK->create_kokkos(k_setflag, setflag, n+1, n+1, "pair_mliap:setflag");

  if (ghostneigh) memoryKK->create_kokkos(k_cutghost, cutghost, n+1, n+1, "pair_mliap:cutghost");
  // this is for the base class so it doesn't double delete
  allocated = 1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::settings(int narg, char ** arg)
{
  std::vector<char*> new_args;
  int iarg=0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"model") == 0) {
      if (strcmp(arg[iarg+1],"linear") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal pair_style mliap command");
        delete model;
        model = new MLIAPModelLinearKokkos<DeviceType>(lmp,arg[iarg+2]);
        iarg += 3;
      } else if (strcmp(arg[iarg+1],"mliappy") == 0) {
#ifdef MLIAP_PYTHON
        if (iarg+3 > narg) utils::missing_cmd_args(FLERR, "pair_style mliap mliappy", error);
        delete model;
        model = new MLIAPModelPythonKokkos<DeviceType>(lmp,arg[iarg+2]);
        iarg += 3;
#else
        error->all(FLERR,"Using pair_style mliap model mliappy requires ML-IAP with python support");
#endif
      } else {
        new_args.push_back(arg[iarg++]);
        new_args.push_back(arg[iarg++]);
      }
    } else if (strcmp(arg[iarg],"descriptor") == 0) {
      if (strcmp(arg[iarg+1],"so3") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal pair_style mliap command");
        delete descriptor;
        descriptor = new MLIAPDescriptorSO3Kokkos<DeviceType>(lmp,arg[iarg+2]);
        iarg += 3;
      } else
        new_args.push_back(arg[iarg++]);
    } else if (strcmp(arg[iarg], "unified") == 0) {
#ifdef MLIAP_PYTHON
      if (model != nullptr) error->all(FLERR,"Illegal multiple pair_style mliap model definitions");
      if (descriptor != nullptr) error->all(FLERR,"Illegal multiple pair_style mliap descriptor definitions");
      if (iarg+2 > narg) utils::missing_cmd_args(FLERR, "pair_style mliap unified", error);
      MLIAPBuildUnifiedKokkos_t<DeviceType> build = build_unified(arg[iarg+1], MLIAPDataKokkos<DeviceType>::cast(data), lmp);
      if (iarg+3 > narg) {
        ghostneigh = 0;
      } else {
        ghostneigh = utils::logical(FLERR, arg[iarg+2], false, lmp);
      }

      iarg += 3;
      model = build.model;
      descriptor = build.descriptor;
#else
      error->all(FLERR,"Using pair_style mliap unified requires ML-IAP with python support");
#endif
    } else
      new_args.push_back(arg[iarg++]);
  }
  PairMLIAP::settings(new_args.size(), new_args.data());

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::coeff(int narg, char **arg) {
  if (narg < 3) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  char* type1 = arg[0];
  char* type2 = arg[1];
  char** elemtypes = &arg[2];

  // ensure I,J args are * *

  if (strcmp(type1,"*") != 0 || strcmp(type2,"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));

  // read args that map atom types to elements
  // map[i] = which element the Ith atom type is, -1 if not mapped
  // map[0] is not used

  for (int i = 1; i <= atom->ntypes; i++) {
    char* elemname = elemtypes[i-1];
    int jelem;
    for (jelem = 0; jelem < descriptor->nelements; jelem++)
      if (strcmp(elemname,descriptor->elements[jelem]) == 0)
        break;

    if (jelem < descriptor->nelements)
      map[i] = jelem;
    else if (strcmp(elemname,"NULL") == 0) map[i] = -1;
    else error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
  }
  k_map.modify_host();
  k_map.sync_device();

  // clear setflag since coeff() called once with I,J = * *

  int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }
  k_setflag.modify_host();
  k_setflag.sync_device();

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));

  // set up model, descriptor, and mliap data structures
  model->init();
  descriptor->init();

  auto h_cutsq=k_cutsq.view_host();
  for (int itype=1; itype <= atom->ntypes; ++itype)
    for (int jtype=1; jtype <= atom->ntypes; ++jtype)
      // do not set cuts for NULL atoms
      if (map[itype] >= 0 && map[jtype] >= 0) {
        h_cutsq(itype,jtype) = descriptor->cutsq[map[itype]][map[jtype]];
      }
  k_cutsq.modify_host();
  k_cutsq.sync<DeviceType>();
  constexpr int gradgradflag = -1;
  delete data;
  data = new MLIAPDataKokkos<DeviceType>(lmp, gradgradflag, map, model, descriptor, this);
  data->init();
}

/* ----------------------------------------------------------------------
   add energies to eng_vdwl and per-atom energy
------------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::e_tally(MLIAPData* data)
{
  if (eflag_global) eng_vdwl += data->energy;
  if (eflag_atom) {
    MLIAPDataKokkos<DeviceType> *k_data = static_cast<MLIAPDataKokkos<DeviceType>*>(data);
    k_data->sync(execution_space, IATOMS_MASK | EATOMS_MASK, true);
    auto d_iatoms = k_data->k_iatoms.template view<DeviceType>();
    auto d_eatoms = k_data->k_eatoms.template view<DeviceType>();
    auto d_eatom = k_eatom.template view<DeviceType>();
    k_eatom.sync<DeviceType>();
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,data->nlistatoms), KOKKOS_LAMBDA (int ii) {
      d_eatom(d_iatoms(ii)) = d_eatoms(ii);
    });
    k_eatom.modify<DeviceType>();
    // This sync has to be here for the hybrid pair type
    k_eatom.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::init_style()
{

  PairMLIAP::init_style();
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   The ML-IAP feature exchange passes pointers to arrays owned by Torch,
   CuPy or NumPy and a vector length that is only known at the time of
   the call.  The plain (CommBrick or CommTiled) pair communication would
   pack through host callbacks into buffers that were sized in Comm::init()
   before that length was known, which overflows them, and for a device
   instance the host callbacks would also read device memory.  So always use the KOKKOS pair communication with the execution space
   of this instance: it grows its buffers as needed and stages only the
   MPI buffers through the host when GPU-aware MPI is off.  This also
   applies when KOKKOS selects the plain pair communication by itself
   ("pair/only on", or "gpu/aware off" on more than one MPI rank).  An
   explicit "comm/pair/forward host" or "no" (or the reverse keyword)
   cannot be honored for a device instance, so stop with an error.
------------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::exchange_pair_comm(bool forward)
{
  if constexpr (is_device_instance<DeviceType>) {
    const int legacy =
      forward ? lmp->kokkos->forward_pair_comm_legacy : lmp->kokkos->reverse_pair_comm_legacy;
    const int changed =
      forward ? lmp->kokkos->forward_pair_comm_changed : lmp->kokkos->reverse_pair_comm_changed;
    if (legacy && !changed) {
      const char *dir = forward ? "forward" : "reverse";
      error->all(FLERR, "Pair style mliap/kk on the GPU does not support package kokkos "
                 "comm/pair/{} host or no, use comm/pair/{} device", dir, dir);
    }
  }

  if (auto *commKK = dynamic_cast<CommKokkos *>(comm)) {
    if (forward) commKK->forward_comm_device<DeviceType>(this);
    else commKK->reverse_comm_device<DeviceType>(this);
  } else if (auto *commTiledKK = dynamic_cast<CommTiledKokkos *>(comm)) {
    if (forward) commTiledKK->forward_comm_device<DeviceType>(this);
    else commTiledKK->reverse_comm_device<DeviceType>(this);
  } else {
    error->all(FLERR, "Pair style mliap/kk requires a KOKKOS communication style");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<class CommType>
int PairMLIAPKokkos<DeviceType>::forward_comm(CommType* copy_from_, CommType* copy_to_, const int vl)
{
  static_assert( std::is_same_v<CommType,float>
              || std::is_same_v<CommType,double>,
                 "Unsupported CommType");
  if constexpr ( std::is_same_v<CommType,float> ) {
    comm_type = COMM_TYPE::FLOAT;
  } else if constexpr ( std::is_same_v<CommType,double> ) {
    comm_type = COMM_TYPE::DOUBLE;
  }
  copy_to = copy_to_;
  copy_from = copy_from_;
  comm_forward = vec_len=vl;

  if ((vl < 0) || ((long long) (atom->nlocal + atom->nghost) * vl > (long long) INT_MAX))
    error->one(FLERR, "ML-IAP forward exchange: invalid vector length {} for {} local and ghost "
               "atoms", vl, atom->nlocal + atom->nghost);
  // the arrays may be written and read on a stream that KOKKOS does not know
  // about, so enclose the exchange of a device instance by global fences
  if constexpr (is_device_instance<DeviceType>)
    Kokkos::fence("PairMLIAPKokkos::forward_comm: before exchange");
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,(atom->nlocal+atom->nghost)*vl), KOKKOS_LAMBDA (int i) {
    copy_to_[i] = copy_from_[i];
  });
  //call comm
  exchange_pair_comm(true);
  if constexpr (is_device_instance<DeviceType>)
    Kokkos::fence("PairMLIAPKokkos::forward_comm: after exchange");

  return 0;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
template<class CommType>
int PairMLIAPKokkos<DeviceType>::reverse_comm(CommType* copy_from_, CommType* copy_to_, const int vl)
{
  static_assert( std::is_same_v<CommType,float>
              || std::is_same_v<CommType,double>,
                 "Unsupported CommType");
  if constexpr ( std::is_same_v<CommType,float> ) {
    comm_type = COMM_TYPE::FLOAT;
  } else if constexpr ( std::is_same_v<CommType,double> ) {
    comm_type = COMM_TYPE::DOUBLE;
  }
  copy_to = copy_to_;
  copy_from = copy_from_;
  comm_reverse = vec_len = vl;

  if ((vl < 0) || ((long long) (atom->nlocal + atom->nghost) * vl > (long long) INT_MAX))
    error->one(FLERR, "ML-IAP reverse exchange: invalid vector length {} for {} local and ghost "
               "atoms", vl, atom->nlocal + atom->nghost);
  if constexpr (is_device_instance<DeviceType>)
    Kokkos::fence("PairMLIAPKokkos::reverse_comm: before exchange");
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,(atom->nlocal+atom->nghost)*vl), KOKKOS_LAMBDA (int i) {
    copy_to_[i] = copy_from_[i]; // Copy inputs
  });

  exchange_pair_comm(false);

  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType>( (atom->nlocal)*vl,(atom->nlocal + atom->nghost)*vl),
    KOKKOS_LAMBDA (int i) {
    copy_to_[i] = 0; //Zero out ghosts
  });
  if constexpr (is_device_instance<DeviceType>)
    Kokkos::fence("PairMLIAPKokkos::reverse_comm: after exchange");

  return 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
int PairMLIAPKokkos<DeviceType>::pack_forward_comm_kokkos(
    int nv, DAT::tdual_int_1d idx_v, DAT::tdual_double_1d &fill, int int2,
    int *intp) {
  switch( comm_type ) {
    case COMM_TYPE::FLOAT:
      return pack_forward_comm_kokkos(nv,idx_v,fill,int2,intp,std::get<float*>(copy_to));
    case COMM_TYPE::DOUBLE:
      return pack_forward_comm_kokkos(nv,idx_v,fill,int2,intp,std::get<double*>(copy_to));
    case COMM_TYPE::UNSET:
    default:
      error->all(FLERR,"comm_type was never set");
      return -1;
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <typename CommType>
int PairMLIAPKokkos<DeviceType>::pack_forward_comm_kokkos(
  int nv, DAT::tdual_int_1d idx_v, DAT::tdual_double_1d &fill, int /*int2*/,
  int* /*intp*/, CommType *copy_to) {
  auto idx=idx_v.view<DeviceType>();
  auto val=fill.view<DeviceType>();
  int nf=vec_len;
  auto to=copy_to;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nv*nf), KOKKOS_LAMBDA (int start) {
      const int i = start/nf;
      const int gstart=idx(i)*nf;
      const int j = start%nf;
      val(start) = static_cast<double>(to[gstart+j]);
    }
  );
  return nv*nf;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMLIAPKokkos<DeviceType>::pack_forward_comm(int, int *, double *, int, int *)
{
  error->one(FLERR, "Internal error: ML-IAP feature exchange must use the KOKKOS pair communication");
  return 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairMLIAPKokkos<DeviceType>::unpack_forward_comm_kokkos(
    int nv, int first_up, DAT::tdual_double_1d &fill) {
  switch( comm_type ) {
    case COMM_TYPE::FLOAT:
      return unpack_forward_comm_kokkos(nv,first_up,fill,std::get<float*>(copy_to));
    case COMM_TYPE::DOUBLE:
      return unpack_forward_comm_kokkos(nv,first_up,fill,std::get<double*>(copy_to));
    case COMM_TYPE::UNSET:
    default:
      error->all(FLERR,"comm_type was never set");
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <typename CommType>
void PairMLIAPKokkos<DeviceType>::unpack_forward_comm_kokkos(
    int nv, int first_up, DAT::tdual_double_1d &fill, CommType *copy_to) {
  auto val=fill.view<DeviceType>();
  int nf=vec_len;
  auto to=copy_to;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nv*nf), KOKKOS_LAMBDA (int start) {
      const int i=start/nf;
      const int gstart=(first_up+i)*nf;
      const int j=start%nf;
      to[gstart+j] = static_cast<CommType>(val(start));
    }
  );
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::unpack_forward_comm(int, int, double *)
{
  error->one(FLERR, "Internal error: ML-IAP feature exchange must use the KOKKOS pair communication");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMLIAPKokkos<DeviceType>::pack_reverse_comm_kokkos(int nv, int first_up, DAT::tdual_double_1d &fill)
{
  switch( comm_type ) {
    case COMM_TYPE::FLOAT:
      return pack_reverse_comm_kokkos(nv,first_up,fill,std::get<float*>(copy_to));
    case COMM_TYPE::DOUBLE:
      return pack_reverse_comm_kokkos(nv,first_up,fill,std::get<double*>(copy_to));
    case COMM_TYPE::UNSET:
    default:
      error->all(FLERR,"comm_type was never set");
  }
  return -1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<typename CommType>
int PairMLIAPKokkos<DeviceType>::pack_reverse_comm_kokkos(int nv, int first_up, DAT::tdual_double_1d &fill, CommType *copy_to)
{
  int nf=vec_len;
  auto val=fill.view<DeviceType>();
  auto to=copy_to;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nv*nf), KOKKOS_LAMBDA (int start) {
      const int i = start/nf;
      const int gstart=(first_up+i)*nf;
      const int j = start%nf;
      val(start) = static_cast<double>(to[gstart+j]);
    }
  );
  return nv*nf;
}
/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMLIAPKokkos<DeviceType>::pack_reverse_comm(int, int, double *)
{
  error->one(FLERR, "Internal error: ML-IAP feature exchange must use the KOKKOS pair communication");
  return 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::unpack_reverse_comm_kokkos(int nv, DAT::tdual_int_1d idx_v, DAT::tdual_double_1d &fill)
{
  switch( comm_type ) {
    case COMM_TYPE::FLOAT:
      return unpack_reverse_comm_kokkos(nv,idx_v,fill,std::get<float*>(copy_to));
    case COMM_TYPE::DOUBLE:
      return unpack_reverse_comm_kokkos(nv,idx_v,fill,std::get<double*>(copy_to));
    case COMM_TYPE::UNSET:
    default:
      error->all(FLERR,"comm_type was never set");
      return;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<typename CommType>
void PairMLIAPKokkos<DeviceType>::unpack_reverse_comm_kokkos(int nv, DAT::tdual_int_1d idx_v, DAT::tdual_double_1d &fill, CommType *copy_to)
{
  int nf=vec_len;
  auto val=fill.view<DeviceType>();
  auto idx=idx_v.view<DeviceType>();
  auto to=copy_to;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nv*nf), KOKKOS_LAMBDA (int start) {
      const int i = start/nf;
      const int gstart=idx(i)*nf;
      const int j=start%nf;
      to[gstart+j] += static_cast<CommType>(val(start));
    }
  );
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMLIAPKokkos<DeviceType>::unpack_reverse_comm(int, int *, double *)
{
  error->one(FLERR, "Internal error: ML-IAP feature exchange must use the KOKKOS pair communication");
}

/* ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   convert a pointer to the base class into one to this class, nullptr if
   it points to an instance of another class.  this must be done here,
   where the class is explicitly instantiated: elsewhere the dynamic_cast
   uses a hidden local copy of the type information with the C++ ABI of
   macOS on arm64, so the linker warns about different visibility settings
------------------------------------------------------------------------- */

template<class DeviceType>
PairMLIAPKokkos<DeviceType> *PairMLIAPKokkos<DeviceType>::cast(PairMLIAP *ptr)
{
  return dynamic_cast<PairMLIAPKokkos<DeviceType> *>(ptr);
}

namespace LAMMPS_NS {
template class PairMLIAPKokkos<LMPDeviceType>;
template int PairMLIAPKokkos<LMPDeviceType>::forward_comm<float>(float*,float*,const int);
template int PairMLIAPKokkos<LMPDeviceType>::forward_comm<double>(double*,double*,const int);
template int PairMLIAPKokkos<LMPDeviceType>::reverse_comm<float>(float*,float*,const int);
template int PairMLIAPKokkos<LMPDeviceType>::reverse_comm<double>(double*,double*,const int);
#ifdef LMP_KOKKOS_GPU
template class PairMLIAPKokkos<LMPHostType>;
template int PairMLIAPKokkos<LMPHostType>::forward_comm<float>(float*,float*,const int);
template int PairMLIAPKokkos<LMPHostType>::forward_comm<double>(double*,double*,const int);
template int PairMLIAPKokkos<LMPHostType>::reverse_comm<float>(float*,float*,const int);
template int PairMLIAPKokkos<LMPHostType>::reverse_comm<double>(double*,double*,const int);
#endif
}
