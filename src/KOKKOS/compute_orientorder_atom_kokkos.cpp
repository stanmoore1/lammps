/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "compute_orientorder_atom_kokkos.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "atom_kokkos.h"
#include "update.h"
#include "modify.h"
#include "neighbor_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "error.h"
#include "math_const.h"
#include "atom_masks.h"
#include "kokkos.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

#ifdef DBL_EPSILON
  #define MY_EPSILON (10.0*DBL_EPSILON)
#else
  #define MY_EPSILON (10.0*2.220446049250313e-16)
#endif

#define QEPSILON 1.0e-6

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeOrientOrderAtomKokkos<DeviceType>::ComputeOrientOrderAtomKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputeOrientOrderAtom(lmp, narg, arg)
{
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeOrientOrderAtomKokkos<DeviceType>::~ComputeOrientOrderAtomKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_qnarray,qnarray);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeOrientOrderAtomKokkos<DeviceType>::init()
{
  ComputeOrientOrderAtom::init();

  d_qlist = t_sna_1i("orientorder/atom:qlist",nqlist);
  auto h_qlist = Kokkos::create_mirror_view(d_qlist);
  for (int i = 0; i < nqlist; i++)
    h_qlist(i) = qlist[i];
  Kokkos::deep_copy(d_qlist,h_qlist);

  // need an occasional full neighbor list

  // irequest = neigh request made by parent class

  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = Kokkos::Impl::is_same<DeviceType,LMPHostType>::value &&
    !Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& max_neighs) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (max_neighs < num_neighs) max_neighs = num_neighs;
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeOrientOrderAtomKokkos<DeviceType>::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow order parameter array if necessary

  if (atom->nmax > nmax) {
    memoryKK->destroy_kokkos(k_qnarray,qnarray);
    nmax = atom->nmax;
    memoryKK->create_kokkos(k_qnarray,qnarray,nmax,ncol,"orientorder/atom:qnarray");
    array_atom = qnarray;
    d_qnarray = k_qnarray.template view<DeviceType>();

    d_qnm = t_sna_3c("orientorder/atom:qnm",nmax,nqlist,2*qmax+1);
  }

  // insure distsq and nearest arrays are long enough

  if (atom->nmax > nmax || maxneigh > d_distsq.extent(1)) {
    d_distsq = t_sna_2d_lr("orientorder/atom:distsq",nmax,maxneigh);
    d_nearest = t_sna_2i_lr("orientorder/atom:nearest",nmax,maxneigh);
    d_rlist = t_sna_3d_lr("orientorder/atom:rlist",nmax,maxneigh,3);

    d_distsq_um = d_distsq;
    d_rlist_um = d_rlist;
    d_nearest_um = d_nearest;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  // compute order parameter for each atom in group
  // use full neighbor list to count atoms less than cutoff

  atomKK->sync(execution_space,X_MASK|MASK_MASK);
  x = atomKK->k_x.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  copymode = 1;
  maxneigh = 0;
  Kokkos::parallel_reduce("ComputeOrientOrderAtomKokkos::find_max_neighs",inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Experimental::Max<int>(maxneigh));

  Kokkos::deep_copy(d_qnm,{0.0,0.0});

  int vector_length = 1;
  int team_size = 1;
  int team_size_max = Kokkos::TeamPolicy<DeviceType>::team_size_max(*this);
#ifdef KOKKOS_ENABLE_CUDA
  team_size = 32;//max_neighs;
  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
#endif

  typename Kokkos::TeamPolicy<DeviceType, TagComputeOrientOrderAtom> policy(inum,team_size,vector_length);
  Kokkos::parallel_for("ComputeOrientOrderAtom",policy,*this);
  copymode = 0;

  k_qnarray.template modify<DeviceType>();
  k_qnarray.template sync<LMPHostType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeOrientOrderAtomKokkos<DeviceType>::operator() (TagComputeOrientOrderAtom,const typename Kokkos::TeamPolicy<DeviceType, TagComputeOrientOrderAtom>::member_type& team) const
{
  const int ii = team.league_rank();
  const int i = d_ilist[ii];
  if (mask[i] & groupbit) {
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int jnum = d_numneigh[i];

    // loop over list of all neighbors within force cutoff
    // distsq[] = distance sq to each
    // rlist[] = distance vector to each
    // nearest[] = atom indices of neighbors

    int ncount = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,jnum),
        [&] (const int jj, int& count) {
      Kokkos::single(Kokkos::PerThread(team), [&] (){
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;
        const F_FLOAT delx = x(j,0) - xtmp;
        const F_FLOAT dely = x(j,1) - ytmp;
        const F_FLOAT delz = x(j,2) - ztmp;
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutsq)
         count++;
      });
    },ncount);

    if (team.team_rank() == 0)
    Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, int& offset, bool final) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      const F_FLOAT delx = x(j,0) - xtmp;
      const F_FLOAT dely = x(j,1) - ytmp;
      const F_FLOAT delz = x(j,2) - ztmp;
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < cutsq) {
        if (final) {
          d_distsq(ii,offset) = rsq;
          d_rlist(ii,offset,0) = delx;
          d_rlist(ii,offset,1) = dely;
          d_rlist(ii,offset,2) = delz;
          d_nearest(ii,offset) = j;
        }
        offset++;
      }
    });

    // if not nnn neighbors, order parameter = 0;

    if ((ncount == 0) || (ncount < nnn)) {
      for (int jj = 0; jj < ncol; jj++)
        d_qnarray(i,jj) = 0.0;
      return;
    }

    // if nnn > 0, use only nearest nnn neighbors

    auto d_distsq_ii = Kokkos::subview(d_distsq_um, ii, Kokkos::ALL);
    auto d_nearest_ii = Kokkos::subview(d_nearest_um, ii, Kokkos::ALL);
    auto d_rlist_ii = Kokkos::subview(d_rlist_um, ii, Kokkos::ALL, Kokkos::ALL);

    if (nnn > 0) {
      select3(nnn,ncount,(double*)d_distsq_ii.data(),(int*)d_nearest_ii.data(),(double**)d_rlist.data());
      ncount = nnn;
    }

    calc_boop(ncount, nqlist, ii);

  }
}

/* ----------------------------------------------------------------------
   select3 routine from Numerical Recipes (slightly modified)
   find k smallest values in array of length n
   sort auxiliary arrays at same time
------------------------------------------------------------------------- */

// Use no-op do while to create single statement

#define SWAP(a,b) do {       \
    tmp = a; a = b; b = tmp; \
  } while(0)

#define ISWAP(a,b) do {        \
    itmp = a; a = b; b = itmp; \
  } while(0)

#define SWAP3(a,b) do {                  \
    tmp = a[0]; a[0] = b[0]; b[0] = tmp; \
    tmp = a[1]; a[1] = b[1]; b[1] = tmp; \
    tmp = a[2]; a[2] = b[2]; b[2] = tmp; \
  } while(0)

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeOrientOrderAtomKokkos<DeviceType>::select3(int k, int n, double *arr, int *iarr, double **arr3) const
{
  int i,ir,j,l,mid,ia,itmp;
  double a,tmp,a3[3];

  arr--;
  iarr--;
  arr3--;
  l = 1;
  ir = n;
  for (;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
        ISWAP(iarr[l],iarr[ir]);
        SWAP3(arr3[l],arr3[ir]);
      }
      return;
    } else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      ISWAP(iarr[mid],iarr[l+1]);
      SWAP3(arr3[mid],arr3[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
        ISWAP(iarr[l],iarr[ir]);
        SWAP3(arr3[l],arr3[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
        ISWAP(iarr[l+1],iarr[ir]);
        SWAP3(arr3[l+1],arr3[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
        ISWAP(iarr[l],iarr[l+1]);
        SWAP3(arr3[l],arr3[l+1]);
      }
      i = l+1;
      j = ir;
      a = arr[l+1];
      ia = iarr[l+1];
      a3[0] = arr3[l+1][0];
      a3[1] = arr3[l+1][1];
      a3[2] = arr3[l+1][2];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
        ISWAP(iarr[i],iarr[j]);
        SWAP3(arr3[i],arr3[j]);
      }
      arr[l+1] = arr[j];
      arr[j] = a;
      iarr[l+1] = iarr[j];
      iarr[j] = ia;
      arr3[l+1][0] = arr3[j][0];
      arr3[l+1][1] = arr3[j][1];
      arr3[l+1][2] = arr3[j][2];
      arr3[j][0] = a3[0];
      arr3[j][1] = a3[1];
      arr3[j][2] = a3[2];
      if (j >= k) ir = j-1;
      if (j <= k) l = i;
    }
  }
}

/* ----------------------------------------------------------------------
   calculate the bond orientational order parameters
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void ComputeOrientOrderAtomKokkos<DeviceType>::calc_boop(int ncount, int nqlist, int iatom) const
{

  //for (int il = 0; il < nqlist; il++) { // move to outside deep_copy
  //  int l = qlist[il];
  //  for(int m = 0; m < 2*l+1; m++) {
  //    d_qnm(il,m).re = 0.0;
  //    d_qnm(il,m).im = 0.0;
  //  }
  //}

  for(int ineigh = 0; ineigh < ncount; ineigh++) { // 
    const double r0 = d_rlist(iatom,ineigh,0);
    const double r1 = d_rlist(iatom,ineigh,1);
    const double r2 = d_rlist(iatom,ineigh,2);
    const double rmag = sqrt(r0*r0 + r1*r1 + r2*r2);
    if(rmag <= MY_EPSILON) {
      return;
    }

    const double costheta = r2 / rmag;
    SNAcomplex expphi = {r0,r1};
    const double rxymag = sqrt(expphi.re*expphi.re+expphi.im*expphi.im);
    if(rxymag <= MY_EPSILON) {
      expphi.re = 1.0;
      expphi.im = 0.0;
    } else {
      const double rxymaginv = 1.0/rxymag;
      expphi.re *= rxymaginv;
      expphi.im *= rxymaginv;
    }

    for (int il = 0; il < nqlist; il++) {
      const int l = d_qlist[il];

      d_qnm(iatom,il,l).re += polar_prefactor(l, 0, costheta);
      SNAcomplex expphim = {expphi.re,expphi.im};
      for(int m = 1; m <= +l; m++) {
        const double prefactor = polar_prefactor(l, m, costheta);
        SNAcomplex c = {prefactor * expphim.re, prefactor * expphim.im};
        d_qnm(iatom,il,m+l).re += c.re;
        d_qnm(iatom,il,m+l).im += c.im;
        if(m & 1) {
          d_qnm(iatom,il,-m+l).re -= c.re;
          d_qnm(iatom,il,-m+l).im += c.im;
        } else {
          d_qnm(iatom,il,-m+l).re += c.re;
          d_qnm(iatom,il,-m+l).im -= c.im;
        }
        SNAcomplex tmp;
        tmp.re = expphim.re*expphi.re - expphim.im*expphi.im;
        tmp.im = expphim.re*expphi.im + expphim.im*expphi.re;
        expphim.re = tmp.re;
        expphim.im = tmp.im;
      }

    }
  }

  // convert sums to averages

  double facn = 1.0 / ncount;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for(int m = 0; m < 2*l+1; m++) {
      d_qnm(iatom,il,m).re *= facn;
      d_qnm(iatom,il,m).im *= facn;
    }
  }

  // calculate Q_l
  // NOTE: optional W_l_hat and components of Q_qlcomp use these stored Q_l values

  int jj = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    double qnormfac = sqrt(MY_4PI/(2*l+1));
    double qm_sum = 0.0;
    for(int m = 0; m < 2*l+1; m++)
      qm_sum += d_qnm(iatom,il,m).re*d_qnm(iatom,il,m).re + d_qnm(iatom,il,m).im*d_qnm(iatom,il,m).im;
    d_qnarray(iatom,jj++) = qnormfac * sqrt(qm_sum);
  }

  // calculate W_l

  if (wlflag) {
    int idxcg_count = 0;
    for (int il = 0; il < nqlist; il++) {
      int l = qlist[il];
      double wlsum = 0.0;
      for(int m1 = 0; m1 < 2*l+1; m1++) {
        for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
          int m = m1 + m2 - l;
          SNAcomplex qm1qm2;
          qm1qm2.re = d_qnm(iatom,il,m1).re*d_qnm(iatom,il,m2).re - d_qnm(iatom,il,m1).im*d_qnm(iatom,il,m2).im;
          qm1qm2.im = d_qnm(iatom,il,m1).re*d_qnm(iatom,il,m2).im + d_qnm(iatom,il,m1).im*d_qnm(iatom,il,m2).re;
          wlsum += (qm1qm2.re*d_qnm(iatom,il,m).re + qm1qm2.im*d_qnm(iatom,il,m).im)*d_cglist[idxcg_count];
          idxcg_count++;
        }
      }
      d_qnarray(iatom,jj++) = wlsum/sqrt(2.0*l+1.0);
    }
  }

  // calculate W_l_hat

  if (wlhatflag) {
    int idxcg_count = 0;
    for (int il = 0; il < nqlist; il++) {
      int l = d_qlist[il];
      double wlsum = 0.0;
      for(int m1 = 0; m1 < 2*l+1; m1++) {
        for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
          const int m = m1 + m2 - l;
          SNAcomplex qm1qm2;
          qm1qm2.re = d_qnm(iatom,il,m1).re*d_qnm(iatom,il,m2).re - d_qnm(iatom,il,m1).im*d_qnm(iatom,il,m2).im;
          qm1qm2.im = d_qnm(iatom,il,m1).re*d_qnm(iatom,il,m2).im + d_qnm(iatom,il,m1).im*d_qnm(iatom,il,m2).re;
          wlsum += (qm1qm2.re*d_qnm(iatom,il,m).re + qm1qm2.im*d_qnm(iatom,il,m).im)*d_cglist[idxcg_count];
          idxcg_count++;
        }
      }
  //      Whats = [w/(q/np.sqrt(np.pi * 4 / (2 * l + 1)))**3 if abs(q) > 1.0e-6 else 0.0 for l,q,w in zip(range(1,max_l+1),Qs,Ws)]
      if (d_qnarray(iatom,il) < QEPSILON)
        d_qnarray(iatom,jj++) = 0.0;
      else {
        const double qnormfac = sqrt(MY_4PI/(2*l+1));
        const double qnfac = qnormfac/d_qnarray(iatom,il);
        d_qnarray(iatom,jj++) = wlsum/sqrt(2.0*l+1.0)*(qnfac*qnfac*qnfac);
      }
    }
  }

  // Calculate components of Q_l, for l=qlcomp

  if (qlcompflag) {
    const int il = iqlcomp;
    const int l = qlcomp;
    if (d_qnarray(iatom,il) < QEPSILON)
      for(int m = 0; m < 2*l+1; m++) {
        d_qnarray(iatom,jj++) = 0.0;
        d_qnarray(iatom,jj++) = 0.0;
      }
    else {
      const double qnormfac = sqrt(MY_4PI/(2*l+1));
      const double qnfac = qnormfac/d_qnarray(iatom,il);
      for(int m = 0; m < 2*l+1; m++) {
        d_qnarray(iatom,jj++) = d_qnm(iatom,il,m).re * qnfac;
        d_qnarray(iatom,jj++) = d_qnm(iatom,il,m).im * qnfac;
      }
    }
  }

}

/* ----------------------------------------------------------------------
   polar prefactor for spherical harmonic Y_l^m, where
   Y_l^m (theta, phi) = prefactor(l, m, cos(theta)) * exp(i*m*phi)
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double ComputeOrientOrderAtomKokkos<DeviceType>::polar_prefactor(int l, int m, double costheta) const
{
  const int mabs = abs(m);

  double prefactor = 1.0;
  for (int i=l-mabs+1; i < l+mabs+1; ++i)
    prefactor *= static_cast<double>(i);

  prefactor = sqrt(static_cast<double>(2*l+1)/(MY_4PI*prefactor))
    * associated_legendre(l,mabs,costheta);

  if ((m < 0) && (m % 2)) prefactor = -prefactor;

  return prefactor;
}

/* ----------------------------------------------------------------------
   associated legendre polynomial
------------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
double ComputeOrientOrderAtomKokkos<DeviceType>::associated_legendre(int l, int m, double x) const
{
  if (l < m) return 0.0;

  double p(1.0), pm1(0.0), pm2(0.0);

  if (m != 0) {
    const double sqx = sqrt(1.0-x*x);
    for (int i=1; i < m+1; ++i)
      p *= static_cast<double>(2*i-1) * sqx;
  }

  for (int i=m+1; i < l+1; ++i) {
    pm2 = pm1;
    pm1 = p;
    p = (static_cast<double>(2*i-1)*x*pm1
         - static_cast<double>(i+m-1)*pm2) / static_cast<double>(i-m);
  }

  return p;
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients
   using the quasi-binomial formula VMK 8.2.1(3)
   specialized for case j1=j2=j=l
------------------------------------------------------------------------- */

template<class DeviceType>
void ComputeOrientOrderAtomKokkos<DeviceType>::init_clebsch_gordan()
{
  double sum,dcg,sfaccg, sfac1, sfac2;
  int m, aa2, bb2, cc2;
  int ifac, idxcg_count;

  idxcg_count = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for(int m1 = 0; m1 < 2*l+1; m1++)
      for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++)
        idxcg_count++;
  }
  idxcg_max = idxcg_count;
  d_cglist = t_sna_1d("orientorder/atom:d_cglist",idxcg_max);
  auto h_cglist = Kokkos::create_mirror_view(d_cglist);

  idxcg_count = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for(int m1 = 0; m1 < 2*l+1; m1++) {
        aa2 = m1 - l;
        for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
          bb2 = m2 - l;
          m = aa2 + bb2 + l;

          sum = 0.0;
          for (int z = MAX(0, MAX(-aa2, bb2));
               z <= MIN(l, MIN(l - aa2, l + bb2)); z++) {
            ifac = z % 2 ? -1 : 1;
            sum += ifac /
              (factorial(z) *
               factorial(l - z) *
               factorial(l - aa2 - z) *
               factorial(l + bb2 - z) *
               factorial(aa2 + z) *
               factorial(-bb2 + z));
          }

          cc2 = m - l;
          sfaccg = sqrt(factorial(l + aa2) *
                        factorial(l - aa2) *
                        factorial(l + bb2) *
                        factorial(l - bb2) *
                        factorial(l + cc2) *
                        factorial(l - cc2) *
                        (2*l + 1));

          sfac1 = factorial(3*l + 1);
          sfac2 = factorial(l);
          dcg = sqrt(sfac2*sfac2*sfac2 / sfac1);

          h_cglist[idxcg_count] = sum * dcg * sfaccg;
          idxcg_count++;
        }
      }
  }
  Kokkos::deep_copy(d_cglist,h_cglist);
}

namespace LAMMPS_NS {
template class ComputeOrientOrderAtomKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class ComputeOrientOrderAtomKokkos<LMPHostType>;
#endif
}

