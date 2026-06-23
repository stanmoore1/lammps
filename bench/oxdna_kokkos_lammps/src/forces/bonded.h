#pragma once

// Bonded interactions (FENE + bonded excluded volume + stacking) computed as a
// single GATHER kernel, mirroring the standalone oxDNA CUDA dna_forces_edge_bonded:
// one thread per particle reads its 3' (n3) and 5' (n5) bonded neighbours,
// accumulates the force/torque on ITSELF only, and writes once with no atomics.
// This replaces the previous two atomic-scatter kernels (backbone + stacking).
//
// For a bond the force on the 3' end equals minus the force on the 5' end
// (Newton's 3rd law), so the per-bond routine returns the 5'-end force F, the
// torque on the 5' end (T5) and on the 3' end (T3); the gather kernel adds
// (+F,T5) when the particle is the 5' end and (-F,T3) when it is the 3' end.
// Energy is counted once per bond (only on the n3 side).
//
// Must run AFTER the nonbonded kernel: it does forces(i) += ... (plain, since
// each thread owns its particle i) on top of the nonbonded result.

#include "../types.h"
#include "../particles.h"
#include "params.h"
#include "orient.h"
#include "mf_oxdna.h"
#include "dna_forces.h"   // compute_lrf (LRF precompute) + frame helpers
#include <Kokkos_Core.hpp>

// Launch-bounds / register tuning for the bonded gather kernel (GPU). See the
// note in dna_forces.h; sweep on the target GPU (no-op on CPU backends).
#ifndef OXDNA_BOND_MAXT
#define OXDNA_BOND_MAXT 128
#endif
#ifndef OXDNA_BOND_MINB
#define OXDNA_BOND_MINB 6
#endif

KOKKOS_INLINE_FUNCTION
void bx_cross(const c_number a[3], const c_number b[3], c_number c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

// One bond, 5' end = (p5, a1/a2/a3), 3' end = (p3, b1/b2/b3).
// Accumulates force on the 5' end into F, torque on 5' into T5, on 3' into T3.
// Returns the bond energy (FENE + bonded excv + stacking).
// =======================================================================
// LAMMPS-FAITHFUL split: the per-bond interaction is computed by TWO separate
// kernels (LAMMPS has `bond oxdna/fene` and `pair oxdna/stk`):
//   * bonded_fene_excv : FENE + the 3 bonded excluded-volume terms
//   * bonded_stk       : stacking only
// The math in each is byte-for-byte the same as the original fused bonded_pair
// (which is retained below as an inline FENE+excv+stk = sum of the two halves).
// =======================================================================
KOKKOS_INLINE_FUNCTION
c_number bonded_fene_excv(const c_number p5[3], const c_number a1[3], const c_number a2[3], const c_number a3[3],
                          const c_number p3[3], const c_number b1[3], const c_number b2[3], const c_number b3[3],
                          const DNAParams &par, const SimBox &box,
                          c_number (&F)[3], c_number (&T5)[3], c_number (&T3)[3]) {
    const c_number pb1 = par.pb1, pb2 = par.pb2, dcbs = par.d_cbs;
    c_number energy = 0;

    // 5'-3' COM separation (wrapped)
    c_number d53[3] = {p5[0]-p3[0], p5[1]-p3[1], p5[2]-p3[2]};
    box.wrap(d53[0], d53[1], d53[2]);

    // interaction sites
    c_number r5bk[3] = {pb1*a1[0]+pb2*a2[0], pb1*a1[1]+pb2*a2[1], pb1*a1[2]+pb2*a2[2]};
    c_number r3bk[3] = {pb1*b1[0]+pb2*b2[0], pb1*b1[1]+pb2*b2[1], pb1*b1[2]+pb2*b2[2]};
    c_number r5bs[3] = {dcbs*a1[0], dcbs*a1[1], dcbs*a1[2]};
    c_number r3bs[3] = {dcbs*b1[0], dcbs*b1[1], dcbs*b1[2]};

    // ---- FENE (backbone sites) ----
    {
        c_number dx = d53[0]+r5bk[0]-r3bk[0];
        c_number dy = d53[1]+r5bk[1]-r3bk[1];
        c_number dz = d53[2]+r5bk[2]-r3bk[2];
        c_number r  = Kokkos::sqrt(dx*dx+dy*dy+dz*dz);
        c_number t  = (r - par.fene.r0) / par.fene.Delta;
        if (Kokkos::fabs(t) < c_number(1.0)) {
            c_number tm = c_number(0.9998);
            if (t > tm) t = tm; if (t < -tm) t = -tm;
            c_number denom = 1 - t*t;
            energy += c_number(-0.5)*par.fene.k*Kokkos::log(denom);
            c_number fpair = -par.fene.k * t / (r * par.fene.Delta * denom);
            c_number df[3] = {dx*fpair, dy*fpair, dz*fpair};
            F[0]+=df[0]; F[1]+=df[1]; F[2]+=df[2];
            c_number c[3];
            bx_cross(r5bk, df, c); T5[0]+=c[0]; T5[1]+=c[1]; T5[2]+=c[2];
            bx_cross(r3bk, df, c); T3[0]-=c[0]; T3[1]-=c[1]; T3[2]-=c[2];
        }
    }

    // ---- Bonded excluded volume: base-base, base(5')-back(3'), back(5')-base(3') ----
    // d = -d53 + (3' site) - (5' site); df = d*fpair; F5 -= df; T5 -= r5site x df; T3 += r3site x df.
    auto excv = [&](const c_number r5s[3], const c_number r3s[3], const ExcvParams &ep) {
        c_number d[3] = {-d53[0]+r3s[0]-r5s[0], -d53[1]+r3s[1]-r5s[1], -d53[2]+r3s[2]-r5s[2]};
        c_number rsq = d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
        if (rsq >= ep.cutsq_c) return;
        c_number fpair = 0;
        energy += MFOxdna::F3(rsq, ep.cutsq_ast, ep.cut_c, ep.lj1, ep.lj2, ep.eps, ep.b, fpair);
        c_number df[3] = {d[0]*fpair, d[1]*fpair, d[2]*fpair};
        F[0]-=df[0]; F[1]-=df[1]; F[2]-=df[2];
        c_number c[3];
        bx_cross(r5s, df, c); T5[0]-=c[0]; T5[1]-=c[1]; T5[2]-=c[2];
        bx_cross(r3s, df, c); T3[0]+=c[0]; T3[1]+=c[1]; T3[2]+=c[2];
    };
    excv(r5bs, r3bs, par.excv_bsbs);   // base(5') - base(3')
    excv(r5bs, r3bk, par.excv_bkbs);   // base(5') - back(3')
    excv(r5bk, r3bs, par.excv_bkbs);   // back(5') - base(3')

    return energy;
}

// ---- Stacking only (LAMMPS `pair oxdna/stk`) ----
KOKKOS_INLINE_FUNCTION
c_number bonded_stk(const c_number p5[3], const c_number a1[3], const c_number a2[3], const c_number a3[3],
                    const c_number p3[3], const c_number b1[3], const c_number b2[3], const c_number b3[3],
                    const DNAParams &par, const SimBox &box,
                    c_number (&F)[3], c_number (&T5)[3], c_number (&T3)[3]) {
    const c_number dcstk = par.d_cstk;
    c_number energy = 0;

    // 5'-3' COM separation (wrapped)
    c_number d53[3] = {p5[0]-p3[0], p5[1]-p3[1], p5[2]-p3[2]};
    box.wrap(d53[0], d53[1], d53[2]);

    // ---- Stacking (b = 5', a = 3'; sites along a1/b1) ----
    {
        const F1Params &f1p = par.stk_f1;
        c_number ra_cstk[3] = {dcstk*b1[0], dcstk*b1[1], dcstk*b1[2]};   // 3' (a)
        c_number rb_cstk[3] = {dcstk*a1[0], dcstk*a1[1], dcstk*a1[2]};   // 5' (b)
        c_number drs[3] = {d53[0]+rb_cstk[0]-ra_cstk[0], d53[1]+rb_cstk[1]-ra_cstk[1], d53[2]+rb_cstk[2]-ra_cstk[2]};
        c_number r_stk = Kokkos::sqrt(drs[0]*drs[0]+drs[1]*drs[1]+drs[2]*drs[2]);
        c_number f1 = MFOxdna::F1(r_stk, f1p.eps, f1p.a, f1p.cut_0, f1p.cut_lc, f1p.cut_hc,
                                  f1p.cut_lo, f1p.cut_hi, f1p.b_lo, f1p.b_hi, f1p.shift);
        if (f1 != 0) {
            c_number rinv = 1 / r_stk;
            c_number n_stk[3] = {drs[0]*rinv, drs[1]*rinv, drs[2]*rinv};
            const c_number *az_a = b3, *az_b = a3, *ay_a = b2, *ay_b = a2;

            c_number cost4 = az_a[0]*az_b[0]+az_a[1]*az_b[1]+az_a[2]*az_b[2];
            if (cost4> 1) cost4= 1; if (cost4<-1) cost4=-1;
            c_number theta4 = Kokkos::acos(cost4);
            c_number f4t4 = MFOxdna::F4(theta4, par.stk_t4.a, par.stk_t4.theta_0, par.stk_t4.dtheta_ast, par.stk_t4.b, par.stk_t4.dtheta_c);
            if (f4t4 != 0) {
                c_number cost5 = n_stk[0]*az_b[0]+n_stk[1]*az_b[1]+n_stk[2]*az_b[2];
                if (cost5> 1) cost5= 1; if (cost5<-1) cost5=-1;
                c_number theta5 = Kokkos::acos(cost5);
                c_number f4t5 = MFOxdna::F4(theta5, par.stk_t5.a, par.stk_t5.theta_0, par.stk_t5.dtheta_ast, par.stk_t5.b, par.stk_t5.dtheta_c);
                if (f4t5 != 0) {
                    c_number cost6 = n_stk[0]*az_a[0]+n_stk[1]*az_a[1]+n_stk[2]*az_a[2];
                    if (cost6> 1) cost6= 1; if (cost6<-1) cost6=-1;
                    c_number theta6 = Kokkos::acos(cost6);

                    c_number ra_cbk[3] = {par.d_cbk*b1[0], par.d_cbk*b1[1], par.d_cbk*b1[2]};   // 3' POS_BACK ref
                    c_number rb_cbk[3] = {par.d_cbk*a1[0], par.d_cbk*a1[1], par.d_cbk*a1[2]};   // 5' POS_BACK ref
                    c_number drb[3] = {d53[0]+rb_cbk[0]-ra_cbk[0], d53[1]+rb_cbk[1]-ra_cbk[1], d53[2]+rb_cbk[2]-ra_cbk[2]};
                    c_number rinv_bk = 1 / Kokkos::sqrt(drb[0]*drb[0]+drb[1]*drb[1]+drb[2]*drb[2]);
                    c_number n_bk[3] = {drb[0]*rinv_bk, drb[1]*rinv_bk, drb[2]*rinv_bk};

                    c_number cosphi1 = n_bk[0]*ay_b[0]+n_bk[1]*ay_b[1]+n_bk[2]*ay_b[2];
                    c_number cosphi2 = n_bk[0]*ay_a[0]+n_bk[1]*ay_a[1]+n_bk[2]*ay_a[2];
                    if (cosphi1> 1) cosphi1= 1; if (cosphi1<-1) cosphi1=-1;
                    if (cosphi2> 1) cosphi2= 1; if (cosphi2<-1) cosphi2=-1;

                    c_number f4t6 = MFOxdna::F4(theta6, par.stk_t6.a, par.stk_t6.theta_0, par.stk_t6.dtheta_ast, par.stk_t6.b, par.stk_t6.dtheta_c);
                    c_number f5c1 = MFOxdna::F5(-cosphi1, par.stk_cp1.a, par.stk_cp1.x_ast, par.stk_cp1.b, par.stk_cp1.x_c);
                    c_number f5c2 = MFOxdna::F5(-cosphi2, par.stk_cp2.a, par.stk_cp2.x_ast, par.stk_cp2.b, par.stk_cp2.x_c);
                    c_number est = f1*f4t4*f4t5*f4t6*f5c1*f5c2;
                    if (est != 0) {
                        energy += est;
                        c_number df1 = MFOxdna::DF1(r_stk, f1p.eps, f1p.a, f1p.cut_0, f1p.cut_lc, f1p.cut_hc, f1p.cut_lo, f1p.cut_hi, f1p.b_lo, f1p.b_hi);
                        c_number sT4 = Kokkos::sqrt(1-cost4*cost4);
                        c_number df4t4 = (sT4>1e-12)? MFOxdna::DF4(theta4,par.stk_t4.a,par.stk_t4.theta_0,par.stk_t4.dtheta_ast,par.stk_t4.b,par.stk_t4.dtheta_c)/sT4 : c_number(0);
                        c_number sT5 = Kokkos::sqrt(1-cost5*cost5);
                        c_number df4t5 = (sT5>1e-12)? MFOxdna::DF4(theta5,par.stk_t5.a,par.stk_t5.theta_0,par.stk_t5.dtheta_ast,par.stk_t5.b,par.stk_t5.dtheta_c)/sT5 : c_number(0);
                        c_number sT6 = Kokkos::sqrt(1-cost6*cost6);
                        c_number df4t6 = (sT6>1e-12)? MFOxdna::DF4(theta6,par.stk_t6.a,par.stk_t6.theta_0,par.stk_t6.dtheta_ast,par.stk_t6.b,par.stk_t6.dtheta_c)/sT6 : c_number(0);
                        c_number df5c1 = MFOxdna::DF5(-cosphi1, par.stk_cp1.a, par.stk_cp1.x_ast, par.stk_cp1.b, par.stk_cp1.x_c);
                        c_number df5c2 = MFOxdna::DF5(-cosphi2, par.stk_cp2.a, par.stk_cp2.x_ast, par.stk_cp2.b, par.stk_cp2.x_c);

                        // stacking-site force (on 5' end = +delf_s)
                        c_number delf_s[3] = {0,0,0}, finc;
                        finc = -df1*f4t4*f4t5*f4t6*f5c1*f5c2;
                        delf_s[0]+=drs[0]*finc; delf_s[1]+=drs[1]*finc; delf_s[2]+=drs[2]*finc;
                        if (theta5 != 0) { finc = -f1*f4t4*df4t5*f4t6*f5c1*f5c2*rinv;
                            delf_s[0]+=(n_stk[0]*cost5-az_b[0])*finc; delf_s[1]+=(n_stk[1]*cost5-az_b[1])*finc; delf_s[2]+=(n_stk[2]*cost5-az_b[2])*finc; }
                        if (theta6 != 0) { finc = -f1*f4t4*f4t5*df4t6*f5c1*f5c2*rinv;
                            delf_s[0]+=(n_stk[0]*cost6-az_a[0])*finc; delf_s[1]+=(n_stk[1]*cost6-az_a[1])*finc; delf_s[2]+=(n_stk[2]*cost6-az_a[2])*finc; }
                        // backbone-site force
                        c_number delf_b[3] = {0,0,0};
                        if (cosphi1 != 0) { finc = -f1*f4t4*f4t5*f4t6*df5c1*f5c2*rinv_bk;
                            delf_b[0]+=(n_bk[0]*cosphi1-ay_b[0])*finc; delf_b[1]+=(n_bk[1]*cosphi1-ay_b[1])*finc; delf_b[2]+=(n_bk[2]*cosphi1-ay_b[2])*finc; }
                        if (cosphi2 != 0) { finc = -f1*f4t4*f4t5*f4t6*f5c1*df5c2*rinv_bk;
                            delf_b[0]+=(n_bk[0]*cosphi2-ay_a[0])*finc; delf_b[1]+=(n_bk[1]*cosphi2-ay_a[1])*finc; delf_b[2]+=(n_bk[2]*cosphi2-ay_a[2])*finc; }

                        // forces: 5' end gets +delf_s+delf_b
                        F[0]+=delf_s[0]+delf_b[0]; F[1]+=delf_s[1]+delf_b[1]; F[2]+=delf_s[2]+delf_b[2];
                        c_number c[3];
                        // site torques: T3 -= ra_site x delf ; T5 += rb_site x delf
                        bx_cross(ra_cstk, delf_s, c); T3[0]-=c[0]; T3[1]-=c[1]; T3[2]-=c[2];
                        bx_cross(rb_cstk, delf_s, c); T5[0]+=c[0]; T5[1]+=c[1]; T5[2]+=c[2];
                        bx_cross(ra_cbk,  delf_b, c); T3[0]-=c[0]; T3[1]-=c[1]; T3[2]-=c[2];
                        bx_cross(rb_cbk,  delf_b, c); T5[0]+=c[0]; T5[1]+=c[1]; T5[2]+=c[2];

                        // pure torques: delta -> 3' (T3 -= delta), deltb -> 5' (T5 += deltb)
                        c_number delta[3]={0,0,0}, deltb[3]={0,0,0}, tp, d[3];
                        if (theta4 != 0) { tp = -f1*df4t4*f4t5*f4t6*f5c1*f5c2; bx_cross(az_a,az_b,d);
                            delta[0]+=d[0]*tp; delta[1]+=d[1]*tp; delta[2]+=d[2]*tp;
                            deltb[0]+=d[0]*tp; deltb[1]+=d[1]*tp; deltb[2]+=d[2]*tp; }
                        if (theta5 != 0) { tp = -f1*f4t4*df4t5*f4t6*f5c1*f5c2; bx_cross(n_stk,az_b,d);
                            deltb[0]+=d[0]*tp; deltb[1]+=d[1]*tp; deltb[2]+=d[2]*tp; }
                        if (theta6 != 0) { tp = -f1*f4t4*f4t5*df4t6*f5c1*f5c2; bx_cross(n_stk,az_a,d);
                            delta[0]-=d[0]*tp; delta[1]-=d[1]*tp; delta[2]-=d[2]*tp; }
                        if (cosphi1 != 0) { tp = -f1*f4t4*f4t5*f4t6*df5c1*f5c2; bx_cross(n_bk,ay_b,d);
                            deltb[0]+=d[0]*tp; deltb[1]+=d[1]*tp; deltb[2]+=d[2]*tp; }
                        if (cosphi2 != 0) { tp = -f1*f4t4*f4t5*f4t6*f5c1*df5c2; bx_cross(n_bk,ay_a,d);
                            delta[0]-=d[0]*tp; delta[1]-=d[1]*tp; delta[2]-=d[2]*tp; }
                        T3[0]-=delta[0]; T3[1]-=delta[1]; T3[2]-=delta[2];
                        T5[0]+=deltb[0]; T5[1]+=deltb[1]; T5[2]+=deltb[2];
                    }
                }
            }
        }
    }
    return energy;
}

KOKKOS_INLINE_FUNCTION
void bonded_load_frame(const Vec4cr &nx, const Vec4cr &ny, const Vec4cr &nz, int i,
                       c_number (&a1)[3], c_number (&a2)[3], c_number (&a3)[3]) {
    a1[0]=nx(i,0); a1[1]=nx(i,1); a1[2]=nx(i,2);
    a2[0]=ny(i,0); a2[1]=ny(i,1); a2[2]=ny(i,2);
    a3[0]=nz(i,0); a3[1]=nz(i,1); a3[2]=nz(i,2);
}

// Gather functor template: one thread per particle, no atomics. Reads the
// PRECOMPUTED body frames (nx/ny/nz) from the LRF pass. The template parameter
// FENE selects which half-interaction this kernel computes:
//   FENE=true  -> bonded_fene_excv (LAMMPS `bond oxdna/fene`)
//   FENE=false -> bonded_stk        (LAMMPS `pair oxdna/stk`)
template <bool FENE>
struct BondedTermFunctor {
    Vec4cr poss;
    Vec4cr nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    Vec4 forces;
    Vec4 torques;
    DNAParams par;
    SimBox box;

    KOKKOS_INLINE_FUNCTION
    c_number bond_call(const c_number p5[3], const c_number a1[3], const c_number a2[3], const c_number a3[3],
                       const c_number p3[3], const c_number b1[3], const c_number b2[3], const c_number b3[3],
                       c_number (&F)[3], c_number (&T5)[3], c_number (&T3)[3]) const {
        if (FENE) return bonded_fene_excv(p5,a1,a2,a3,p3,b1,b2,b3,par,box,F,T5,T3);
        else      return bonded_stk      (p5,a1,a2,a3,p3,b1,b2,b3,par,box,F,T5,T3);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const { c_number ev=0; (*this)(i, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, c_number &ev) const {
        int n3 = bonds(i).n3;
        int n5 = bonds(i).n5;
        if (n3 < 0 && n5 < 0) return;

        c_number ai1[3], ai2[3], ai3[3];
        bonded_load_frame(nx, ny, nz, i, ai1, ai2, ai3);
        c_number pi[3] = {poss(i,0), poss(i,1), poss(i,2)};

        c_number F[3] = {0,0,0}, Tt[3] = {0,0,0};

        // bond (i = 5', n3 = 3'): take 5'-end contribution; count energy here
        if (n3 >= 0) {
            c_number bj1[3], bj2[3], bj3[3];
            bonded_load_frame(nx, ny, nz, n3, bj1, bj2, bj3);
            c_number pj[3] = {poss(n3,0), poss(n3,1), poss(n3,2)};
            c_number F5[3] = {0,0,0}, T5[3] = {0,0,0}, T3[3] = {0,0,0};
            ev += bond_call(pi, ai1, ai2, ai3, pj, bj1, bj2, bj3, F5, T5, T3);
            F[0]+=F5[0]; F[1]+=F5[1]; F[2]+=F5[2];
            Tt[0]+=T5[0]; Tt[1]+=T5[1]; Tt[2]+=T5[2];
        }
        // bond (n5 = 5', i = 3'): take 3'-end contribution (force = -F5); energy counted by n5
        if (n5 >= 0) {
            c_number bj1[3], bj2[3], bj3[3];
            bonded_load_frame(nx, ny, nz, n5, bj1, bj2, bj3);
            c_number pj[3] = {poss(n5,0), poss(n5,1), poss(n5,2)};
            c_number F5[3] = {0,0,0}, T5[3] = {0,0,0}, T3[3] = {0,0,0};
            bond_call(pj, bj1, bj2, bj3, pi, ai1, ai2, ai3, F5, T5, T3);
            F[0]-=F5[0]; F[1]-=F5[1]; F[2]-=F5[2];
            Tt[0]+=T3[0]; Tt[1]+=T3[1]; Tt[2]+=T3[2];
        }

        forces(i,0)+=F[0];  forces(i,1)+=F[1];  forces(i,2)+=F[2];
        torques(i,0)+=Tt[0]; torques(i,1)+=Tt[1]; torques(i,2)+=Tt[2];
    }
};

// ---------------------------------------------------------------------------
// LAMMPS-overhead-mode bonded kernel: faithful per-BOND + ATOMIC scatter (each
// bond computed ONCE by its 5' end, force/torque atomically scattered to both
// endpoints), mirroring LAMMPS `pair oxdna/stk` and `bond oxdna/fene` (which run
// over nbondlist with atomic dup_f/dup_torque) - as opposed to the lean
// per-particle gather above (each bond computed twice, no atomics). Also models
// LAMMPS's 4D sequence-dependent ("tetramer") coefficient indexing: 4 extra type
// reads + uniform table lookups per bond (physics unchanged; only the memory
// traffic is reproduced). Selected when lammps_overhead is on.
// ---------------------------------------------------------------------------
template <bool FENE>
struct BondedScatterFunctor {
    Vec4cr poss;
    Vec4cr nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    RandomRead<int> btype;
    Kokkos::View<const c_number *> tet;   // 256-entry uniform (==1) tetramer table
    ScatterF4 sf, st;                     // atomic force/torque (like LAMMPS dup_f)
    DNAParams par;
    SimBox box;

    KOKKOS_INLINE_FUNCTION
    c_number bond_call(const c_number p5[3], const c_number a1[3], const c_number a2[3], const c_number a3[3],
                       const c_number p3[3], const c_number b1[3], const c_number b2[3], const c_number b3[3],
                       c_number (&F)[3], c_number (&T5)[3], c_number (&T3)[3]) const {
        if (FENE) return bonded_fene_excv(p5,a1,a2,a3,p3,b1,b2,b3,par,box,F,T5,T3);
        else      return bonded_stk      (p5,a1,a2,a3,p3,b1,b2,b3,par,box,F,T5,T3);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const { c_number ev=0; (*this)(i, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, c_number &ev) const {
        const int n3 = bonds(i).n3;
        if (n3 < 0) return;          // i is the 5' end of bond (i,n3); process once
        const int j = n3;            // 3' end

        // Tetramer coefficient indexing overhead (4 type reads + 4D table lookups);
        // tet is uniform (==1) so cf==1 and physics is unchanged.
        const int n5i = bonds(i).n5, n3j = bonds(j).n3;
        const int ta = btype(i), tb = btype(j);
        const int t3 = (n5i >= 0) ? btype(n5i) : 0;
        const int t4 = (n3j >= 0) ? btype(n3j) : 0;
        const int idx = (((t3 & 3) * 4 + (ta & 3)) * 4 + (tb & 3)) * 4 + (t4 & 3);
        const c_number cf = tet(idx) * tet((idx + 1) & 255) * tet((idx + 2) & 255)
                          * tet((idx + 3) & 255) * tet((idx + 4) & 255);

        c_number ai1[3], ai2[3], ai3[3], bj1[3], bj2[3], bj3[3];
        bonded_load_frame(nx, ny, nz, i, ai1, ai2, ai3);
        bonded_load_frame(nx, ny, nz, j, bj1, bj2, bj3);
        c_number pi[3] = {poss(i,0), poss(i,1), poss(i,2)};
        c_number pj[3] = {poss(j,0), poss(j,1), poss(j,2)};
        c_number F5[3] = {0,0,0}, T5[3] = {0,0,0}, T3[3] = {0,0,0};
        ev += cf * bond_call(pi, ai1, ai2, ai3, pj, bj1, bj2, bj3, F5, T5, T3);

        auto af = sf.access();
        auto at = st.access();
        af(i,0)+=cf*F5[0]; af(i,1)+=cf*F5[1]; af(i,2)+=cf*F5[2];
        at(i,0)+=T5[0];    at(i,1)+=T5[1];    at(i,2)+=T5[2];
        af(j,0)-=cf*F5[0]; af(j,1)-=cf*F5[1]; af(j,2)-=cf*F5[2];
        at(j,0)+=T3[0];    at(j,1)+=T3[1];    at(j,2)+=T3[2];
    }
};

template <bool FENE>
inline c_number run_bonded_term_scatter(ParticleArrays &p, const DNAParams &par,
                                        const SimBox &box, bool want_energy,
                                        const char *label) {
    ScatterF4 sf(p.forces);
    ScatterF4 st(p.torques);
    BondedScatterFunctor<FENE> fun;
    fun.poss = p.poss; fun.nx = p.nx; fun.ny = p.ny; fun.nz = p.nz;
    fun.bonds = p.bonds; fun.btype = p.btype; fun.tet = p.tetramer_tbl;
    fun.sf = sf; fun.st = st; fun.par = par; fun.box = box;
    using BondPolicy = Kokkos::RangePolicy<Kokkos::LaunchBounds<OXDNA_BOND_MAXT, OXDNA_BOND_MINB>>;
    c_number etot = 0;
    if (want_energy) Kokkos::parallel_reduce(label, BondPolicy(0, p.N), fun, etot);
    else             Kokkos::parallel_for(label, BondPolicy(0, p.N), fun);
    Kokkos::Experimental::contribute(p.forces, sf);
    Kokkos::Experimental::contribute(p.torques, st);
    return etot;
}

// ---------------------------------------------------------------------------
// LAMMPS-overhead-mode: per-step bond "prime-neigh" precompute (no-op for the
// physics; faithfully reproduces the LAMMPS kernel that re-derives the 3'/5'
// bonded-neighbour indices from the bond list + atom map every step, which the
// lean standalone skips because it stores bonds(i).n3/n5 directly). One thread
// per particle reads its bonded neighbours (and theirs) and writes a scratch
// table - matching the launch + per-bond memory pass of
// TagPairOxdnaStkPrecomputeBondPrimeNeighs / the FENE equivalent.
// ---------------------------------------------------------------------------
struct BondPrecomputeFunctor {
    Kokkos::View<LR_bonds *> bonds;
    Kokkos::View<int *[4]>   prime;
    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
        const int n3 = bonds(i).n3;
        const int n5 = bonds(i).n5;
        prime(i,0) = n3;
        prime(i,1) = n5;
        prime(i,2) = (n3 >= 0) ? bonds(n3).n5 : -1;
        prime(i,3) = (n5 >= 0) ? bonds(n5).n3 : -1;
    }
};
inline void bond_precompute(ParticleArrays &p, const char *label) {
    BondPrecomputeFunctor f; f.bonds = p.bonds; f.prime = p.bond_prime_neighs;
    Kokkos::parallel_for(label, p.N, f);
}

// Run one bonded term kernel. NOTE: requires the LRF precompute (compute_lrf)
// to have populated p.nx/ny/nz first (done by compute_nonbonded_forces).
template <bool FENE>
inline c_number run_bonded_term(ParticleArrays &p, const DNAParams &par,
                                const SimBox &box, bool want_energy,
                                const char *label) {
    BondedTermFunctor<FENE> fun;
    fun.poss = p.poss; fun.nx = p.nx; fun.ny = p.ny; fun.nz = p.nz;
    fun.bonds = p.bonds;
    fun.forces = p.forces; fun.torques = p.torques; fun.par = par; fun.box = box;
    using BondPolicy = Kokkos::RangePolicy<Kokkos::LaunchBounds<OXDNA_BOND_MAXT, OXDNA_BOND_MINB>>;
    c_number etot = 0;
    if (want_energy) Kokkos::parallel_reduce(label, BondPolicy(0, p.N), fun, etot);
    else             Kokkos::parallel_for(label, BondPolicy(0, p.N), fun);
    return etot;
}

// LAMMPS-faithful bonded driver: TWO separate kernels (stk, then fene),
// mirroring `pair oxdna/stk` + `bond oxdna/fene`. Same public signature as
// the original fused driver (so fd_test / xcheck build unchanged).
inline c_number compute_bonded_forces(ParticleArrays &p, const DNAParams &par,
                                      const SimBox &box, bool want_energy = true,
                                      bool lammps_overhead = false,
                                      bool neigh_rebuilt = true) {
    // Ensure the precomputed body frames (nx/ny/nz) are current. In the normal
    // per-step sequence compute_nonbonded_forces already ran the LRF pass, but
    // calling it here too keeps this entry point self-contained (the validation
    // tools may invoke the bonded kernels on their own).
    compute_lrf(p);
    c_number e = 0;
    if (lammps_overhead) {
        // Faithful LAMMPS bonded: per-bond + atomic scatter + tetramer indexing.
        // The bond->prime-neigh precompute is a no-op for the physics (the table
        // it writes is never read here); LAMMPS now caches it per neighbor build
        // (gated on neighbor->lastcall), so only launch it on rebuild steps to
        // match the optimized per-step kernel count.
        if (neigh_rebuilt) bond_precompute(p, "oxdna_stk_precompute");
        e += run_bonded_term_scatter<false>(p, par, box, want_energy, "oxdna_stk");
        if (neigh_rebuilt) bond_precompute(p, "oxdna_fene_precompute");
        e += run_bonded_term_scatter<true> (p, par, box, want_energy, "oxdna_fene");
    } else {
        // Lean (CUDA-standalone) bonded: per-particle gather, no atomics.
        e += run_bonded_term<false>(p, par, box, want_energy, "oxdna_stk");   // stacking
        e += run_bonded_term<true> (p, par, box, want_energy, "oxdna_fene");  // FENE + bonded excv
    }
    // LAMMPS bond/fene copies its 1-int overstretch flag device->host. It now
    // throttles that copy to output/thermo steps (eflag||vflag) instead of every
    // step, so reproduce the host round-trip only when energy is requested.
    if (lammps_overhead && want_energy) {
        Kokkos::deep_copy(p.overstretch_flag_host, p.overstretch_flag);
    }
    return e;
}
