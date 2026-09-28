#pragma once

// Bonded interactions in the LAMMPS-faithful structure (tracks LAMMPS
// origin/oxdna3KK): two separate styles, `pair oxdna*/stk` (stacking) and
// `bond oxdna*/fene` (FENE only -- the bonded excluded volume is computed by
// the excv kernel from the special 1-2 neighbours, as in LAMMPS).
//
// Two execution models:
//   * lean (default): one thread per particle GATHERS its two bonds (each bond
//     evaluated twice, no atomics), minimising framework cost;
//   * lammps_overhead: one thread per BOND over the bond list with atomic
//     scatter to both ends, reading atoms and tetramer context from the
//     fix OXDNA/PRIME_NEIGHS bond table, like the LAMMPS kernels.
// Both use a plain RangePolicy (LAMMPS stk/fene have no launch bounds).
//
// compute_pair_forces_step() / compute_bond_forces_step() at the end run a
// whole MD step's force evaluation in LAMMPS order:
//   LRF -> [prime_neighs_bond] -> excv -> [prime_neighs_bond] stk -> hbond ->
//   xstk -> coaxstk -> dh      |  bond: fene (+ overstretch flag copy)

#include "../types.h"
#include "../particles.h"
#include "params.h"
#include "orient.h"
#include "mf_oxdna.h"
#include "dna_forces.h"   // compute_lrf, ScatterF4, the nonbonded launchers
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
void bx_cross(const c_number a[3], const c_number b[3], c_number c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

// -----------------------------------------------------------------------
// One bond, 5' end = (p5, a1/a2/a3), 3' end = (p3, b1/b2/b3). Each helper
// accumulates the force on the 5' end into F, the torques on the 5' / 3' ends
// into T5 / T3, and returns the bond energy (force on the 3' end is -F).
// -----------------------------------------------------------------------

// ---- FENE only (LAMMPS `bond oxdna/fene`) ----
// The log() for the energy is only evaluated on energy steps (LAMMPS evaluates
// it only when eflag). overstretched is set when the bond is stretched past
// LAMMPS' rlogarg < 0.2 warning threshold (the physics keeps the standalone
// oxDNA clamp; only the flag is modelled).
KOKKOS_INLINE_FUNCTION
c_number bonded_fene(const c_number p5[3], const c_number a1[3], const c_number a2[3],
                     const c_number p3[3], const c_number b1[3], const c_number b2[3],
                     const DNAParams &par, const SimBox &box, bool want_energy,
                     c_number (&F)[3], c_number (&T5)[3], c_number (&T3)[3],
                     bool &overstretched) {
    const c_number pb1 = par.pb1, pb2 = par.pb2;
    c_number d53[3] = {p5[0]-p3[0], p5[1]-p3[1], p5[2]-p3[2]};
    box.wrap(d53[0], d53[1], d53[2]);
    c_number r5bk[3] = {pb1*a1[0]+pb2*a2[0], pb1*a1[1]+pb2*a2[1], pb1*a1[2]+pb2*a2[2]};
    c_number r3bk[3] = {pb1*b1[0]+pb2*b2[0], pb1*b1[1]+pb2*b2[1], pb1*b1[2]+pb2*b2[2]};

    c_number energy = 0;
    c_number dx = d53[0]+r5bk[0]-r3bk[0];
    c_number dy = d53[1]+r5bk[1]-r3bk[1];
    c_number dz = d53[2]+r5bk[2]-r3bk[2];
    c_number r  = Kokkos::sqrt(dx*dx+dy*dy+dz*dz);
    c_number t  = (r - par.fene.r0) / par.fene.Delta;
    if (Kokkos::fabs(t) < c_number(1.0)) {
        c_number tm = c_number(0.9998);
        if (t > tm) t = tm; if (t < -tm) t = -tm;
        c_number denom = 1 - t*t;
        if (denom < c_number(0.2)) overstretched = true;
        if (want_energy) energy = c_number(-0.5)*par.fene.k*Kokkos::log(denom);
        c_number fpair = -par.fene.k * t / (r * par.fene.Delta * denom);
        c_number df[3] = {dx*fpair, dy*fpair, dz*fpair};
        F[0]+=df[0]; F[1]+=df[1]; F[2]+=df[2];
        c_number c[3];
        bx_cross(r5bk, df, c); T5[0]+=c[0]; T5[1]+=c[1]; T5[2]+=c[2];
        bx_cross(r3bk, df, c); T3[0]-=c[0]; T3[1]-=c[1]; T3[2]-=c[2];
    } else {
        overstretched = true;
    }
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
        c_number r_stk = Kokkos::sqrt(Kokkos::fma(drs[2], drs[2], Kokkos::fma(drs[1], drs[1], drs[0]*drs[0])));
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
                    c_number rinv_bk = 1 / Kokkos::sqrt(Kokkos::fma(drb[2], drb[2], Kokkos::fma(drb[1], drb[1], drb[0]*drb[0])));
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

template <bool FENE>
KOKKOS_INLINE_FUNCTION
c_number bond_term(const c_number p5[3], const c_number a1[3], const c_number a2[3], const c_number a3[3],
                   const c_number p3[3], const c_number b1[3], const c_number b2[3], const c_number b3[3],
                   const DNAParams &par, const SimBox &box, bool want_energy,
                   c_number (&F)[3], c_number (&T5)[3], c_number (&T3)[3], bool &overstretched) {
    if (FENE) return bonded_fene(p5, a1, a2, p3, b1, b2, par, box, want_energy, F, T5, T3, overstretched);
    else      return bonded_stk (p5, a1, a2, a3, p3, b1, b2, b3, par, box, F, T5, T3);
}

// -----------------------------------------------------------------------
// Lean gather kernel: one thread per particle, no atomics. FENE selects
//   FENE=true  -> bond oxdna/fene,  FENE=false -> pair oxdna/stk.
// -----------------------------------------------------------------------
template <bool FENE>
struct BondedTermFunctor {
    Vec4cr poss;
    Vec4cr nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    VecA4 forces;
    VecA4 torques;
    DNAParams par;
    SimBox box;
    bool want_energy;

    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const { c_acc ev=0; (*this)(i, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int i, c_acc &ev) const {
        int n3 = bonds(i).n3;
        int n5 = bonds(i).n5;
        if (n3 < 0 && n5 < 0) return;

        c_number ai1[3], ai2[3], ai3[3];
        bonded_load_frame(nx, ny, nz, i, ai1, ai2, ai3);
        c_number pi[3] = {poss(i,0), poss(i,1), poss(i,2)};

        c_acc F[3] = {0,0,0}, Tt[3] = {0,0,0};
        bool over = false;

        // bond (i = 5', n3 = 3'): take 5'-end contribution; count energy here
        if (n3 >= 0) {
            c_number bj1[3], bj2[3], bj3[3];
            bonded_load_frame(nx, ny, nz, n3, bj1, bj2, bj3);
            c_number pj[3] = {poss(n3,0), poss(n3,1), poss(n3,2)};
            c_number F5[3] = {0,0,0}, T5[3] = {0,0,0}, T3[3] = {0,0,0};
            ev += bond_term<FENE>(pi, ai1, ai2, ai3, pj, bj1, bj2, bj3, par, box, want_energy, F5, T5, T3, over);
            F[0]+=F5[0]; F[1]+=F5[1]; F[2]+=F5[2];
            Tt[0]+=T5[0]; Tt[1]+=T5[1]; Tt[2]+=T5[2];
        }
        // bond (n5 = 5', i = 3'): take 3'-end contribution (force = -F5); energy counted by n5
        if (n5 >= 0) {
            c_number bj1[3], bj2[3], bj3[3];
            bonded_load_frame(nx, ny, nz, n5, bj1, bj2, bj3);
            c_number pj[3] = {poss(n5,0), poss(n5,1), poss(n5,2)};
            c_number F5[3] = {0,0,0}, T5[3] = {0,0,0}, T3[3] = {0,0,0};
            bond_term<FENE>(pj, bj1, bj2, bj3, pi, ai1, ai2, ai3, par, box, false, F5, T5, T3, over);
            F[0]-=F5[0]; F[1]-=F5[1]; F[2]-=F5[2];
            Tt[0]+=T3[0]; Tt[1]+=T3[1]; Tt[2]+=T3[2];
        }

        forces(i,0)+=F[0];  forces(i,1)+=F[1];  forces(i,2)+=F[2];
        torques(i,0)+=Tt[0]; torques(i,1)+=Tt[1]; torques(i,2)+=Tt[2];
    }
};

// -----------------------------------------------------------------------
// lammps_overhead: per-BOND kernel with atomic scatter, as LAMMPS pair
// oxdna/stk and bond oxdna/fene. Reads the bond's atoms and tetramer context
// (a = 3' end, b = 5' end, a3p, b5p) only from the fix OXDNA/PRIME_NEIGHS
// table, then the 4 type reads + 4D coefficient lookups (the 256-entry table is
// uniform 1.0, so physics is unchanged). stk returns before any atomics when
// the bond does not stack (LAMMPS' f1/f4t4/f4t5/evdwl == 0 early returns);
// fene always scatters and raises the device overstretch flag.
// -----------------------------------------------------------------------
template <bool FENE>
struct BondedScatterFunctor {
    Vec4cr poss;
    Vec4cr nx, ny, nz;
    Kokkos::View<const int *[4], Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::RandomAccess>> prime;
    RandomRead<int> btype;
    Kokkos::View<const c_number *> tet;
    Kokkos::View<int> flag;
    ScatterF4 sf, st;
    DNAParams par;
    SimBox box;
    bool want_energy;

    KOKKOS_INLINE_FUNCTION
    void operator()(int in) const { c_acc ev=0; (*this)(in, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int in, c_acc &ev) const {
        const int a = prime(in,0), b = prime(in,1);          // a = 3' end, b = 5' end
        const int a3p = prime(in,2), b5p = prime(in,3);
        const int ta = btype(a), tb = btype(b);
        const int t3 = (a3p >= 0) ? btype(a3p) : 0;
        const int t5 = (b5p >= 0) ? btype(b5p) : 0;
        const int idx = (((t3 & 3) * 4 + (ta & 3)) * 4 + (tb & 3)) * 4 + (t5 & 3);
        const c_number cf = tet(idx) * tet((idx + 1) & 255) * tet((idx + 2) & 255)
                          * tet((idx + 3) & 255) * tet((idx + 4) & 255);

        c_number bi1[3], bi2[3], bi3[3], aj1[3], aj2[3], aj3[3];
        bonded_load_frame(nx, ny, nz, b, bi1, bi2, bi3);
        bonded_load_frame(nx, ny, nz, a, aj1, aj2, aj3);
        c_number pb[3] = {poss(b,0), poss(b,1), poss(b,2)};
        c_number pa[3] = {poss(a,0), poss(a,1), poss(a,2)};
        c_number F5[3] = {0,0,0}, T5[3] = {0,0,0}, T3[3] = {0,0,0};
        bool over = false;
        const c_number e = cf * bond_term<FENE>(pb, bi1, bi2, bi3, pa, aj1, aj2, aj3,
                                                par, box, want_energy || !FENE, F5, T5, T3, over);
        if (!FENE && e == 0) return;
        if (FENE && over) flag() = 1;
        ev += e;

        auto af = sf.access();
        auto at = st.access();
        af(b,0)+=cf*F5[0]; af(b,1)+=cf*F5[1]; af(b,2)+=cf*F5[2];
        at(b,0)+=T5[0];    at(b,1)+=T5[1];    at(b,2)+=T5[2];
        af(a,0)-=cf*F5[0]; af(a,1)-=cf*F5[1]; af(a,2)-=cf*F5[2];
        at(a,0)+=T3[0];    at(a,1)+=T3[1];    at(a,2)+=T3[2];
    }
};

// fix OXDNA/PRIME_NEIGHS::compute_prime_neighs_bond (lammps_overhead mode,
// neighbor-rebuild steps): one thread per bond over the bond list; direction
// test, then (a, b, a3p, b5p) into the table. LAMMPS launches it twice per
// rebuild (the fix's pre_force, and again from pair oxdna/stk, which keeps its
// own rebuild counter); bond oxdna/fene only reads the table.
inline void bond_precompute(ParticleArrays &p, const char *label) {
    auto bl = p.bondlist; auto bonds = p.bonds; auto tab = p.prime_bond;
    Kokkos::parallel_for(label, p.nbonds, KOKKOS_LAMBDA(int in) {
        const int i5 = bl(in,0), j3 = bl(in,1);          // 5' end, 3' end
        const bool fwd = (bonds(i5).n3 == j3);           // direction test (tag/id5p)
        const int a = fwd ? j3 : i5, b = fwd ? i5 : j3;
        tab(in,0) = a;
        tab(in,1) = b;
        tab(in,2) = bonds(a).n3;
        tab(in,3) = bonds(b).n5;
    });
}

template <bool FENE>
inline c_acc run_bonded_term(ParticleArrays &p, const DNAParams &par, const SimBox &box,
                             bool want_energy, bool lammps_overhead, const char *label) {
    using Plain = Kokkos::RangePolicy<>;
    c_acc etot = 0;
    if (lammps_overhead) {
        BondedScatterFunctor<FENE> f;
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
        f.prime = p.prime_bond; f.btype = p.btype; f.tet = p.tetramer_tbl;
        f.flag = p.overstretch_flag;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        f.par = par; f.box = box; f.want_energy = want_energy;
        if (p.nbonds <= 0) return etot;
        if (want_energy) Kokkos::parallel_reduce(label, Plain(0, p.nbonds), f, etot);
        else             Kokkos::parallel_for(label, Plain(0, p.nbonds), f);
        return etot;
    }
    BondedTermFunctor<FENE> f;
    f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz; f.bonds = p.bonds;
    f.forces = p.forces; f.torques = p.torques; f.par = par; f.box = box;
    f.want_energy = want_energy;
    if (want_energy) Kokkos::parallel_reduce(label, Plain(0, p.N), f, etot);
    else             Kokkos::parallel_for(label, Plain(0, p.N), f);
    return etot;
}

inline void ensure_bondlist(ParticleArrays &p) {
    if (p.bondlist.extent(0) == 0 && p.nbonds == 0) p.build_bondlist();
}

// bond oxdna/fene + its overstretch-flag host round trip, which LAMMPS only
// does on energy/virial steps (eflag || vflag), resetting the device flag
// when it was raised.
inline c_acc run_fene(ParticleArrays &p, const DNAParams &par, const SimBox &box,
                      bool want_energy, bool lammps_overhead) {
    c_acc e = run_bonded_term<true>(p, par, box, want_energy, lammps_overhead, "oxdna_fene");
    if (lammps_overhead && want_energy) {
        Kokkos::deep_copy(p.overstretch_flag_host, p.overstretch_flag);
        if (p.overstretch_flag_host() == 1) Kokkos::deep_copy(p.overstretch_flag, 0);
    }
    return e;
}

// Bonded driver for standalone callers (fd_test / xcheck): stk + fene.
inline c_number compute_bonded_forces(ParticleArrays &p, const DNAParams &par,
                                      const SimBox &box, bool want_energy = true,
                                      bool lammps_overhead = false,
                                      bool neigh_rebuilt = true,
                                      bool run_lrf = true) {
    if (run_lrf) compute_lrf(p);
    c_acc e = 0;
    if (lammps_overhead) {
        ensure_bondlist(p);
        if (neigh_rebuilt) bond_precompute(p, "oxdna_stk_prime_neighs_bond");
    }
    e += run_bonded_term<false>(p, par, box, want_energy, lammps_overhead, "oxdna_stk");
    e += run_fene(p, par, box, want_energy, lammps_overhead);
    return static_cast<c_number>(e);
}

// -----------------------------------------------------------------------
// Per-step drivers in LAMMPS order (Pair section, then Bond section).
// -----------------------------------------------------------------------
inline c_acc compute_pair_forces_step(ParticleArrays &p, const NeighborList &nl,
                                      const DNAParams &par, const SimBox &box,
                                      bool want_energy, bool lammps_overhead,
                                      bool fuse_hbond_xstk, bool neigh_rebuilt) {
    compute_lrf(p);                                                       // fix OXDNA/LRF
    if (lammps_overhead) {
        ensure_bondlist(p);
        if (neigh_rebuilt) {
            // LAMMPS neigh_bond build_topology_kk ends with a device->host
            // copy of the rebuilt bond list.
            auto h_bl = Kokkos::create_mirror_view(p.bondlist);
            Kokkos::deep_copy(h_bl, p.bondlist);
            bond_precompute(p, "oxdna_prime_neighs_bond");                // fix pre_force
        }
    }
    c_acc e = 0;
    e += run_excv(p, nl, par, box, want_energy, lammps_overhead, neigh_rebuilt);
    if (lammps_overhead && neigh_rebuilt)
        bond_precompute(p, "oxdna_stk_prime_neighs_bond");                // stk's own copy
    e += run_bonded_term<false>(p, par, box, want_energy, lammps_overhead, "oxdna_stk");
    e += run_hbond_xstk(p, nl, par, box, want_energy, fuse_hbond_xstk);
    e += run_coaxstk(p, nl, par, box, want_energy);
    e += run_dh(p, nl, par, box, want_energy);
    return e;
}

inline c_acc compute_bond_forces_step(ParticleArrays &p, const DNAParams &par,
                                      const SimBox &box, bool want_energy,
                                      bool lammps_overhead) {
    return run_fene(p, par, box, want_energy, lammps_overhead);
}
