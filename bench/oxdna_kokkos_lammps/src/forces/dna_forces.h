#pragma once

// oxDNA nonbonded physics helpers + the LAMMPS-faithful nonbonded kernels.
//
// The physics helpers (add_excv_contrib, hbond_pair, crst_pair, cxst_pair,
// dh_pair, eval_f4, ...) are the validated standalone-oxDNA ports shared with
// bench/oxdna_kokkos, so energies agree with it. The kernels at the end of this
// file reproduce the KERNEL STRUCTURE of the LAMMPS KOKKOS oxDNA styles
// (tracking LAMMPS origin/oxdna3KK-kk-fixes): which list each term iterates, what it
// reads, how it accumulates and scatters, and its launch policy.

#include "../types.h"

#include "../particles.h"
#include "../neighbor_list.h"
#include "orient.h"
#include "mf_oxdna.h"
#include "params.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <cmath>

using namespace MFOxdna;

// -----------------------------------------------------------------------
// Small vector helpers
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void cross3(const c_number a[3], const c_number b[3], c_number c[3]) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

KOKKOS_INLINE_FUNCTION
c_number dot3(const c_number a[3], const c_number b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// f4 value and its derivative w.r.t. cos(theta) (== standalone _custom_f4D).
KOKKOS_INLINE_FUNCTION
void eval_f4(c_number cost, const F4Params &P, c_number &f4v, c_number &Dc) {
    if (cost >  1) cost =  1;
    if (cost < -1) cost = -1;
    c_number th = Kokkos::acos(cost);
    f4v = F4(th, P.a, P.theta_0, P.dtheta_ast, P.b, P.dtheta_c);
    c_number sint = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost*cost));
    Dc = (sint > c_number(1e-8))
       ? (-DF4(th, P.a, P.theta_0, P.dtheta_ast, P.b, P.dtheta_c) / sint)
       : c_number(0);
}

// Coaxial theta1 (CXST_F4_THETA1): standalone _fakef4_cxst_t1.
//   mode 0 (oxDNA1): g = f4(t) + f4(2*pi - t)
//   mode 1 (oxDNA2): g = f4(t) + SA*(t-SB)^2   (t > SB)
// Returns g and Dc = dg/dcost (== standalone _custom_f4D for this mesh).
KOKKOS_INLINE_FUNCTION
void eval_cxst_t1(c_number cost, const F4Params &P, int mode, c_number SA, c_number SB,
                  c_number &g, c_number &Dc) {
    constexpr c_number PI = c_number(3.141592653589793);
    if (cost >  1) cost =  1;
    if (cost < -1) cost = -1;
    c_number t  = Kokkos::acos(cost);
    c_number f4b  = F4(t, P.a, P.theta_0, P.dtheta_ast, P.b, P.dtheta_c);
    c_number dfb  = DF4(t, P.a, P.theta_0, P.dtheta_ast, P.b, P.dtheta_c);  // df4/dt
    c_number gg, dgdt;
    if (mode == 0) {
        c_number tr = 2*PI - t;
        gg   = f4b + F4(tr, P.a, P.theta_0, P.dtheta_ast, P.b, P.dtheta_c);
        dgdt = dfb - DF4(tr, P.a, P.theta_0, P.dtheta_ast, P.b, P.dtheta_c);
    } else {
        c_number h  = (t > SB) ? SA*(t-SB)*(t-SB) : c_number(0);
        c_number dh = (t > SB) ? 2*SA*(t-SB)      : c_number(0);
        gg = f4b + h;  dgdt = dfb + dh;
    }
    g = gg;
    c_number sint = Kokkos::sqrt(Kokkos::fmax(c_number(0), 1 - cost*cost));
    Dc = (sint > c_number(1e-8)) ? (-dgdt / sint) : c_number(0);
}

// -----------------------------------------------------------------------
// Force/torque accumulation for one site-site F3 (excluded volume) term.
// delf/delta accumulate force/torque on particle a; delf_b/delta_b on b.
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void add_excv_contrib(const c_number ra_site[3], const c_number rb_site[3],
                      const c_number delr_site[3], const ExcvParams &ep,
                      c_number rsq,
                      c_number (&delf)[3], c_number (&delta)[3],
                      c_number (&delf_b)[3], c_number (&delta_b)[3],
                      c_number &evdwl) {
    if (rsq >= ep.cutsq_c) return;

    c_number fpair = 0;
    c_number U = F3(rsq, ep.cutsq_ast, ep.cut_c, ep.lj1, ep.lj2, ep.eps, ep.b, fpair);
    evdwl += U;

    // df is the standalone "force" vector (points a->b for repulsion, fpair>0).
    // Force on a is -df, on b is +df (matches _repulsive_lj: p->force -= force).
    c_number df[3] = {delr_site[0]*fpair, delr_site[1]*fpair, delr_site[2]*fpair};
    delf[0] -= df[0]; delf[1] -= df[1]; delf[2] -= df[2];
    delf_b[0] += df[0]; delf_b[1] += df[1]; delf_b[2] += df[2];

    c_number d[3], db[3];
    cross3(ra_site, df, d);
    delta[0] -= d[0]; delta[1] -= d[1]; delta[2] -= d[2];
    cross3(rb_site, df, db);
    delta_b[0] += db[0]; delta_b[1] += db[1]; delta_b[2] += db[2];
}

// -----------------------------------------------------------------------
// Hydrogen bonding (a = particle ia, b = particle ib).
// delr_bs is the a-base -> b-base separation; norm is its unit vector.
// Accumulates force on a into delf_a (force -= force convention is folded in:
// here we add the standalone "force on a", i.e. -force).
// -----------------------------------------------------------------------
template <class P>
KOKKOS_INLINE_FUNCTION
c_number hbond_pair(const c_number ra_bs[3], const c_number rb_bs[3],
                    const c_number delr_bs[3], c_number r_bs, c_number rinv,
                    const P &par, c_number alpha,
                    const c_number a1[3], const c_number a3[3],
                    const c_number b1[3], const c_number b3[3],
                    c_number (&delf_a)[3], c_number (&delta_a)[3],
                    c_number (&delf_b)[3], c_number (&delta_b)[3]) {
    const F1Params &fp = par.hb_f1;
    if (r_bs <= fp.cut_lc || r_bs >= fp.cut_hc) return 0;

    c_number f1 = F1(r_bs, fp.eps, fp.a, fp.cut_0, fp.cut_lc, fp.cut_hc,
                     fp.cut_lo, fp.cut_hi, fp.b_lo, fp.b_hi, fp.shift);

    c_number norm[3] = {delr_bs[0]*rinv, delr_bs[1]*rinv, delr_bs[2]*rinv};

    c_number cost1 = -dot3(a1, b1);
    c_number cost2 = -dot3(b1, norm);
    c_number cost3 =  dot3(a1, norm);
    c_number cost4 =  dot3(a3, b3);
    c_number cost7 = -dot3(b3, norm);
    c_number cost8 =  dot3(a3, norm);

    c_number f4t1, D1, f4t2, D2, f4t3, D3, f4t4, D4, f4t7, D7, f4t8, D8;
    eval_f4(cost1, par.hb_t1, f4t1, D1);
    eval_f4(cost2, par.hb_t2, f4t2, D2);
    eval_f4(cost3, par.hb_t3, f4t3, D3);
    eval_f4(cost4, par.hb_t4, f4t4, D4);
    eval_f4(cost7, par.hb_t7, f4t7, D7);
    eval_f4(cost8, par.hb_t8, f4t8, D8);

    c_number prod = f4t1*f4t2*f4t3*f4t4*f4t7*f4t8;
    c_number energy = f1 * prod;
    if (energy == 0) return 0;

    c_number df1 = DF1(r_bs, fp.eps, fp.a, fp.cut_0, fp.cut_lc, fp.cut_hc,
                       fp.cut_lo, fp.cut_hi, fp.b_lo, fp.b_hi);  // = df1/dr / r

    // standalone f4tXDsin (= d f4/d cost, signed)
    c_number f4t1Ds =  D1, f4t2Ds =  D2, f4t3Ds = -D3,
             f4t4Ds = -D4, f4t7Ds =  D7, f4t8Ds = -D8;

    c_number force[3] = {0,0,0}, tp[3] = {0,0,0}, tq[3] = {0,0,0};

    // RADIAL  (force = -rhat * df1/dr * prod = -delr * (df1/dr/r) * prod)
    c_number fr = df1 * prod;
    force[0] -= delr_bs[0]*fr; force[1] -= delr_bs[1]*fr; force[2] -= delr_bs[2]*fr;

    // THETA4 (pure torque)
    { c_number dir[3]; cross3(a3,b3,dir);
      c_number tm = -f1*f4t1*f4t2*f4t3*f4t4Ds*f4t7*f4t8;
      tp[0]-=dir[0]*tm; tp[1]-=dir[1]*tm; tp[2]-=dir[2]*tm;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA1 (pure torque)
    { c_number dir[3]; cross3(a1,b1,dir);
      c_number tm = -f1*f4t1Ds*f4t2*f4t3*f4t4*f4t7*f4t8;
      tp[0]-=dir[0]*tm; tp[1]-=dir[1]*tm; tp[2]-=dir[2]*tm;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA2
    { c_number fact = f1*f4t1*f4t2Ds*f4t3*f4t4*f4t7*f4t8;
      c_number fr2 = fact*rinv;
      force[0]+=(b1[0]+norm[0]*cost2)*fr2; force[1]+=(b1[1]+norm[1]*cost2)*fr2; force[2]+=(b1[2]+norm[2]*cost2)*fr2;
      c_number dir[3]; cross3(norm,b1,dir);
      tq[0]-=dir[0]*fact; tq[1]-=dir[1]*fact; tq[2]-=dir[2]*fact; }

    // THETA3
    { c_number fact = f1*f4t1*f4t2*f4t3Ds*f4t4*f4t7*f4t8;
      c_number fr3 = fact*rinv;
      force[0]+=(a1[0]-norm[0]*cost3)*fr3; force[1]+=(a1[1]-norm[1]*cost3)*fr3; force[2]+=(a1[2]-norm[2]*cost3)*fr3;
      c_number dir[3]; cross3(norm,a1,dir);
      c_number tm = -fact;
      tp[0]+=dir[0]*tm; tp[1]+=dir[1]*tm; tp[2]+=dir[2]*tm; }

    // THETA7
    { c_number fact = f1*f4t1*f4t2*f4t3*f4t4*f4t7Ds*f4t8;
      c_number fr7 = fact*rinv;
      force[0]+=(b3[0]+norm[0]*cost7)*fr7; force[1]+=(b3[1]+norm[1]*cost7)*fr7; force[2]+=(b3[2]+norm[2]*cost7)*fr7;
      c_number dir[3]; cross3(norm,b3,dir);
      c_number tm = -fact;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA8
    { c_number fact = f1*f4t1*f4t2*f4t3*f4t4*f4t7*f4t8Ds;
      c_number fr8 = fact*rinv;
      force[0]+=(a3[0]-norm[0]*cost8)*fr8; force[1]+=(a3[1]-norm[1]*cost8)*fr8; force[2]+=(a3[2]-norm[2]*cost8)*fr8;
      c_number dir[3]; cross3(norm,a3,dir);
      c_number tm = -fact;
      tp[0]+=dir[0]*tm; tp[1]+=dir[1]*tm; tp[2]+=dir[2]*tm; }

    // site torque contributions: tp -= ra x force ; tq += rb x force
    { c_number c[3]; cross3(ra_bs,force,c); tp[0]-=c[0]; tp[1]-=c[1]; tp[2]-=c[2]; }
    { c_number c[3]; cross3(rb_bs,force,c); tq[0]+=c[0]; tq[1]+=c[1]; tq[2]+=c[2]; }

    // force on a = -force, force on b = +force; scale everything by alpha
    delf_a[0] -= alpha*force[0]; delf_a[1] -= alpha*force[1]; delf_a[2] -= alpha*force[2];
    delf_b[0] += alpha*force[0]; delf_b[1] += alpha*force[1]; delf_b[2] += alpha*force[2];
    delta_a[0]+= alpha*tp[0]; delta_a[1]+= alpha*tp[1]; delta_a[2]+= alpha*tp[2];
    delta_b[0]+= alpha*tq[0]; delta_b[1]+= alpha*tq[1]; delta_b[2]+= alpha*tq[2];

    return alpha * energy;
}

// -----------------------------------------------------------------------
// Cross stacking (a = ia, b = ib). Uses the base-site separation, same six
// angles as H-bonding but with t4/t7/t8 symmetrised and an F2 radial term.
// -----------------------------------------------------------------------
template <class P>
KOKKOS_INLINE_FUNCTION
c_number crst_pair(const c_number ra_bs[3], const c_number rb_bs[3],
                   const c_number delr_bs[3], c_number r_bs, c_number rinv,
                   const P &par,
                   const c_number a1[3], const c_number a3[3],
                   const c_number b1[3], const c_number b3[3],
                   c_number (&delf_a)[3], c_number (&delta_a)[3],
                   c_number (&delf_b)[3], c_number (&delta_b)[3]) {
    const F2Params &fp = par.xstk_f2;
    if (r_bs <= fp.cut_lc || r_bs >= fp.cut_hc) return 0;

    c_number norm[3] = {delr_bs[0]*rinv, delr_bs[1]*rinv, delr_bs[2]*rinv};

    c_number cost1 = -dot3(a1, b1);
    c_number cost2 = -dot3(b1, norm);
    c_number cost3 =  dot3(a1, norm);
    c_number cost4 =  dot3(a3, b3);
    c_number cost7 = -dot3(b3, norm);
    c_number cost8 =  dot3(a3, norm);

    // t1,t2,t3 simple; t4,t7,t8 symmetrised: f4(c)+f4(-c)
    c_number f4t1, D1, f4t2, D2, f4t3, D3;
    eval_f4(cost1, par.xstk_t1, f4t1, D1);
    eval_f4(cost2, par.xstk_t2, f4t2, D2);
    eval_f4(cost3, par.xstk_t3, f4t3, D3);
    c_number f4p, Dp, f4m, Dm;
    eval_f4( cost4, par.xstk_t4, f4p, Dp); eval_f4(-cost4, par.xstk_t4, f4m, Dm);
    c_number f4t4 = f4p + f4m;  c_number f4t4Ds = -Dp + Dm;
    eval_f4( cost7, par.xstk_t7, f4p, Dp); eval_f4(-cost7, par.xstk_t7, f4m, Dm);
    c_number f4t7 = f4p + f4m;  c_number f4t7Ds =  Dp - Dm;
    eval_f4( cost8, par.xstk_t8, f4p, Dp); eval_f4(-cost8, par.xstk_t8, f4m, Dm);
    c_number f4t8 = f4p + f4m;  c_number f4t8Ds = -Dp + Dm;

    c_number prod = f4t1*f4t2*f4t3*f4t4*f4t7*f4t8;
    c_number f2 = F2(r_bs, fp.k, fp.cut_0, fp.cut_lc, fp.cut_hc,
                     fp.cut_lo, fp.cut_hi, fp.b_lo, fp.b_hi, fp.cut_c);
    c_number energy = f2 * prod;
    if (energy == 0) return 0;

    c_number f2D = DF2(r_bs, fp.k, fp.cut_0, fp.cut_lc, fp.cut_hc,
                       fp.cut_lo, fp.cut_hi, fp.b_lo, fp.b_hi);  // df2/dr

    c_number f4t1Ds = D1, f4t2Ds = D2, f4t3Ds = -D3;

    c_number force[3] = {0,0,0}, tp[3] = {0,0,0}, tq[3] = {0,0,0};

    // RADIAL  force = -rhat * df2/dr * prod
    c_number fr = f2D * prod;
    force[0]-=norm[0]*fr; force[1]-=norm[1]*fr; force[2]-=norm[2]*fr;

    // THETA1 (pure torque)
    { c_number dir[3]; cross3(a1,b1,dir);
      c_number tm = -f2*f4t1Ds*f4t2*f4t3*f4t4*f4t7*f4t8;
      tp[0]-=dir[0]*tm; tp[1]-=dir[1]*tm; tp[2]-=dir[2]*tm;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA2
    { c_number fact = f2*f4t1*f4t2Ds*f4t3*f4t4*f4t7*f4t8;
      c_number f = fact*rinv;
      force[0]+=(b1[0]+norm[0]*cost2)*f; force[1]+=(b1[1]+norm[1]*cost2)*f; force[2]+=(b1[2]+norm[2]*cost2)*f;
      c_number dir[3]; cross3(norm,b1,dir);
      tq[0]-=dir[0]*fact; tq[1]-=dir[1]*fact; tq[2]-=dir[2]*fact; }

    // THETA3
    { c_number fact = f2*f4t1*f4t2*f4t3Ds*f4t4*f4t7*f4t8;
      c_number f = fact*rinv;
      force[0]+=(a1[0]-norm[0]*cost3)*f; force[1]+=(a1[1]-norm[1]*cost3)*f; force[2]+=(a1[2]-norm[2]*cost3)*f;
      c_number dir[3]; cross3(norm,a1,dir);
      tp[0]-=dir[0]*fact; tp[1]-=dir[1]*fact; tp[2]-=dir[2]*fact; }

    // THETA4 (pure torque)
    { c_number dir[3]; cross3(a3,b3,dir);
      c_number tm = -f2*f4t1*f4t2*f4t3*f4t4Ds*f4t7*f4t8;
      tp[0]-=dir[0]*tm; tp[1]-=dir[1]*tm; tp[2]-=dir[2]*tm;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA7
    { c_number fact = f2*f4t1*f4t2*f4t3*f4t4*f4t7Ds*f4t8;
      c_number f = fact*rinv;
      force[0]+=(b3[0]+norm[0]*cost7)*f; force[1]+=(b3[1]+norm[1]*cost7)*f; force[2]+=(b3[2]+norm[2]*cost7)*f;
      c_number dir[3]; cross3(norm,b3,dir);
      tq[0]-=dir[0]*fact; tq[1]-=dir[1]*fact; tq[2]-=dir[2]*fact; }

    // THETA8
    { c_number fact = f2*f4t1*f4t2*f4t3*f4t4*f4t7*f4t8Ds;
      c_number f = fact*rinv;
      force[0]+=(a3[0]-norm[0]*cost8)*f; force[1]+=(a3[1]-norm[1]*cost8)*f; force[2]+=(a3[2]-norm[2]*cost8)*f;
      c_number dir[3]; cross3(norm,a3,dir);
      tp[0]-=dir[0]*fact; tp[1]-=dir[1]*fact; tp[2]-=dir[2]*fact; }

    { c_number c[3]; cross3(ra_bs,force,c); tp[0]-=c[0]; tp[1]-=c[1]; tp[2]-=c[2]; }
    { c_number c[3]; cross3(rb_bs,force,c); tq[0]+=c[0]; tq[1]+=c[1]; tq[2]+=c[2]; }

    delf_a[0]-=force[0]; delf_a[1]-=force[1]; delf_a[2]-=force[2];
    delf_b[0]+=force[0]; delf_b[1]+=force[1]; delf_b[2]+=force[2];
    delta_a[0]+=tp[0]; delta_a[1]+=tp[1]; delta_a[2]+=tp[2];
    delta_b[0]+=tq[0]; delta_b[1]+=tq[1]; delta_b[2]+=tq[2];

    return energy;
}

// -----------------------------------------------------------------------
// Coaxial stacking (a = ia, b = ib). Uses the stacking-site separation plus a
// backbone reference vector for the cosphi3 dihedral (faithful standalone port).
// -----------------------------------------------------------------------
template <class P>
KOKKOS_INLINE_FUNCTION
c_number cxst_pair(const c_number ra_st[3], const c_number rb_st[3],
                   const c_number delr_st[3], c_number r_st, c_number rinv,
                   const c_number delr_com[3], const P &par,
                   const c_number a1[3], const c_number a2[3], const c_number a3[3],
                   const c_number b1[3], const c_number b3[3],
                   c_number (&delf_a)[3], c_number (&delta_a)[3],
                   c_number (&delf_b)[3], c_number (&delta_b)[3]) {
    const F2Params &fp = par.cxst_f2;
    if (r_st <= fp.cut_lc || r_st >= fp.cut_hc) return 0;

    c_number rdir[3] = {delr_st[0]*rinv, delr_st[1]*rinv, delr_st[2]*rinv};

    c_number cost1 = -dot3(a1, b1);
    c_number cost4 =  dot3(a3, b3);
    c_number cost5 =  dot3(a3, rdir);
    c_number cost6 = -dot3(b3, rdir);

    // backbone reference vector (symmetric grooves): rbackref = rcom + b1*POS_BACK - a1*POS_BACK.
    // Only the oxDNA1 coaxial term uses the cosphi3 dihedral; oxDNA2 omits it.
    const c_number pb = par.d_cbk;       // POS_BACK (-0.4)
    c_number rbrefmod = 0, rbrefinv = 0, cosphi3 = 0;
    c_number rbrefdir[3] = {0,0,0};
    if (par.cxst_has_cosphi) {
        c_number rbref[3] = { delr_com[0] + pb*b1[0] - pb*a1[0],
                              delr_com[1] + pb*b1[1] - pb*a1[1],
                              delr_com[2] + pb*b1[2] - pb*a1[2] };
        rbrefmod = Kokkos::sqrt(rbref[0]*rbref[0]+rbref[1]*rbref[1]+rbref[2]*rbref[2]);
        rbrefinv = 1 / rbrefmod;
        rbrefdir[0]=rbref[0]*rbrefinv; rbrefdir[1]=rbref[1]*rbrefinv; rbrefdir[2]=rbref[2]*rbrefinv;
        c_number cr[3]; cross3(rbrefdir, a1, cr);
        cosphi3 = dot3(rdir, cr);
    }

    c_number f4t1, D1, f4t4, D4;
    eval_cxst_t1(cost1, par.cxst_t1, par.cxst_t1_mode, par.cxst_t1_SA, par.cxst_t1_SB, f4t1, D1);
    eval_f4(cost4, par.cxst_t4, f4t4, D4);
    c_number f4p, Dp, f4m, Dm;
    c_number f4t4Ds = -D4;
    if (par.cxst_t4_blunt) {
        // LAMMPS oxdna2/coaxstk (NEW, "3'3'/5'5' blunt end stacking"): a second,
        // mirrored theta4 lobe F4(theta4; pi - theta4_0) == f4(-cost4).
        eval_f4(-cost4, par.cxst_t4, f4m, Dm);
        f4t4 += f4m;  f4t4Ds += Dm;
    }
    eval_f4( cost5, par.cxst_t5, f4p, Dp); eval_f4(-cost5, par.cxst_t5, f4m, Dm);
    c_number f4t5 = f4p + f4m;  c_number f4t5Ds = -Dp + Dm;
    eval_f4( cost6, par.cxst_t6, f4p, Dp); eval_f4(-cost6, par.cxst_t6, f4m, Dm);
    c_number f4t6 = f4p + f4m;  c_number f4t6Ds =  Dp - Dm;

    // oxDNA1 multiplies by f5(cosphi3)^2; oxDNA2 drops it (f5 == 1).
    c_number f5  = par.cxst_has_cosphi
        ? F5(cosphi3, par.cxst_cp.a, par.cxst_cp.x_ast, par.cxst_cp.b, par.cxst_cp.x_c)
        : c_number(1);
    c_number f2 = F2(r_st, fp.k, fp.cut_0, fp.cut_lc, fp.cut_hc,
                     fp.cut_lo, fp.cut_hi, fp.b_lo, fp.b_hi, fp.cut_c);
    c_number energy = f2 * f4t1 * f4t4 * f4t5 * f4t6 * (f5*f5);
    if (energy == 0) return 0;

    c_number f2D = DF2(r_st, fp.k, fp.cut_0, fp.cut_lc, fp.cut_hc,
                       fp.cut_lo, fp.cut_hi, fp.b_lo, fp.b_hi);
    c_number f5D = par.cxst_has_cosphi
        ? DF5(cosphi3, par.cxst_cp.a, par.cxst_cp.x_ast, par.cxst_cp.b, par.cxst_cp.x_c)
        : c_number(0);
    c_number f4t1Ds = D1;

    c_number force[3] = {0,0,0}, tp[3] = {0,0,0}, tq[3] = {0,0,0};

    // RADIAL
    c_number fr = f2D * f4t1 * f4t4 * f4t5 * f4t6 * (f5*f5);
    force[0]-=rdir[0]*fr; force[1]-=rdir[1]*fr; force[2]-=rdir[2]*fr;

    // THETA1 (pure torque)
    { c_number dir[3]; cross3(a1,b1,dir);
      c_number tm = -f2*f4t1Ds*f4t4*f4t5*f4t6*(f5*f5);
      tp[0]-=dir[0]*tm; tp[1]-=dir[1]*tm; tp[2]-=dir[2]*tm;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA4 (pure torque)
    { c_number dir[3]; cross3(a3,b3,dir);
      c_number tm = -f2*f4t1*f4t4Ds*f4t5*f4t6*(f5*f5);
      tp[0]-=dir[0]*tm; tp[1]-=dir[1]*tm; tp[2]-=dir[2]*tm;
      tq[0]+=dir[0]*tm; tq[1]+=dir[1]*tm; tq[2]+=dir[2]*tm; }

    // THETA5
    { c_number fact = f2*f4t1*f4t4*f4t5Ds*f4t6*(f5*f5);
      c_number f = fact*rinv;
      force[0]+=(a3[0]-rdir[0]*cost5)*f; force[1]+=(a3[1]-rdir[1]*cost5)*f; force[2]+=(a3[2]-rdir[2]*cost5)*f;
      c_number dir[3]; cross3(rdir,a3,dir);
      tp[0]-=dir[0]*fact; tp[1]-=dir[1]*fact; tp[2]-=dir[2]*fact; }

    // THETA6
    { c_number fact = f2*f4t1*f4t4*f4t5*f4t6Ds*(f5*f5);
      c_number f = fact*rinv;
      force[0]+=(b3[0]+rdir[0]*cost6)*f; force[1]+=(b3[1]+rdir[1]*cost6)*f; force[2]+=(b3[2]+rdir[2]*cost6)*f;
      c_number dir[3]; cross3(rdir,b3,dir);
      tq[0]-=dir[0]*fact; tq[1]-=dir[1]*fact; tq[2]-=dir[2]*fact; }

    // COSPHI3 (gamma = POS_STACK - POS_BACK)
    if (par.cxst_has_cosphi) {
        c_number gamma = par.d_cstk - par.d_cbk;          // 0.34 - (-0.4) = 0.74
        c_number gammacub = gamma*gamma*gamma;
        c_number rbrefcub = rbrefmod*rbrefmod*rbrefmod;
        c_number a2b1 = dot3(a2,b1);
        c_number a3b1 = dot3(a3,b1);
        c_number ra1 = dot3(rdir,a1);
        c_number ra2 = dot3(rdir,a2);
        c_number ra3 = dot3(rdir,a3);
        c_number rb1 = dot3(rdir,b1);
        c_number paren = ra3*a2b1 - ra2*a3b1;

        c_number dcdr    = -gamma*paren*(gamma*(ra1-rb1)+r_st)/rbrefcub;
        c_number dcda1b1 =  gammacub*paren/rbrefcub;
        c_number dcda2b1 =  gamma*ra3*rbrefinv;
        c_number dcda3b1 = -gamma*ra2*rbrefinv;
        c_number dcdra1  = -gamma*gamma*paren*r_st/rbrefcub;
        c_number dcdra2  = -gamma*a3b1*rbrefinv;
        c_number dcdra3  =  gamma*a2b1*rbrefinv;
        c_number dcdrb1  =  gamma*gamma*paren*r_st/rbrefcub;

        c_number fc = f2*f4t1*f4t4*f4t5*f4t6*2*f5*f5D;

        // force += -fc*( rdir*dcdr + ((a1 - rdir*ra1)*dcdra1 + (a2 - rdir*ra2)*dcdra2
        //                            + (a3 - rdir*ra3)*dcdra3 + (b1 - rdir*rb1)*dcdrb1)/r_st )
        for (int k=0;k<3;k++) {
            c_number perp = (a1[k]-rdir[k]*ra1)*dcdra1 + (a2[k]-rdir[k]*ra2)*dcdra2
                          + (a3[k]-rdir[k]*ra3)*dcdra3 + (b1[k]-rdir[k]*rb1)*dcdrb1;
            force[k] += -fc*( rdir[k]*dcdr + perp*rinv );
        }

        c_number ca1[3], ca2[3], ca3[3], cb1[3];
        cross3(rdir,a1,ca1); cross3(rdir,a2,ca2); cross3(rdir,a3,ca3); cross3(rdir,b1,cb1);
        for (int k=0;k<3;k++) {
            tp[k] += fc*( ca1[k]*dcdra1 + ca2[k]*dcdra2 + ca3[k]*dcdra3 );
            tq[k] += fc*( cb1[k]*dcdrb1 );
        }
        c_number a1b1[3], a2b1v[3], a3b1v[3];
        cross3(a1,b1,a1b1); cross3(a2,b1,a2b1v); cross3(a3,b1,a3b1v);
        for (int k=0;k<3;k++) {
            c_number pt = fc*( a1b1[k]*dcda1b1 + a2b1v[k]*dcda2b1 + a3b1v[k]*dcda3b1 );
            tp[k] -= pt; tq[k] += pt;
        }
    }

    { c_number c[3]; cross3(ra_st,force,c); tp[0]-=c[0]; tp[1]-=c[1]; tp[2]-=c[2]; }
    { c_number c[3]; cross3(rb_st,force,c); tq[0]+=c[0]; tq[1]+=c[1]; tq[2]+=c[2]; }

    delf_a[0]-=force[0]; delf_a[1]-=force[1]; delf_a[2]-=force[2];
    delf_b[0]+=force[0]; delf_b[1]+=force[1]; delf_b[2]+=force[2];
    delta_a[0]+=tp[0]; delta_a[1]+=tp[1]; delta_a[2]+=tp[2];
    delta_b[0]+=tq[0]; delta_b[1]+=tq[1]; delta_b[2]+=tq[2];

    return energy;
}

// -----------------------------------------------------------------------
// Debye-Huckel electrostatics (oxDNA2). Acts on the backbone-site separation
// delr_bk (a->b, magnitude rmod). cut_factor halves the charge per terminus.
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
c_number dh_pair(const c_number ra_bk[3], const c_number rb_bk[3],
                 const c_number delr_bk[3], c_number rmod, c_number cut_factor,
                 const DNAParams &par,
                 c_number (&delf_a)[3], c_number (&delta_a)[3],
                 c_number (&delf_b)[3], c_number (&delta_b)[3]) {
    if (rmod >= par.dh_RC) return 0;
    c_number rinv = 1 / rmod;
    c_number rbackdir[3] = {delr_bk[0]*rinv, delr_bk[1]*rinv, delr_bk[2]*rinv};

    c_number energy, fmag;   // standalone "force" = rbackdir * fmag
    if (rmod < par.dh_RHIGH) {
        c_number ex = Kokkos::exp(rmod * par.dh_minus_kappa);
        energy = ex * (par.dh_prefactor * rinv);
        fmag   = -par.dh_prefactor * ex * (par.dh_minus_kappa * rinv - rinv*rinv);
    } else {
        c_number dr = rmod - par.dh_RC;
        energy = par.dh_B * dr * dr;
        fmag   = -2 * par.dh_B * dr;
    }
    energy *= cut_factor;
    fmag   *= cut_factor;

    c_number force[3] = {rbackdir[0]*fmag, rbackdir[1]*fmag, rbackdir[2]*fmag};
    delf_a[0]-=force[0]; delf_a[1]-=force[1]; delf_a[2]-=force[2];
    delf_b[0]+=force[0]; delf_b[1]+=force[1]; delf_b[2]+=force[2];
    c_number c[3];
    cross3(ra_bk, force, c); delta_a[0]-=c[0]; delta_a[1]-=c[1]; delta_a[2]-=c[2];
    cross3(rb_bk, force, c); delta_b[0]+=c[0]; delta_b[1]+=c[1]; delta_b[2]+=c[2];
    return energy;
}


// =======================================================================
// LAMMPS-FAITHFUL force kernels (tracks LAMMPS origin/oxdna3KK-kk-fixes,
// 392462c401).
//
// Mirrors the LAMMPS KOKKOS oxDNA kernel structure, one kernel per style:
//   * fix OXDNA/LRF: a per-atom pass stores the body frames nx/ny/nz; every
//     force kernel READS them. In lammps_overhead mode it runs over the
//     local + ghost atoms and reads the quaternion through atom->ellipsoid.
//   * pair oxdna*/excv: one thread per atom over the half neighbor list
//     (HALFTHREAD), bonded (special) pairs included: backbone-backbone is
//     knocked out by special_lj = 0, the other three site pairs give the
//     bonded excluded volume. The base-base term first runs the per-pair
//     topology (tetramer) test. Atom a accumulates in registers and is flushed
//     once; atom b gets atomics per active term. OxdnaRangePolicy launch bounds.
//   * pair oxdna/hbond, oxdna/xstk, oxdna2/coaxstk: one thread per screened
//     pair (fix OXDNA/NPAIR packed uint64 list), special pairs exit early,
//     plain RangePolicy, atomics to both atoms (ScatterAtomic access on every
//     backend, kk-fixes e7796be5e2).
//   * pair oxdna/coaxstk (oxDNA1): one thread per atom over the half list.
//   * pair oxdna2/dh: one thread per atom over the half list, per-atom qeff,
//     rsq cutoff test before rsqrt, register accumulation of atom a,
//     OxdnaRangePolicy launch bounds.
// lammps_overhead additionally switches on what LAMMPS reads per pair / atom
// beyond the physics: the list index through d_ilist, the tag-based topology
// test of excv (tag(a) == id3p(b) && tag(b) == id5p(a)), the prime-neighbor
// tables; lammps_tables the per-type coefficient tables (tables.h).
// Force / torque accumulation is in c_acc (KK_ACC_FLOAT). Every kernel makes
// its own ScatterView (non-duplicated = atomics), as each LAMMPS pair style does.
// =======================================================================

using ScatterF4 = Kokkos::Experimental::ScatterView<
    c_acc *[4],
    Kokkos::LayoutRight,
    Kokkos::DefaultExecutionSpace,
    Kokkos::Experimental::ScatterSum,
    Kokkos::Experimental::ScatterNonDuplicated>;
using OxScatterAtomic = Kokkos::Experimental::ScatterAtomic;

template <class T>
using TabView2 = Kokkos::View<const T **>;
template <class T>
using TabView1 = Kokkos::View<const T *>;

// -----------------------------------------------------------------------
// LRF precompute: one thread per atom. Compute a1,a2,a3 from the quaternion
// and STORE them in nx/ny/nz. Mirrors LAMMPS `fix OXDNA/LRF`
// (TagFixOxdnaLRFComputeQuatToXYZ): with lmp = true over nlocal + nghost and
// through the bonus index (bonus(ellipsoid(i)).quat).
// -----------------------------------------------------------------------
inline void compute_lrf(ParticleArrays &p, bool lmp = false) {
    auto ori = p.orientations;
    auto nx = p.nx, ny = p.ny, nz = p.nz;
    if (lmp) {
        auto ell = p.ellipsoid;
        Kokkos::parallel_for("oxdna_lrf", p.N + p.nghost, KOKKOS_LAMBDA(int i) {
            const int e = ell(i);
            if (e < 0) return;
            c_number a1[3], a2[3], a3[3];
            get_vectors_from_quat(ori(e, 0), ori(e, 1), ori(e, 2), ori(e, 3), a1, a2, a3);
            nx(i,0)=a1[0]; nx(i,1)=a1[1]; nx(i,2)=a1[2]; nx(i,3)=0;
            ny(i,0)=a2[0]; ny(i,1)=a2[1]; ny(i,2)=a2[2]; ny(i,3)=0;
            nz(i,0)=a3[0]; nz(i,1)=a3[1]; nz(i,2)=a3[2]; nz(i,3)=0;
        });
        return;
    }
    Kokkos::parallel_for("oxdna_lrf", p.N, KOKKOS_LAMBDA(int i) {
        c_number a1[3], a2[3], a3[3];
        get_vectors_from_quat_view(ori, i, a1, a2, a3);
        nx(i,0)=a1[0]; nx(i,1)=a1[1]; nx(i,2)=a1[2]; nx(i,3)=0;
        ny(i,0)=a2[0]; ny(i,1)=a2[1]; ny(i,2)=a2[2]; ny(i,3)=0;
        nz(i,0)=a3[0]; nz(i,1)=a3[1]; nz(i,2)=a3[2]; nz(i,3)=0;
    });
}

KOKKOS_INLINE_FUNCTION
void load_frame(const Vec4cr &nx, const Vec4cr &ny, const Vec4cr &nz, int i,
                c_number (&a1)[3], c_number (&a2)[3], c_number (&a3)[3]) {
    a1[0]=nx(i,0); a1[1]=nx(i,1); a1[2]=nx(i,2);
    a2[0]=ny(i,0); a2[1]=ny(i,1); a2[2]=ny(i,2);
    a3[0]=nz(i,0); a3[1]=nz(i,1); a3[2]=nz(i,2);
}

KOKKOS_INLINE_FUNCTION
c_number fma_dot3(const c_number d[3]) {
    return Kokkos::fma(d[2], d[2], Kokkos::fma(d[1], d[1], d[0]*d[0]));
}

// Scatter one pair's force/torque increments to both atoms (atomics on GPU).
template <class AF, class AT>
KOKKOS_INLINE_FUNCTION
void scatter_pair(const AF &af, const AT &at, int ia, int ib,
                  const c_number (&delf_a)[3], const c_number (&delta_a)[3],
                  const c_number (&delf_b)[3], const c_number (&delta_b)[3]) {
    af(ia,0)+=delf_a[0];  af(ia,1)+=delf_a[1];  af(ia,2)+=delf_a[2];
    af(ib,0)+=delf_b[0];  af(ib,1)+=delf_b[1];  af(ib,2)+=delf_b[2];
    at(ia,0)+=delta_a[0]; at(ia,1)+=delta_a[1]; at(ia,2)+=delta_a[2];
    at(ib,0)+=delta_b[0]; at(ib,1)+=delta_b[1]; at(ib,2)+=delta_b[2];
}

// Unpack a LAMMPS fix OXDNA/NPAIR screened pair: a in the upper 32 bits, the
// raw neighbor index (special bits preserved) in the lower 32 bits.
KOKKOS_INLINE_FUNCTION
void unpack_pair(uint64_t v, int &ia, int &braw) {
    ia   = static_cast<int>(v >> 32);
    braw = static_cast<int>(static_cast<uint32_t>(v & 0xffffffffu));
}

// Index into the (ntypes+1)^4 = 5^4 tetramer tables of LAMMPS (atom types 1..4,
// index 0 = no neighbour). Bench base types are 0..3 and -1 = none.
KOKKOS_INLINE_FUNCTION
int tet_index(int t3p, int ta, int tb, int t5p) {
    return lmp_tet_index(t3p, ta, tb, t5p);
}

// atom->map() of a 3'/5' partner tag (-1 = none), as the prime-neighbor
// precomputes resolve them (map_style array)
KOKKOS_INLINE_FUNCTION
int map_tag(const Kokkos::View<const int *, Kokkos::MemoryTraits<Kokkos::RandomAccess>> &map, int t) {
    return (t >= 0 && t < static_cast<int>(map.extent(0))) ? map(t) : -1;
}

// LAMMPS 4D base-base excluded-volume table (d_cut4sq_bsbs_c & co). For the
// oxDNA1/2 models every entry equals the 2D value (vanilla pair_oxdna_excv
// fills them uniformly), so this only reproduces the memory traffic. Refilled
// only when the base-base parameters change. (lammps_tables builds the full
// set, OxdnaTables::excv_bsbs4; this one is used with lammps_tables = 0.)
inline void ensure_tet_excv(ParticleArrays &p, const DNAParams &par) {
    const ExcvParams &e = par.excv_bsbs;
    const ExcvParams &c = p.tet_excv_key;
    if (p.tet_excv_bsbs.extent(0) == 625 && c.eps == e.eps && c.lj1 == e.lj1 &&
        c.lj2 == e.lj2 && c.b == e.b && c.cutsq_ast == e.cutsq_ast &&
        c.cutsq_c == e.cutsq_c && c.cut_c == e.cut_c) return;
    if (p.tet_excv_bsbs.extent(0) != 625)
        p.tet_excv_bsbs = Kokkos::View<ExcvParams *>("tet_excv_bsbs", 625);
    auto h = Kokkos::create_mirror_view(p.tet_excv_bsbs);
    for (int i = 0; i < 625; i++) h(i) = e;
    Kokkos::deep_copy(p.tet_excv_bsbs, h);
    p.tet_excv_key = e;
}

// -----------------------------------------------------------------------
// fix OXDNA/PRIME_NEIGHS::compute_prime_neighs_pair (lammps_overhead mode,
// neighbor-rebuild steps, called from pair oxdna*/excv): one thread per list
// atom over all its neighbors, writing (map(id3p(a)), map(id5p(b)),
// map(id3p(b)), map(id5p(a))) per neighbor slot. The table has as many rows
// as the neighbor list (d_neighbors.extent(0), kk-fixes 360d7a5a2f); grow-only.
// -----------------------------------------------------------------------
inline void build_prime_pair(const ParticleArrays &p, const NeighborList &nl) {
    const size_t nrows = nl.d_neigh_matrix.extent(0);
    const size_t ncols = nl.d_neigh_matrix.extent(1);
    if (nl.prime_pair.extent(0) < nrows || nl.prime_pair.extent(1) < ncols)
        nl.prime_pair = Kokkos::View<int ***>("prime_neighs_pair", nrows, ncols, 4);
    auto tab = nl.prime_pair; auto nnum = nl.d_num_neigh; auto nmat = nl.d_neigh_matrix;
    auto bonds = p.bonds; auto ilist = nl.d_ilist;
    const bool use_ilist = (nl.d_ilist.extent_int(0) >= p.N && p.N > 0);
    Kokkos::View<const int *, Kokkos::MemoryTraits<Kokkos::RandomAccess>> map = p.map_array;
    Kokkos::parallel_for("oxdna_prime_neighs_pair", p.N, KOKKOS_LAMBDA(int ii) {
        const int a = use_ilist ? ilist(ii) : ii;
        const int m = nnum(a);
        for (int k = 0; k < m; k++) {
            const int b = nmat(a, k) & OX_NEIGHMASK;
            tab(a,k,0) = map_tag(map, bonds(a).n3);
            tab(a,k,1) = map_tag(map, bonds(b).n5);
            tab(a,k,2) = map_tag(map, bonds(b).n3);
            tab(a,k,3) = map_tag(map, bonds(a).n5);
        }
    });
}

// -----------------------------------------------------------------------
// EXCV (pair oxdna*/excv): per atom over the half list incl. bonded pairs.
// Term order as LAMMPS: bkbk (x special_lj), bk(a)-bs(b), bs(a)-bk(b), bsbs.
// GROOVED = oxDNA2 backbone site (-0.34 a1 + 0.3408 a2); oxDNA1 reads only nx.
// LMP: ilist, tag-based topology test, prime_neighs_pair flanks + 4D table
// for the bonded base-base pair. TAB: per-type 2D tables for the site pairs.
// -----------------------------------------------------------------------
template <bool GROOVED, bool LMP = false, bool TAB = false>
struct ExcvFunctor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    RandomRead<int> btype, tag;
    Kokkos::View<const int *>  ilist;
    Kokkos::View<const int *>  num_neigh;
    Kokkos::View<const int **> neigh_matrix;
    Kokkos::View<const int ***> prime_pair;
    TabView1<ExcvParams> bsbs4;
    TabView2<ExcvParams> t_bkbk, t_bkbs, t_bsbs;
    DNAParams par;
    ScatterF4 sf, st;
    SimBox box;

    // One excluded-volume site pair. d = (b site) - (a site). Force on a is
    // -d*fpair (kept in registers), on b +d*fpair (atomic, immediately).
    template <class AF, class AT>
    KOKKOS_INLINE_FUNCTION
    void term(const c_number ra[3], const c_number rb[3], const c_number dcom[3],
              const ExcvParams &ep, c_number factor,
              c_acc (&ftmp)[3], c_acc (&ttmp)[3],
              const AF &af, const AT &at, int ib, c_acc &ev) const {
        c_number d[3] = {dcom[0]+rb[0]-ra[0], dcom[1]+rb[1]-ra[1], dcom[2]+rb[2]-ra[2]};
        const c_number rsq = fma_dot3(d);
        if (rsq >= ep.cutsq_c) return;
        c_number fpair = 0;
        c_number U = F3(rsq, ep.cutsq_ast, ep.cut_c, ep.lj1, ep.lj2, ep.eps, ep.b, fpair);
        fpair *= factor;
        U     *= factor;
        ev += U;
        c_number df[3] = {d[0]*fpair, d[1]*fpair, d[2]*fpair};
        c_number ta[3], tb[3];
        cross3(ra, df, ta);
        cross3(rb, df, tb);
        ftmp[0] -= df[0]; ftmp[1] -= df[1]; ftmp[2] -= df[2];
        ttmp[0] -= ta[0]; ttmp[1] -= ta[1]; ttmp[2] -= ta[2];
        af(ib,0) += df[0]; af(ib,1) += df[1]; af(ib,2) += df[2];
        at(ib,0) += tb[0]; at(ib,1) += tb[1]; at(ib,2) += tb[2];
    }

    KOKKOS_INLINE_FUNCTION void operator()(int ii) const { c_acc ev=0; (*this)(ii, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int ii, c_acc &ev) const {
        const int ia = LMP ? ilist(ii) : ii;
        const int m = num_neigh(ia);
        const c_number xai = poss(ia,0), yai = poss(ia,1), zai = poss(ia,2);
        const int ta = btype(ia);
        const LR_bonds ba = bonds(ia);
        const c_number pb1 = par.pb1, pb2 = par.pb2, d_cbs = par.d_cbs;
        c_number ra_cbk[3], ra_cbs[3];
        {
            const c_number a1[3] = {nx(ia,0), nx(ia,1), nx(ia,2)};
            if (GROOVED) {
                const c_number a2[3] = {ny(ia,0), ny(ia,1), ny(ia,2)};
                for (int c = 0; c < 3; c++) ra_cbk[c] = Kokkos::fma(pb2, a2[c], pb1*a1[c]);
            } else {
                for (int c = 0; c < 3; c++) ra_cbk[c] = pb1*a1[c];
            }
            for (int c = 0; c < 3; c++) ra_cbs[c] = d_cbs*a1[c];
        }

        c_acc ftmp[3] = {0,0,0}, ttmp[3] = {0,0,0};
        auto af = sf.access();
        auto at = st.access();

        for (int k = 0; k < m; k++) {
            const int braw = neigh_matrix(ia, k);
            const c_number factor_lj = ox_sbmask(braw) ? c_number(0) : c_number(1);
            const int ib = braw & OX_NEIGHMASK;
            const int tb = btype(ib);

            c_number rb_cbk[3], rb_cbs[3];
            {
                const c_number b1[3] = {nx(ib,0), nx(ib,1), nx(ib,2)};
                if (GROOVED) {
                    const c_number b2[3] = {ny(ib,0), ny(ib,1), ny(ib,2)};
                    for (int c = 0; c < 3; c++) rb_cbk[c] = Kokkos::fma(pb2, b2[c], pb1*b1[c]);
                } else {
                    for (int c = 0; c < 3; c++) rb_cbk[c] = pb1*b1[c];
                }
                for (int c = 0; c < 3; c++) rb_cbs[c] = d_cbs*b1[c];
            }
            c_number dcom[3] = {poss(ib,0)-xai, poss(ib,1)-yai, poss(ib,2)-zai};
            box.wrap(dcom[0], dcom[1], dcom[2]);

            if (TAB) {
                term(ra_cbk, rb_cbk, dcom, t_bkbk(ta+1, tb+1), factor_lj, ftmp, ttmp, af, at, ib, ev); // bkbk
                term(ra_cbk, rb_cbs, dcom, t_bkbs(ta+1, tb+1), 1,         ftmp, ttmp, af, at, ib, ev); // bk(a)-bs(b)
                term(ra_cbs, rb_cbk, dcom, t_bkbs(ta+1, tb+1), 1,         ftmp, ttmp, af, at, ib, ev); // bs(a)-bk(b)
            } else {
                term(ra_cbk, rb_cbk, dcom, par.excv_bkbk, factor_lj, ftmp, ttmp, af, at, ib, ev); // bkbk
                term(ra_cbk, rb_cbs, dcom, par.excv_bkbs, 1,         ftmp, ttmp, af, at, ib, ev); // bk(a)-bs(b)
                term(ra_cbs, rb_cbk, dcom, par.excv_bkbs, 1,         ftmp, ttmp, af, at, ib, ev); // bs(a)-bk(b)
            }

            // base-base: tetramer topology test (a is b's 3' neighbour, or its
            // 5' neighbour) before falling back to the 2D table.
            const LR_bonds bb = bonds(ib);
            bool bond1, bond2;
            if (LMP) {   // tag(a) == id3p(b) && tag(b) == id5p(a), then the mirrored test
                const int tag_a = tag(ia);
                bond1 = (bb.n3 == tag_a && tag(ib) == ba.n5);
                bond2 = !bond1 && (bb.n5 == tag_a && tag(ib) == ba.n3);
            } else {
                bond1 = (bb.n3 == ia && ba.n5 == ib);
                bond2 = !bond1 && (bb.n5 == ia && ba.n3 == ib);
            }
            if (bond1) {
                if (LMP) {
                    const int p3 = prime_pair(ia,k,0), p5 = prime_pair(ia,k,1);
                    const int t3 = (p3 >= 0) ? btype(p3) : -1;
                    const int t5 = (p5 >= 0) ? btype(p5) : -1;
                    term(ra_cbs, rb_cbs, dcom, bsbs4(tet_index(t3, ta, tb, t5)), 1, ftmp, ttmp, af, at, ib, ev);
                } else {
                    term(ra_cbs, rb_cbs, dcom, par.excv_bsbs, 1, ftmp, ttmp, af, at, ib, ev);
                }
            } else if (bond2) {
                if (LMP) {
                    const int p3 = prime_pair(ia,k,2), p5 = prime_pair(ia,k,3);
                    const int t3 = (p3 >= 0) ? btype(p3) : -1;
                    const int t5 = (p5 >= 0) ? btype(p5) : -1;
                    term(ra_cbs, rb_cbs, dcom, bsbs4(tet_index(t3, tb, ta, t5)), 1, ftmp, ttmp, af, at, ib, ev);
                } else {
                    term(ra_cbs, rb_cbs, dcom, par.excv_bsbs, 1, ftmp, ttmp, af, at, ib, ev);
                }
            } else if (TAB) {
                term(ra_cbs, rb_cbs, dcom, t_bsbs(ta+1, tb+1), 1, ftmp, ttmp, af, at, ib, ev);
            } else {
                term(ra_cbs, rb_cbs, dcom, par.excv_bsbs, 1, ftmp, ttmp, af, at, ib, ev);
            }
        }

        af(ia,0) += ftmp[0]; af(ia,1) += ftmp[1]; af(ia,2) += ftmp[2];
        at(ia,0) += ttmp[0]; at(ia,1) += ttmp[1]; at(ia,2) += ttmp[2];
    }
};

// -----------------------------------------------------------------------
// DH (pair oxdna2/dh): per atom over the half list. Special pairs skipped
// first; rsq tested against the cutoff before any sqrt; rinv = rsqrt(rsq),
// r = rsq*rinv; per-atom qeff (qeff(a) hoisted, qeff(b) after the cutoff);
// atom a accumulated in registers, atom b by atomics. TAB: the coefficients
// (and the cutoff) come from the (type a, type b) table.
// -----------------------------------------------------------------------
template <bool LMP = false, bool TAB = false>
struct DHFunctor {
    Vec4cr poss, nx, ny;
    RandomRead<c_number> qeff;
    RandomRead<int> btype;
    Kokkos::View<const int *>  ilist;
    Kokkos::View<const int *>  num_neigh;
    Kokkos::View<const int **> neigh_matrix;
    TabView2<DhCoeffs> tdh;
    DNAParams par;
    c_number rcsq;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int ii) const { c_acc ev=0; (*this)(ii, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int ii, c_acc &ev) const {
        const int ia = LMP ? ilist(ii) : ii;
        const int m = num_neigh(ia);
        const c_number qeff_a = qeff(ia);
        const int ta = TAB ? btype(ia) : 0;
        const c_number xai = poss(ia,0), yai = poss(ia,1), zai = poss(ia,2);
        const c_number pb1 = par.pb1, pb2 = par.pb2;
        const c_number ra0 = Kokkos::fma(pb2, ny(ia,0), pb1*nx(ia,0));
        const c_number ra1 = Kokkos::fma(pb2, ny(ia,1), pb1*nx(ia,1));
        const c_number ra2 = Kokkos::fma(pb2, ny(ia,2), pb1*nx(ia,2));

        c_acc ftmp0 = 0, ftmp1 = 0, ftmp2 = 0, ttmp0 = 0, ttmp1 = 0, ttmp2 = 0;
        auto af = sf.access();
        auto at = st.access();

        for (int k = 0; k < m; k++) {
            const int braw = neigh_matrix(ia, k);
            if (ox_sbmask(braw)) continue;          // special_lj = 0
            const int ib = braw & OX_NEIGHMASK;
            c_number dx = poss(ib,0)-xai, dy = poss(ib,1)-yai, dz = poss(ib,2)-zai;
            box.wrap(dx, dy, dz);
            const c_number rb0 = Kokkos::fma(pb2, ny(ib,0), pb1*nx(ib,0));
            const c_number rb1 = Kokkos::fma(pb2, ny(ib,1), pb1*nx(ib,1));
            const c_number rb2 = Kokkos::fma(pb2, ny(ib,2), pb1*nx(ib,2));
            const c_number d0 = dx + rb0 - ra0, d1 = dy + rb1 - ra1, d2 = dz + rb2 - ra2;
            const c_number rsq = Kokkos::fma(d2, d2, Kokkos::fma(d1, d1, d0*d0));
            const int tb = TAB ? btype(ib) : 0;
            if (rsq > (TAB ? tdh(ta+1, tb+1).cutsq_c : rcsq)) continue;

            const DhCoeffs dc = TAB ? tdh(ta+1, tb+1)
                                    : DhCoeffs{par.dh_prefactor, par.dh_minus_kappa, par.dh_B,
                                               par.dh_RHIGH, par.dh_RC, rcsq};
            const c_number qq   = qeff_a * qeff(ib);
            const c_number rinv = Kokkos::rsqrt(rsq);
            const c_number r    = rsq * rinv;
            c_number U, fmag;                        // standalone "force" = d*rinv*fmag
            if (r <= dc.dh_RHIGH) {
                const c_number ex = Kokkos::exp(r * dc.dh_minus_kappa);
                U    = qq * dc.dh_prefactor * ex * rinv;
                fmag = -qq * dc.dh_prefactor * ex * (dc.dh_minus_kappa * rinv - rinv*rinv);
            } else {
                const c_number dr = r - dc.dh_RC;
                U    = qq * dc.dh_B * dr * dr;
                fmag = -2 * qq * dc.dh_B * dr;
            }
            ev += U;
            const c_number s  = fmag * rinv;
            const c_number f0 = d0*s, f1 = d1*s, f2 = d2*s;
            ftmp0 -= f0; ftmp1 -= f1; ftmp2 -= f2;
            ttmp0 -= Kokkos::fma(ra1, f2, -ra2*f1);
            ttmp1 -= Kokkos::fma(ra2, f0, -ra0*f2);
            ttmp2 -= Kokkos::fma(ra0, f1, -ra1*f0);
            af(ib,0) += f0; af(ib,1) += f1; af(ib,2) += f2;
            at(ib,0) += Kokkos::fma(rb1, f2, -rb2*f1);
            at(ib,1) += Kokkos::fma(rb2, f0, -rb0*f2);
            at(ib,2) += Kokkos::fma(rb0, f1, -rb1*f0);
        }

        af(ia,0) += ftmp0; af(ia,1) += ftmp1; af(ia,2) += ftmp2;
        at(ia,0) += ttmp0; at(ia,1) += ttmp1; at(ia,2) += ttmp2;
    }
};

// -----------------------------------------------------------------------
// Screened-pair functors (one thread per fix OXDNA/NPAIR pair). All updates
// through ScatterAtomic access (several threads update the same atoms).
// -----------------------------------------------------------------------
template <bool TAB = false>
struct HbondFunctor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<int> btype;
    Kokkos::View<const uint64_t *> sp;
    TabView2<HbondCoeffs> thb;
    DNAParams par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev=0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;
        int at_t = btype(ia), bt_t = btype(ib);
        c_number alpha = TAB ? thb(at_t+1, bt_t+1).eps : par.alpha_hb[at_t][bt_t];
        if (alpha == 0) return;

        c_number dx = poss(ib,0)-poss(ia,0), dy = poss(ib,1)-poss(ia,1), dz = poss(ib,2)-poss(ia,2);
        box.wrap(dx, dy, dz);
        c_number a1[3], a2[3], a3[3], b1[3], b2[3], b3[3];
        load_frame(nx, ny, nz, ia, a1, a2, a3);
        load_frame(nx, ny, nz, ib, b1, b2, b3);
        c_number d_cbs = par.d_cbs;
        c_number ra_cbs[3] = {d_cbs*a1[0], d_cbs*a1[1], d_cbs*a1[2]};
        c_number rb_cbs[3] = {d_cbs*b1[0], d_cbs*b1[1], d_cbs*b1[2]};
        c_number d[3] = {dx+rb_cbs[0]-ra_cbs[0], dy+rb_cbs[1]-ra_cbs[1], dz+rb_cbs[2]-ra_cbs[2]};
        c_number rsq = fma_dot3(d);
        if (rsq <= 0) return;
        c_number r = Kokkos::sqrt(rsq);
        c_number rinv = 1 / r;

        c_number delf_a[3]={0,0,0}, delf_b[3]={0,0,0};
        c_number delta_a[3]={0,0,0}, delta_b[3]={0,0,0};
        c_number e_p = TAB ? hbond_pair(ra_cbs, rb_cbs, d, r, rinv, thb(at_t+1, bt_t+1), alpha,
                                        a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b)
                           : hbond_pair(ra_cbs, rb_cbs, d, r, rinv, par, alpha,
                                        a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
        if (e_p == 0) return;
        ev += e_p;
        scatter_pair(sf.template access<OxScatterAtomic>(), st.template access<OxScatterAtomic>(),
                     ia, ib, delf_a, delta_a, delf_b, delta_b);
    }
};

template <bool TAB = false>
struct XstkFunctor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<int> btype;
    Kokkos::View<const uint64_t *> sp;
    TabView2<XstkCoeffs> txs;
    DNAParams par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev=0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;
        const int at_t = TAB ? btype(ia) : 0, bt_t = TAB ? btype(ib) : 0;
        c_number dx = poss(ib,0)-poss(ia,0), dy = poss(ib,1)-poss(ia,1), dz = poss(ib,2)-poss(ia,2);
        box.wrap(dx, dy, dz);
        c_number a1[3], a2[3], a3[3], b1[3], b2[3], b3[3];
        load_frame(nx, ny, nz, ia, a1, a2, a3);
        load_frame(nx, ny, nz, ib, b1, b2, b3);
        c_number d_cbs = par.d_cbs;
        c_number ra_cbs[3] = {d_cbs*a1[0], d_cbs*a1[1], d_cbs*a1[2]};
        c_number rb_cbs[3] = {d_cbs*b1[0], d_cbs*b1[1], d_cbs*b1[2]};
        c_number d[3] = {dx+rb_cbs[0]-ra_cbs[0], dy+rb_cbs[1]-ra_cbs[1], dz+rb_cbs[2]-ra_cbs[2]};
        c_number rsq = fma_dot3(d);
        if (rsq <= 0) return;                        // rsq_hb <= 0 guard
        c_number r = Kokkos::sqrt(rsq);
        c_number rinv = 1 / r;

        c_number delf_a[3]={0,0,0}, delf_b[3]={0,0,0};
        c_number delta_a[3]={0,0,0}, delta_b[3]={0,0,0};
        c_number e_p = TAB ? crst_pair(ra_cbs, rb_cbs, d, r, rinv, txs(at_t+1, bt_t+1),
                                       a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b)
                           : crst_pair(ra_cbs, rb_cbs, d, r, rinv, par,
                                       a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
        if (e_p == 0) return;
        ev += e_p;
        scatter_pair(sf.template access<OxScatterAtomic>(), st.template access<OxScatterAtomic>(),
                     ia, ib, delf_a, delta_a, delf_b, delta_b);
    }
};

// Coaxial stacking for one (ia, ib) pair (shared by the screened oxDNA2 kernel
// and the per-atom oxDNA1 kernel).
template <class P>
KOKKOS_INLINE_FUNCTION
c_number coaxstk_pair_geom(const Vec4cr &poss, const Vec4cr &nx, const Vec4cr &ny,
                           const Vec4cr &nz, const P &par, const SimBox &box,
                           int ia, int ib,
                           c_number (&delf_a)[3], c_number (&delta_a)[3],
                           c_number (&delf_b)[3], c_number (&delta_b)[3]) {
    c_number dx = poss(ib,0)-poss(ia,0), dy = poss(ib,1)-poss(ia,1), dz = poss(ib,2)-poss(ia,2);
    box.wrap(dx, dy, dz);
    c_number delr_com[3] = {dx, dy, dz};
    c_number a1[3], a2[3], a3[3], b1[3], b2[3], b3[3];
    load_frame(nx, ny, nz, ia, a1, a2, a3);
    load_frame(nx, ny, nz, ib, b1, b2, b3);
    c_number d_cstk = par.d_cstk;
    c_number ra_st[3] = {d_cstk*a1[0], d_cstk*a1[1], d_cstk*a1[2]};
    c_number rb_st[3] = {d_cstk*b1[0], d_cstk*b1[1], d_cstk*b1[2]};
    c_number d[3] = {dx+rb_st[0]-ra_st[0], dy+rb_st[1]-ra_st[1], dz+rb_st[2]-ra_st[2]};
    c_number r = Kokkos::sqrt(fma_dot3(d));
    if (r <= 0) return 0;
    c_number rinv = 1 / r;
    return cxst_pair(ra_st, rb_st, d, r, rinv, delr_com, par,
                     a1, a2, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
}

// coaxstk coefficients of one type pair: the tabulated values plus the model
// constants (flags and site offsets) of DNAParams
KOKKOS_INLINE_FUNCTION
CxstCoeffs cxst_coeffs(const CxstCoeffs &t, const DNAParams &par) {
    CxstCoeffs c = t;
    c.cxst_t1_mode = par.cxst_t1_mode; c.cxst_has_cosphi = par.cxst_has_cosphi;
    c.cxst_t4_blunt = par.cxst_t4_blunt; c.d_cbk = par.d_cbk; c.d_cstk = par.d_cstk;
    return c;
}

// pair oxdna2/coaxstk: one thread per screened pair. With the LAMMPS-only
// terminal criterion enabled (par.cxst_terminal_only), both nucleotides must be
// strand ends, tested right after the special check (4 extra int loads).
template <bool TAB = false>
struct Coaxstk2Functor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    RandomRead<int> btype;
    Kokkos::View<const uint64_t *> sp;
    TabView2<CxstCoeffs> tcx;
    DNAParams par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev=0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;
        if (par.cxst_terminal_only) {
            const LR_bonds ba = bonds(ia), bb = bonds(ib);
            if (ba.n3 >= 0 && ba.n5 >= 0) return;
            if (bb.n3 >= 0 && bb.n5 >= 0) return;
        }
        c_number delf_a[3]={0,0,0}, delf_b[3]={0,0,0};
        c_number delta_a[3]={0,0,0}, delta_b[3]={0,0,0};
        c_number e_p;
        if (TAB) {
            const CxstCoeffs cc = cxst_coeffs(tcx(btype(ia)+1, btype(ib)+1), par);
            e_p = coaxstk_pair_geom(poss, nx, ny, nz, cc, box, ia, ib, delf_a, delta_a, delf_b, delta_b);
        } else {
            e_p = coaxstk_pair_geom(poss, nx, ny, nz, par, box, ia, ib, delf_a, delta_a, delf_b, delta_b);
        }
        if (e_p == 0) return;
        ev += e_p;
        scatter_pair(sf.template access<OxScatterAtomic>(), st.template access<OxScatterAtomic>(),
                     ia, ib, delf_a, delta_a, delf_b, delta_b);
    }
};

// pair oxdna/coaxstk (oxDNA1): one thread per atom over the half list (it is
// not registered with the screen in LAMMPS); atom a in registers. With the
// LAMMPS-only terminal criterion (kk-fixes: also oxDNA1, together with the
// mirrored theta4 lobe) a non-terminal atom a returns before any load of its
// frame, and non-terminal neighbours b are skipped after the special test.
template <bool LMP = false, bool TAB = false>
struct Coaxstk1Functor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<LR_bonds> bonds;
    RandomRead<int> btype;
    Kokkos::View<const int *>  ilist;
    Kokkos::View<const int *>  num_neigh;
    Kokkos::View<const int **> neigh_matrix;
    TabView2<CxstCoeffs> tcx;
    DNAParams par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int ii) const { c_acc ev=0; (*this)(ii, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int ii, c_acc &ev) const {
        const int ia = LMP ? ilist(ii) : ii;
        if (par.cxst_terminal_only) {
            const LR_bonds ba = bonds(ia);
            if (ba.n3 >= 0 && ba.n5 >= 0) return;    // a has to be a terminal nucleotide
        }
        const int m = num_neigh(ia);
        const int ta = TAB ? btype(ia) : 0;
        c_acc ftmp[3] = {0,0,0}, ttmp[3] = {0,0,0};
        auto af = sf.access();
        auto at = st.access();
        for (int k = 0; k < m; k++) {
            const int braw = neigh_matrix(ia, k);
            if (ox_sbmask(braw)) continue;           // special_lj = 0
            const int ib = braw & OX_NEIGHMASK;
            if (par.cxst_terminal_only) {
                const LR_bonds bb = bonds(ib);
                if (bb.n3 >= 0 && bb.n5 >= 0) continue;   // b has to be terminal
            }
            c_number delf_a[3]={0,0,0}, delf_b[3]={0,0,0};
            c_number delta_a[3]={0,0,0}, delta_b[3]={0,0,0};
            c_number e_p;
            if (TAB) {
                const CxstCoeffs cc = cxst_coeffs(tcx(ta+1, btype(ib)+1), par);
                e_p = coaxstk_pair_geom(poss, nx, ny, nz, cc, box, ia, ib, delf_a, delta_a, delf_b, delta_b);
            } else {
                e_p = coaxstk_pair_geom(poss, nx, ny, nz, par, box, ia, ib, delf_a, delta_a, delf_b, delta_b);
            }
            if (e_p == 0) continue;
            ev += e_p;
            for (int c = 0; c < 3; c++) { ftmp[c] += delf_a[c]; ttmp[c] += delta_a[c]; }
            af(ib,0) += delf_b[0];  af(ib,1) += delf_b[1];  af(ib,2) += delf_b[2];
            at(ib,0) += delta_b[0]; at(ib,1) += delta_b[1]; at(ib,2) += delta_b[2];
        }
        af(ia,0) += ftmp[0]; af(ia,1) += ftmp[1]; af(ia,2) += ftmp[2];
        at(ia,0) += ttmp[0]; at(ia,1) += ttmp[1]; at(ia,2) += ttmp[2];
    }
};

// -----------------------------------------------------------------------
// FUSED hbond+xstk (bench-only experiment, fuse_hbond_xstk toggle): one thread
// per screened pair computes BOTH terms, sharing the base-site geometry and
// scattering once. No LAMMPS equivalent; physics identical up to summation order.
// -----------------------------------------------------------------------
struct HbondXstkFusedFunctor {
    Vec4cr poss, nx, ny, nz;
    RandomRead<int> btype;
    Kokkos::View<const uint64_t *> sp;
    DNAParams par;
    ScatterF4 sf, st;
    SimBox box;

    KOKKOS_INLINE_FUNCTION void operator()(int e) const { c_acc ev=0; (*this)(e, ev); }

    KOKKOS_INLINE_FUNCTION
    void operator()(int e, c_acc &ev) const {
        int ia, braw;
        unpack_pair(sp(e), ia, braw);
        if (ox_sbmask(braw)) return;                 // special_lj = 0
        const int ib = braw & OX_NEIGHMASK;

        c_number dx = poss(ib,0)-poss(ia,0), dy = poss(ib,1)-poss(ia,1), dz = poss(ib,2)-poss(ia,2);
        box.wrap(dx, dy, dz);
        c_number a1[3], a2[3], a3[3], b1[3], b2[3], b3[3];
        load_frame(nx, ny, nz, ia, a1, a2, a3);
        load_frame(nx, ny, nz, ib, b1, b2, b3);
        c_number d_cbs = par.d_cbs;
        c_number ra_cbs[3] = {d_cbs*a1[0], d_cbs*a1[1], d_cbs*a1[2]};
        c_number rb_cbs[3] = {d_cbs*b1[0], d_cbs*b1[1], d_cbs*b1[2]};
        c_number d[3] = {dx+rb_cbs[0]-ra_cbs[0], dy+rb_cbs[1]-ra_cbs[1], dz+rb_cbs[2]-ra_cbs[2]};
        c_number rsq = fma_dot3(d);
        if (rsq <= 0) return;
        c_number r = Kokkos::sqrt(rsq);
        c_number rinv = 1 / r;

        c_number delf_a[3]={0,0,0}, delf_b[3]={0,0,0};
        c_number delta_a[3]={0,0,0}, delta_b[3]={0,0,0};
        c_number e_p = 0;
        c_number alpha = par.alpha_hb[btype(ia)][btype(ib)];
        if (alpha != 0) {
            e_p += hbond_pair(ra_cbs, rb_cbs, d, r, rinv, par, alpha,
                              a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
        }
        e_p += crst_pair(ra_cbs, rb_cbs, d, r, rinv, par,
                         a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
        if (e_p == 0) return;
        ev += e_p;
        scatter_pair(sf.template access<OxScatterAtomic>(), st.template access<OxScatterAtomic>(),
                     ia, ib, delf_a, delta_a, delf_b, delta_b);
    }
};

// -----------------------------------------------------------------------
// Per-style launchers. Each makes its own ScatterView (like each LAMMPS pair
// style's dup_f / dup_torque) and runs parallel_reduce only on energy steps.
// -----------------------------------------------------------------------
template <class Policy, class F>
inline c_acc launch_term(const char *label, int n, const F &f, bool want_energy) {
    c_acc e = 0;
    if (n <= 0) return e;
    if (want_energy) Kokkos::parallel_reduce(label, Policy(0, n), f, e);
    else             Kokkos::parallel_for(label, Policy(0, n), f);
    return e;
}

// the lammps_ilist view of the neighbor list, or an empty one (lean)
inline bool nl_has_ilist(const ParticleArrays &p, const NeighborList &nl) {
    return nl.d_ilist.extent_int(0) >= p.N && p.N > 0;
}

inline c_acc run_excv(ParticleArrays &p, const NeighborList &nl, const DNAParams &par,
                      const SimBox &box, bool want_energy, bool lammps_overhead,
                      bool neigh_rebuilt) {
    const bool tab = lammps_overhead && p.use_tables;
    if (lammps_overhead) {
        if (tab) p.ensure_tables(par);
        else     ensure_tet_excv(p, par);
        if (neigh_rebuilt || nl.prime_pair.extent(0) == 0) build_prime_pair(p, nl);
    }
    auto setup = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
        f.bonds = p.bonds; f.btype = p.btype; f.tag = p.tag;
        f.num_neigh = nl.d_num_neigh; f.neigh_matrix = nl.d_neigh_matrix;
        if (lammps_overhead) {
            f.prime_pair = nl.prime_pair;
            f.bsbs4 = tab ? p.tab.excv_bsbs4 : p.tet_excv_bsbs;
            f.ilist = nl.d_ilist;
        }
        if (tab) { f.t_bkbk = p.tab.excv_bkbk; f.t_bkbs = p.tab.excv_bkbs; f.t_bsbs = p.tab.excv_bsbs; }
        f.par = par; f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques); f.box = box;
    };
    if (lammps_overhead && !nl_has_ilist(p, nl)) {
        // minimum-image list without ilist: give the LAMMPS-mode kernels an identity ilist
        nl.d_ilist = Kokkos::View<int *>("neighlist:ilist", nl.d_num_neigh.extent(0));
        auto il = nl.d_ilist;
        Kokkos::parallel_for("neighlist_ilist_init", il.extent_int(0), KOKKOS_LAMBDA(int i) { il(i) = i; });
    }
    const char *label = "oxdna_excv";
    if (par.pb2 != 0) {
        if (!lammps_overhead) { ExcvFunctor<true, false, false> f; setup(f); return launch_term<OxdnaRangePolicy>(label, p.N, f, want_energy); }
        if (!tab)             { ExcvFunctor<true, true,  false> f; setup(f); return launch_term<OxdnaRangePolicy>(label, p.N, f, want_energy); }
        ExcvFunctor<true, true, true> f; setup(f); return launch_term<OxdnaRangePolicy>(label, p.N, f, want_energy);
    }
    if (!lammps_overhead) { ExcvFunctor<false, false, false> f; setup(f); return launch_term<OxdnaRangePolicy>(label, p.N, f, want_energy); }
    if (!tab)             { ExcvFunctor<false, true,  false> f; setup(f); return launch_term<OxdnaRangePolicy>(label, p.N, f, want_energy); }
    ExcvFunctor<false, true, true> f; setup(f); return launch_term<OxdnaRangePolicy>(label, p.N, f, want_energy);
}

inline c_acc run_hbond_xstk(ParticleArrays &p, const NeighborList &nl, const DNAParams &par,
                            const SimBox &box, bool want_energy, bool fuse_hbond_xstk,
                            bool lammps_overhead = false) {
    using Plain = Kokkos::RangePolicy<>;
    const bool tab = lammps_overhead && p.use_tables;
    if (tab) p.ensure_tables(par);
    c_acc e = 0;
    if (fuse_hbond_xstk) {
        HbondXstkFusedFunctor f;
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz; f.btype = p.btype;
        f.sp = nl.screened_pair; f.par = par; f.box = box;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        return launch_term<Plain>("oxdna_hbond_xstk", nl.N_screened, f, want_energy);
    }
    auto hb = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz; f.btype = p.btype;
        f.sp = nl.screened_pair; f.par = par; f.box = box;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        return launch_term<Plain>("oxdna_hbond", nl.N_screened, f, want_energy);
    };
    if (tab) { HbondFunctor<true> f; f.thb = p.tab.hb; e += hb(f); }
    else     { HbondFunctor<false> f; e += hb(f); }
    auto xs = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz; f.btype = p.btype;
        f.sp = nl.screened_pair; f.par = par; f.box = box;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        return launch_term<Plain>("oxdna_xstk", nl.N_screened, f, want_energy);
    };
    if (tab) { XstkFunctor<true> f; f.txs = p.tab.xstk; e += xs(f); }
    else     { XstkFunctor<false> f; e += xs(f); }
    return e;
}

inline c_acc run_coaxstk(ParticleArrays &p, const NeighborList &nl, const DNAParams &par,
                         const SimBox &box, bool want_energy, bool lammps_overhead = false) {
    using Plain = Kokkos::RangePolicy<>;
    const bool tab = lammps_overhead && p.use_tables;
    if (tab) p.ensure_tables(par);
    if (par.model == 1) {
        auto run = [&](auto &f) {
            f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz;
            f.bonds = p.bonds; f.btype = p.btype;
            f.num_neigh = nl.d_num_neigh; f.neigh_matrix = nl.d_neigh_matrix; f.ilist = nl.d_ilist;
            f.par = par; f.box = box;
            f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
            return launch_term<Plain>("oxdna_coaxstk", p.N, f, want_energy);
        };
        if (!lammps_overhead) { Coaxstk1Functor<false, false> f; return run(f); }
        if (!tab)             { Coaxstk1Functor<true, false> f; return run(f); }
        Coaxstk1Functor<true, true> f; f.tcx = p.tab.cxst; return run(f);
    }
    auto run2 = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.nz = p.nz; f.bonds = p.bonds; f.btype = p.btype;
        f.sp = nl.screened_pair; f.par = par; f.box = box;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        return launch_term<Plain>("oxdna2_coaxstk", nl.N_screened, f, want_energy);
    };
    if (tab) { Coaxstk2Functor<true> f; f.tcx = p.tab.cxst; return run2(f); }
    Coaxstk2Functor<false> f; return run2(f);
}

inline c_acc run_dh(ParticleArrays &p, const NeighborList &nl, const DNAParams &par,
                    const SimBox &box, bool want_energy, bool lammps_overhead = false) {
    if (!par.dh_enabled) return 0;
    p.ensure_qeff(par.dh_half_ends);
    const bool tab = lammps_overhead && p.use_tables && par.model != 3;
    if (tab) p.ensure_tables(par);
    auto run = [&](auto &f) {
        f.poss = p.poss; f.nx = p.nx; f.ny = p.ny; f.qeff = p.qeff; f.btype = p.btype;
        f.num_neigh = nl.d_num_neigh; f.neigh_matrix = nl.d_neigh_matrix; f.ilist = nl.d_ilist;
        f.par = par; f.rcsq = par.dh_RC * par.dh_RC; f.box = box;
        f.sf = ScatterF4(p.forces); f.st = ScatterF4(p.torques);
        return launch_term<OxdnaRangePolicy>("oxdna2_dh", p.N, f, want_energy);
    };
    if (!lammps_overhead) { DHFunctor<false, false> f; return run(f); }
    if (!tab)             { DHFunctor<true, false> f; return run(f); }
    DHFunctor<true, true> f; f.tdh = p.tab.dh; return run(f);
}

// -----------------------------------------------------------------------
// Nonbonded driver for standalone callers (fd_test / xcheck): LRF + excv
// (incl. the bonded excluded volume, as in LAMMPS) + hbond + xstk + coaxstk
// + dh. The MD loop uses compute_pair_forces_step() (bonded.h), which also
// interleaves stk in LAMMPS' pair_style hybrid/overlay order.
// -----------------------------------------------------------------------
inline c_number compute_nonbonded_forces(
    ParticleArrays &p,
    const NeighborList &nl,
    const DNAParams &par,
    const SimBox &box,
    bool want_energy = true,
    bool lammps_overhead = false,
    bool fuse_hbond_xstk = false,
    bool neigh_rebuilt = true)
{
    compute_lrf(p, lammps_overhead);
    c_acc e = 0;
    e += run_excv(p, nl, par, box, want_energy, lammps_overhead, neigh_rebuilt);
    e += run_hbond_xstk(p, nl, par, box, want_energy, fuse_hbond_xstk, lammps_overhead);
    e += run_coaxstk(p, nl, par, box, want_energy, lammps_overhead);
    e += run_dh(p, nl, par, box, want_energy, lammps_overhead);
    return static_cast<c_number>(e);
}
