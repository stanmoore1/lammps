#pragma once

// DNA nonbonded force computation kernel (oxDNA1).
// One Kokkos thread per edge (pair i,j) — no inner loop, perfect load balance.
//
// Physics (matches the standalone DNAInteraction::pair_interaction_nonbonded):
//   nonbonded excluded volume + hydrogen bonding + cross stacking + coaxial stacking.
// Each angular/dihedral term is a faithful port of the standalone force/torque
// expressions; torques are accumulated in the lab frame (the integrator uses unit
// isotropic inertia, so lab-frame angular momentum integration is exact).

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
KOKKOS_INLINE_FUNCTION
c_number hbond_pair(const c_number ra_bs[3], const c_number rb_bs[3],
                    const c_number delr_bs[3], c_number r_bs, c_number rinv,
                    const DNAParams &par, c_number alpha,
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
KOKKOS_INLINE_FUNCTION
c_number crst_pair(const c_number ra_bs[3], const c_number rb_bs[3],
                   const c_number delr_bs[3], c_number r_bs, c_number rinv,
                   const DNAParams &par,
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
KOKKOS_INLINE_FUNCTION
c_number cxst_pair(const c_number ra_st[3], const c_number rb_st[3],
                   const c_number delr_st[3], c_number r_st, c_number rinv,
                   const c_number delr_com[3], const DNAParams &par,
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
    c_number f4t1Ds = D1, f4t4Ds = -D4;

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

// -----------------------------------------------------------------------
// Main nonbonded force dispatch — one kernel per pair (flat edge list)
// -----------------------------------------------------------------------
struct DNAForcesFunctor {
    Kokkos::View<const c_number *[4]> poss;
    Kokkos::View<const c_number *[4]> orientations;
    Kokkos::View<const int *>         btype;
    Kokkos::View<const LR_bonds *>    bonds;
    Kokkos::View<const int *>         edge_i;
    Kokkos::View<const int *>         edge_j;

    DNAParams par;

    using ScatterF = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::DefaultExecutionSpace::array_layout,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    ScatterF sf;  // forces
    ScatterF st;  // torques

    SimBox box;

    KOKKOS_INLINE_FUNCTION
    void operator()(int edge, c_number &ev) const {
        const int ia = edge_i(edge);
        const int ib = edge_j(edge);

        c_number xai = poss(ia,0), yai = poss(ia,1), zai = poss(ia,2);
        c_number xbi = poss(ib,0), ybi = poss(ib,1), zbi = poss(ib,2);

        c_number dx = xbi - xai, dy = ybi - yai, dz = zbi - zai;
        box.wrap(dx, dy, dz);
        c_number delr_com[3] = {dx, dy, dz};

        c_number a1[3], a2[3], a3[3];
        c_number b1[3], b2[3], b3[3];
        get_vectors_from_quat_view(orientations, ia, a1, a2, a3);
        get_vectors_from_quat_view(orientations, ib, b1, b2, b3);

        c_number d_cbs = par.d_cbs, d_cstk = par.d_cstk;
        c_number pb1 = par.pb1, pb2 = par.pb2;

        // Backbone site (grooved for oxDNA2): pb1*a1 + pb2*a2
        c_number ra_cbk[3] = {pb1*a1[0]+pb2*a2[0], pb1*a1[1]+pb2*a2[1], pb1*a1[2]+pb2*a2[2]};
        c_number rb_cbk[3] = {pb1*b1[0]+pb2*b2[0], pb1*b1[1]+pb2*b2[1], pb1*b1[2]+pb2*b2[2]};
        c_number ra_cbs[3] = {d_cbs*a1[0], d_cbs*a1[1], d_cbs*a1[2]};
        c_number rb_cbs[3] = {d_cbs*b1[0], d_cbs*b1[1], d_cbs*b1[2]};

        c_number delf_a[3] = {0,0,0}, delf_b[3] = {0,0,0};
        c_number delta_a[3]= {0,0,0}, delta_b[3]= {0,0,0};
        c_number evdwl = 0;

        // ---- Nonbonded excluded volume (base-base, base-back, back-base, back-back) ----
        {
            c_number d[3] = {dx + rb_cbs[0] - ra_cbs[0], dy + rb_cbs[1] - ra_cbs[1], dz + rb_cbs[2] - ra_cbs[2]};
            add_excv_contrib(ra_cbs, rb_cbs, d, par.excv_bsbs, dot3(d,d), delf_a, delta_a, delf_b, delta_b, evdwl);
        }
        {
            c_number d[3] = {dx + rb_cbs[0] - ra_cbk[0], dy + rb_cbs[1] - ra_cbk[1], dz + rb_cbs[2] - ra_cbk[2]};
            add_excv_contrib(ra_cbk, rb_cbs, d, par.excv_bkbs, dot3(d,d), delf_a, delta_a, delf_b, delta_b, evdwl);
        }
        {
            c_number d[3] = {dx + rb_cbk[0] - ra_cbs[0], dy + rb_cbk[1] - ra_cbs[1], dz + rb_cbk[2] - ra_cbs[2]};
            add_excv_contrib(ra_cbs, rb_cbk, d, par.excv_bkbs, dot3(d,d), delf_a, delta_a, delf_b, delta_b, evdwl);
        }
        {
            c_number d[3] = {dx + rb_cbk[0] - ra_cbk[0], dy + rb_cbk[1] - ra_cbk[1], dz + rb_cbk[2] - ra_cbk[2]};
            add_excv_contrib(ra_cbk, rb_cbk, d, par.excv_bkbk, dot3(d,d), delf_a, delta_a, delf_b, delta_b, evdwl);
        }

        // ---- Base-site separation (H-bonding + cross stacking) ----
        {
            c_number d[3] = {dx + rb_cbs[0] - ra_cbs[0], dy + rb_cbs[1] - ra_cbs[1], dz + rb_cbs[2] - ra_cbs[2]};
            c_number rsq = dot3(d,d);
            c_number r = Kokkos::sqrt(rsq);
            if (r > 0) {
                c_number rinv = 1 / r;
                int at = btype(ia), bt = btype(ib);
                c_number alpha = par.alpha_hb[at][bt];
                if (alpha != 0)
                    evdwl += hbond_pair(ra_cbs, rb_cbs, d, r, rinv, par, alpha,
                                        a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
                evdwl += crst_pair(ra_cbs, rb_cbs, d, r, rinv, par,
                                   a1, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
            }
        }

        // ---- Stacking-site separation (coaxial stacking) ----
        {
            c_number ra_st[3] = {d_cstk*a1[0], d_cstk*a1[1], d_cstk*a1[2]};
            c_number rb_st[3] = {d_cstk*b1[0], d_cstk*b1[1], d_cstk*b1[2]};
            c_number d[3] = {dx + rb_st[0] - ra_st[0], dy + rb_st[1] - ra_st[1], dz + rb_st[2] - ra_st[2]};
            c_number rsq = dot3(d,d);
            c_number r = Kokkos::sqrt(rsq);
            if (r > 0) {
                c_number rinv = 1 / r;
                evdwl += cxst_pair(ra_st, rb_st, d, r, rinv, delr_com, par,
                                   a1, a2, a3, b1, b3, delf_a, delta_a, delf_b, delta_b);
            }
        }

        // ---- Debye-Huckel electrostatics (oxDNA2; backbone-site separation) ----
        if (par.dh_enabled) {
            c_number d[3] = {dx + rb_cbk[0] - ra_cbk[0], dy + rb_cbk[1] - ra_cbk[1], dz + rb_cbk[2] - ra_cbk[2]};
            c_number rmod = Kokkos::sqrt(dot3(d,d));
            if (rmod > 0 && rmod < par.dh_RC) {
                c_number cut_factor = 1;
                if (par.dh_half_ends) {
                    if (bonds(ia).n3 < 0 || bonds(ia).n5 < 0) cut_factor *= c_number(0.5);
                    if (bonds(ib).n3 < 0 || bonds(ib).n5 < 0) cut_factor *= c_number(0.5);
                }
                evdwl += dh_pair(ra_cbk, rb_cbk, d, rmod, cut_factor, par,
                                 delf_a, delta_a, delf_b, delta_b);
            }
        }

        ev += evdwl;

        // Skip the atomic accumulation entirely for pairs that contributed
        // nothing (in the Verlet list but beyond every interaction cutoff) —
        // mirrors oxDNA's `if(dF·dF > 0)` / `if(dT·dT > 0)` atomic guards.
        c_number nz = dot3(delf_a,delf_a) + dot3(delta_a,delta_a)
                    + dot3(delf_b,delf_b) + dot3(delta_b,delta_b);
        if (nz > c_number(0)) {
            auto af = sf.access();
            auto at_v = st.access();
            af(ia, 0) += delf_a[0]; af(ia, 1) += delf_a[1]; af(ia, 2) += delf_a[2];
            af(ib, 0) += delf_b[0]; af(ib, 1) += delf_b[1]; af(ib, 2) += delf_b[2];
            at_v(ia, 0) += delta_a[0]; at_v(ia, 1) += delta_a[1]; at_v(ia, 2) += delta_a[2];
            at_v(ib, 0) += delta_b[0]; at_v(ib, 1) += delta_b[1]; at_v(ib, 2) += delta_b[2];
        }
    }
};

// -----------------------------------------------------------------------
// Compute all nonbonded forces for all N_edges pairs
// -----------------------------------------------------------------------
inline c_number compute_nonbonded_forces(
    ParticleArrays &p,
    const NeighborList &nl,
    const DNAParams &par,
    const SimBox &box)
{
    if (nl.N_edges == 0) return 0;

    using SV = Kokkos::Experimental::ScatterView<
        c_number *[4],
        Kokkos::DefaultExecutionSpace::array_layout,
        Kokkos::DefaultExecutionSpace,
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>;

    SV sf(p.forces);
    SV st(p.torques);

    DNAForcesFunctor fun;
    fun.poss         = p.poss;
    fun.orientations = p.orientations;
    fun.btype        = p.btype;
    fun.bonds        = p.bonds;
    fun.edge_i       = nl.edge_i;
    fun.edge_j       = nl.edge_j;
    fun.par          = par;
    fun.sf           = sf;
    fun.st           = st;
    fun.box          = box;

    c_number etot = 0;
    Kokkos::parallel_reduce("dna_forces_nonbonded",
        Kokkos::RangePolicy<>(0, nl.N_edges), fun, etot);

    Kokkos::Experimental::contribute(p.forces, sf);
    Kokkos::Experimental::contribute(p.torques, st);

    return etot;
}
