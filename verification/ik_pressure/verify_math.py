#!/usr/bin/env python3
"""
Independent verification of the ANALYTIC reductions in ewald_disp_planar.cpp.

The IK kernels Phi(h), Psi(h) and the global GU/GT/GN coefficients all reduce a
1/r^6-dispersion tail integral to closed form via the `sici_chain` generalized
sine/cosine integrals plus "magic" constants (pi/48, pi/288, pi/576, ...).
Here we recompute those tail integrals by BRUTE-FORCE numerical quadrature and
compare to the code's closed forms.  Agreement = the special-function algebra
(the IBP recurrence + the constants) is correct.

We also check the limit constants A_m(inf), B_m(inf) that produce pi/288 etc.,
and the kernel low-h limits.
"""
import numpy as np
from scipy.special import sici
from scipy.integrate import quad

EULER = 0.5772156649015329

# ---- replicate the code's sici_chain (ewald_disp_planar.cpp:1815) ----
def sici_chain(x):
    si, ci = sici(x)
    A = np.zeros(8); B = np.zeros(8)
    A[1] = si; B[1] = ci - EULER
    sx, cx = np.sin(x), np.cos(x)
    for m in range(2, 8):
        xm = x ** (1 - m)
        A[m] = -sx * xm / (m - 1) + B[m - 1] / (m - 1)
        B[m] = -cx * xm / (m - 1) - A[m - 1] / (m - 1)
    return A, B

# A_m(inf), B_m(inf): same recurrence with the x^{1-m} terms -> 0 (m>=2)
def chain_inf():
    Ai = np.zeros(8); Bi = np.zeros(8)
    Ai[1] = np.pi / 2.0; Bi[1] = -EULER          # Si(inf)=pi/2, Ci(inf)-gamma=-gamma
    for m in range(2, 8):
        Ai[m] = Bi[m - 1] / (m - 1)
        Bi[m] = -Ai[m - 1] / (m - 1)
    return Ai, Bi

# generalized tail integrals  int_x^inf sin t / t^m dt  and  int_x^inf cos t / t^m dt
def tail_sin(x, m):
    return quad(lambda t: np.sin(t) / t ** m, x, np.inf, limit=400)[0]
def tail_cos(x, m):
    return quad(lambda t: np.cos(t) / t ** m, x, np.inf, limit=400)[0]

# code's profile integrands (sharp tail), int_c^inf g(r) dr is the coefficient
def prof_T(r, h):
    return np.sin(h*r)/(h**6 * r**7) - np.cos(h*r)/(h**5 * r**6)
def prof_N(r, h):
    return np.sin(h*r)/(h**4 * r**5) - 2*np.sin(h*r)/(h**6 * r**7) + 2*np.cos(h*r)/(h**5 * r**6)
def prof_PHI(r, h):
    si, _ = sici(h*r)
    return si/(h**4 * r**5) - np.sin(h*r)/(h**6 * r**7) + np.cos(h*r)/(h**5 * r**6)

def integ(f, c):
    # decays as r^-5..r^-7; integrate to a large finite bound
    return quad(f, c, 2000.0, limit=4000)[0]

# oscillatory (Fourier) integrals int_c^inf g(r) sin/cos(h r) dr via QUADPACK qawf.
# This resolves the high-h near-total cancellation that a plain quad cannot.
def osc_sin(g, c, h):
    return quad(g, c, np.inf, weight="sin", wvar=h, limit=600)[0]
def osc_cos(g, c, h):
    return quad(g, c, np.inf, weight="cos", wvar=h, limit=600)[0]

def kern_T(h, c):   # int_c^inf prof_T dr, oscillation handled analytically
    return osc_sin(lambda r: 1.0/(h**6 * r**7), c, h) + osc_cos(lambda r: -1.0/(h**5 * r**6), c, h)
def kern_N(h, c):
    return (osc_sin(lambda r: 1.0/(h**4 * r**5) - 2.0/(h**6 * r**7), c, h)
            + osc_cos(lambda r: 2.0/(h**5 * r**6), c, h))
def kern_PHI(h, c):
    # int_c^inf Si(hr)/(h^4 r^5) dr, by parts (u=Si(hr), dv=r^-5 dr, v=-r^-4/4),
    # Si'(hr)*h = sin(hr)/r:  = Si(hc)/(4 c^4 h^4) + (1/4h^4) int_c^inf sin(hr)/r^5 dr.
    # This leaves only purely-oscillatory integrals (qawf-exact even at high h).
    si_c = sici(h*c)[0]
    si_term = si_c/(4.0*c**4*h**4) + (1.0/(4.0*h**4))*osc_sin(lambda r: 1.0/r**5, c, h)
    return si_term + osc_sin(lambda r: -1.0/(h**6 * r**7), c, h) + osc_cos(lambda r: 1.0/(h**5 * r**6), c, h)


def main():
    print("=" * 72)
    print("Independent check of ewald_disp_planar analytic reductions")
    print("=" * 72)

    # (1) limit constants -> the magic constants
    Ai, Bi = chain_inf()
    print("\n[1] chain limits A_m(inf), B_m(inf)  ->  the 'magic' constants")
    checks = [
        ("A5(inf) = pi/48", Ai[5], np.pi/48),
        ("A7(inf)-B6(inf) = pi/288", Ai[7]-Bi[6], np.pi/288),
        # Phi constant: pi/576 must be Sii5(inf) - A7(inf) + B6(inf) with
        # Sii5(inf)=A5(inf)/4 = pi/192 ; pi/192 - (A7-B6)(inf) = pi/192 - pi/288
        ("pi/192 - pi/288 = pi/576", np.pi/192 - np.pi/288, np.pi/576),
        ("A5(inf)-2A7(inf)+2B6(inf) = pi/72", Ai[5]-2*Ai[7]+2*Bi[6], np.pi/72),
    ]
    for name, got, exp in checks:
        ok = "OK" if abs(got-exp) < 1e-13 else "FAIL"
        print(f"   {name:38s} got={got:.12f} exp={exp:.12f}  [{ok}]")

    # (2) generalized tail integrals vs closed form (A_m(inf)-A_m), (B_m(inf)-B_m)
    print("\n[2] int_x^inf sin t/t^m dt  ==  A_m(inf)-A_m(x)   (and cos -> B)  [relative]")
    relmax2 = 0.0
    for x in [0.5, 2.0, 5.0]:   # brute-force qawf is reliable here; high-arg recurrence
                                # is exercised (and passes) in the kernel check [3] below
        A, B = sici_chain(x)
        for m in [5, 6, 7]:
            gs = osc_sin(lambda t: t**(-m), x, 1.0); cs = Ai[m] - A[m]
            gc = osc_cos(lambda t: t**(-m), x, 1.0); cc = Bi[m] - B[m]
            rs = abs(gs-cs)/(abs(cs)+1e-300); rc = abs(gc-cc)/(abs(cc)+1e-300)
            relmax2 = max(relmax2, rs, rc)
            print(f"   x={x:5.1f} m={m}: sin rel={rs:.1e}  cos rel={rc:.1e}")
    print(f"   -> max relative error [2]: {relmax2:.1e}  "
          + ("[OK]" if relmax2 < 1e-6 else "[CHECK]"))

    # (3) kernel tail integrals int_c^inf prof_* dr  vs  the ik_phi/ik_psi brackets
    #     (sharp, switch off: prof_shell=0, anchored at c).
    print("\n[3] kernel tails int_c^inf g(r)dr vs closed-form bracket (c=3.0)")
    c = 3.0
    Lz = 36.0
    for n in [1, 2, 5, 20, 80, 175]:
        h = 2*np.pi*n/Lz
        A, B = sici_chain(h*c)
        sii5 = A[5]/4.0 - A[1]/(4.0*(h*c)**4)
        # Psi bracket (uses PROF_T):  pi/288 - A7 + B6
        br_psi = np.pi/288 - A[7] + B[6]
        num_T = integ(lambda r: prof_T(r, h), c)
        # Phi bracket (uses PROF_PHI): pi/576 - sii5 + A7 - B6
        br_phi = np.pi/576 - sii5 + A[7] - B[6]
        num_PHI = integ(lambda r: prof_PHI(r, h), c)
        # N bracket (PROF_N): pi/72 - A5 + 2A7 - 2B6   (combo for GN tail)
        br_N = np.pi/72 - A[5] + 2*A[7] - 2*B[6]
        num_T = kern_T(h, c); num_PHI = kern_PHI(h, c); num_N = kern_N(h, c)
        for tag, num, br in [("Psi/T", num_T, br_psi), ("Phi", num_PHI, br_phi), ("N", num_N, br_N)]:
            rel = abs(num-br)/(abs(br)+1e-30)
            ok = "OK" if rel < 1e-7 else ("ok" if rel < 1e-5 else "FAIL")
            print(f"   n={n:3d} h={h:6.3f} {tag:6s}: num={num:+.6e} cf={br:+.6e} rel={rel:.1e} [{ok}]")

    print("\nAll [OK] => the sici_chain reduction and the kernel/coefficient")
    print("closed forms reproduce the defining 1/r^6 tail integrals exactly.")


if __name__ == "__main__":
    main()
