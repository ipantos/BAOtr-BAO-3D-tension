"""
generate_tables.py — Produce all numerical tables for the paper.

Tables produced:
  Table 1: Overcorrection from the rescaling approach
  Table 5: CMB-consistent (Omega_m, H0) at published posteriors
  Table 6: Per-point baseline tension under LCDM (both methods)
  Table 7: Chi2 summary for all models (both methods)
  Table 8: Sensitivity to extrapolation scheme and BGS anchor

Usage:
  python generate_tables.py
"""

import numpy as np
from cosmo import (
    PUBLISHED_MODELS, DESI_DATA, SDSS_DATA,
    Z_BAOTR, DM_BAOTR, SIG_BAOTR,
    best_fit_desi, method_alpha, chi2_baotr,
    H0_from_theta_star, OM_PLANCK, H0_PLANCK,
    DM_over_rd, DV_over_rd, DH_over_rd,
)


# ══════════════════════════════════════════════════════════════════
#  Table 1: Overcorrection from the rescaling approach
# ══════════════════════════════════════════════════════════════════
def table1_overcorrection():
    """
    Table 1: Overcorrection from the rescaling approach.

    R_perp(z) = D_M^CPL(z) / D_M^LCDM(z) is evaluated numerically
    at each DESI redshift using the CMB+DESI CPL model
    (w0 = -0.42, wa = -1.75) and the Planck 2018 LCDM baseline,
    both at fixed (Omega_m = 0.3153, H0 = 67.36).

    For BGS (z = 0.295, isotropic), R_V = D_V^CPL / D_V^LCDM.

    The "True residual" column quotes the mock-validated residual
    fiducial bias from DESI (AbdulKarim et al. 2025;
    Perez-Fernandez et al. 2024); these are NOT computed here.
    """
    # DESI redshifts (all except z = 1.484)
    z_table = [0.295, 0.510, 0.706, 0.934, 1.321, 2.330]

    # Mock-validated residual fiducial bias (from DESI papers)
    true_residual = {
        0.295: 0.3,
        0.510: 0.3,
        0.706: 0.2,
        0.934: 0.2,
        1.321: 0.1,
        2.330: 0.1,
    }

    # CPL model: CMB+DESI best fit
    w0_cpl, wa_cpl = -0.42, -1.75

    # Fixed Planck parameters for both models
    Om, H0 = OM_PLANCK, H0_PLANCK

    print("=" * 85)
    print("TABLE 1: Overcorrection from rescaling approach")
    print(f"  CPL: w0 = {w0_cpl}, wa = {wa_cpl}")
    print(f"  Fixed background: Om = {Om}, H0 = {H0}")
    print("  R_perp = D_M^CPL / D_M^LCDM  (R_V for BGS)")
    print("=" * 85)
    print(f"{'z':>6s} {'(D/rd)pub':>10s} {'R':>7s} {'Rescaled':>9s} "
          f"{'Shift%':>8s} {'Resid%':>8s} {'Overcorr':>10s}")
    print("-" * 70)

    for z in z_table:
        obs_type, val_pub, sig_pub = DESI_DATA[z]

        if obs_type == 'DV':
            d_cpl  = DV_over_rd(z, Om, H0, w0_cpl, wa_cpl)
            d_lcdm = DV_over_rd(z, Om, H0, -1.0, 0.0)
        else:
            d_cpl  = DM_over_rd(z, Om, H0, w0_cpl, wa_cpl)
            d_lcdm = DM_over_rd(z, Om, H0, -1.0, 0.0)

        R = d_cpl / d_lcdm
        rescaled = val_pub * R
        shift_pct = (R - 1.0) * 100.0
        resid = true_residual[z]
        overcorr = abs(shift_pct) / resid

        tag = '*' if obs_type == 'DV' else ' '
        print(f"{z:6.3f}{tag} {val_pub:10.3f} {R:7.3f} {rescaled:9.3f} "
              f"{shift_pct:+8.1f}   <={resid:.1f}     "
              f">={overcorr:.0f}x")

    print()
    print("  * BGS D_V/r_d; R here denotes R_V = D_V^CPL / D_V^LCDM")
    print()


# ══════════════════════════════════════════════════════════════════
#  Table 5: CMB-consistent parameters
# ══════════════════════════════════════════════════════════════════
def table5_parameters():
    """Table 5: DESI-optimized parameters at published posteriors."""
    print("=" * 90)
    print("TABLE 5: CMB-consistent (Omega_m, H0) at published posteriors")
    print("=" * 90)
    print(f"{'Model':<25s} {'w0':>7s} {'wa':>7s} {'Om':>8s} "
          f"{'H0':>8s} {'chi2_D':>8s} {'chi2_B':>8s}")
    print("-" * 90)
    for name, w0, wa in PUBLISHED_MODELS:
        r = best_fit_desi(w0, wa)
        print(f"{name:<25s} {w0:7.3f} {wa:7.3f} {r['Om']:8.4f} "
              f"{r['H0']:8.2f} {r['chi2_3d']:8.1f} {r['chi2_baotr']:8.1f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Table 6: Per-point baseline tension
# ══════════════════════════════════════════════════════════════════
def table6_baseline():
    """Table 6: Per-point baseline tension under DESI-optimized LCDM."""
    r_A = best_fit_desi(-1.0, 0.0)
    r_B = method_alpha(-1.0, 0.0)

    print("=" * 90)
    print(f"TABLE 6: Per-point tension under DESI-optimized LCDM "
          f"(Om={r_A['Om']:.3f}, H0={r_A['H0']:.1f})")
    print("=" * 90)
    print(f"{'#':>2s} {'z':>6s} {'BAOtr':>8s} {'sig':>6s} "
          f"{'T_A':>7s} {'chi2_A':>7s} {'T_B':>7s} {'chi2_B':>7s}")
    print("-" * 60)
    for i in range(len(Z_BAOTR)):
        Ti_A = r_A['Ti_baotr'][i]
        Ti_B = r_B['Ti'][i]
        print(f"{i+1:2d} {Z_BAOTR[i]:6.3f} {DM_BAOTR[i]:8.3f} "
              f"{SIG_BAOTR[i]:6.3f} {Ti_A:7.2f} {Ti_A**2:7.2f} "
              f"{Ti_B:7.2f} {Ti_B**2:7.2f}")
    print(f"{'':>24s} {'Total':>7s} {r_A['chi2_baotr']:7.1f} "
          f"{'':>7s} {r_B['chi2']:7.1f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Table 7: Chi2 summary
# ══════════════════════════════════════════════════════════════════
def table7_chi2_summary():
    """Table 7: Chi2 summary for all published models."""
    print("=" * 100)
    print("TABLE 7: Chi2 summary (both methods, DESI-optimized)")
    print("=" * 100)
    print(f"{'Model':<25s} {'w0':>6s} {'wa':>6s} {'Om':>7s} {'H0':>7s} "
          f"{'A:BAOtr':>8s} {'A:DESI':>7s} {'dA_baotr':>9s} "
          f"{'B:chi2':>7s} {'dB':>7s}")
    print("-" * 100)

    chi2_A_ref = None
    chi2_B_ref = None

    for name, w0, wa in PUBLISHED_MODELS:
        rA = best_fit_desi(w0, wa)
        rB = method_alpha(w0, wa)

        if name == 'LCDM':
            chi2_A_ref = rA['chi2_baotr']
            chi2_B_ref = rB['chi2']
            dA_str = '---'
            dB_str = '---'
        else:
            dA = rA['chi2_baotr'] - chi2_A_ref
            dB = rB['chi2'] - chi2_B_ref
            dA_str = f'{dA:+8.1f}'
            dB_str = f'{dB:+6.1f}'

        print(f"{name:<25s} {w0:6.2f} {wa:6.2f} {rA['Om']:7.4f} "
              f"{rA['H0']:7.2f} {rA['chi2_baotr']:8.1f} "
              f"{rA['chi2_3d']:7.1f} {dA_str:>9s} "
              f"{rB['chi2']:7.1f} {dB_str:>7s}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Table 8: Sensitivity
# ══════════════════════════════════════════════════════════════════
def table8_sensitivity():
    """Table 8: Sensitivity to extrapolation scheme and BGS anchor."""
    print("=" * 80)
    print("TABLE 8: Method B sensitivity (LCDM baseline)")
    print("=" * 80)

    DESI_no_BGS = {k: v for k, v in DESI_DATA.items() if k != 0.295}

    print(f"{'Extrapolation':<30s} {'With BGS':>10s} {'No BGS':>10s} "
          f"{'W/N':>8s} {'No/N':>8s}")
    print("-" * 70)

    for name, ext in [('Constant-alpha', 'constant'),
                      ('Model (alpha=1)', 'model'),
                      ('Linear', 'linear')]:
        r_with = method_alpha(-1.0, 0.0, DESI_DATA, ext)
        r_without = method_alpha(-1.0, 0.0, DESI_no_BGS, ext)
        c_w = r_with['chi2'] if r_with else np.nan
        c_wo = r_without['chi2'] if r_without else np.nan
        print(f"{name:<30s} {c_w:10.1f} {c_wo:10.1f} "
              f"{c_w/15:8.2f} {c_wo/15:8.2f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  SDSS cross-check
# ══════════════════════════════════════════════════════════════════
def table_sdss_crosscheck():
    """SDSS cross-check: Method B with SDSS anchors."""
    print("=" * 60)
    print("SDSS CROSS-CHECK (Method B, LCDM)")
    print("=" * 60)
    r_desi = method_alpha(-1.0, 0.0, DESI_DATA)
    r_sdss = method_alpha(-1.0, 0.0, SDSS_DATA)
    print(f"DESI anchors: chi2_B = {r_desi['chi2']:.1f}, "
          f"H0 = {r_desi['H0']:.1f}")
    print(f"SDSS anchors: chi2_B = {r_sdss['chi2']:.1f}, "
          f"H0 = {r_sdss['H0']:.1f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    table1_overcorrection()
    table5_parameters()
    table6_baseline()
    table7_chi2_summary()
    table8_sensitivity()
    table_sdss_crosscheck()