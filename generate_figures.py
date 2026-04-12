"""
generate_figures.py — Produce all figures for the paper.

Figures:
  1. Per-point tension bar chart (LCDM baseline, both methods)
  2. Trade-off plot: chi2_DESI vs chi2_BAOtr (central figure)
  3. Chi2 surfaces in the (w0, wa) plane
  4. DESI vs SDSS per-point tensions (Method B)
  5. Residual plot relative to Planck LCDM (both methods)

Usage:
  python generate_figures.py

Output:
  figures/fig1_tension_bars.pdf
  figures/fig2_tradeoff.pdf
  figures/fig3_chi2_surfaces.pdf
  figures/fig4_sdss_comparison.pdf
  figures/fig5_methods_comparison.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150,
})

from cosmo import (
    PUBLISHED_MODELS, DESI_DATA, SDSS_DATA,
    Z_BAOTR, DM_BAOTR, SIG_BAOTR,
    OM_PLANCK, H0_PLANCK,
    best_fit_desi, method_alpha, chi2_baotr,
    DM_over_rd, DH_over_rd, DV_over_rd,
    H0_from_theta_star, alpha_warped_dense,
    chi2_3d as chi2_3d_func,
)

OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# Precompute published models
# ══════════════════════════════════════════════════════════════════
print("Computing published models...")
RES_A, RES_B = {}, {}
for name, w0, wa in PUBLISHED_MODELS:
    RES_A[name] = best_fit_desi(w0, wa)
    RES_B[name] = method_alpha(w0, wa)
    rA, rB = RES_A[name], RES_B[name]
    print(f"  {name:<25s} Om={rA['Om']:.4f} H0={rA['H0']:.2f} "
          f"A:D={rA['chi2_3d']:.1f} A:B={rA['chi2_baotr']:.1f} "
          f"B:{rB['chi2']:.1f}")


# ══════════════════════════════════════════════════════════════════
# Figure 1: Per-point tension bars
# ══════════════════════════════════════════════════════════════════
def figure1():
    print("\nFigure 1: tension bars...")
    rA = RES_A['LCDM']
    rB = RES_B['LCDM']
    x = np.arange(len(Z_BAOTR))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
        height_ratios=[1, 0.8], sharex=True,
        gridspec_kw={'hspace': 0.08})

    ax1.bar(x - w/2, rA['Ti_baotr'], w, color='steelblue', alpha=0.85,
            label=f"Method A (direct), $\\chi^2={rA['chi2_baotr']:.1f}$")
    ax1.bar(x + w/2, rB['Ti'], w, color='grey', alpha=0.6,
            label=f"Method B ($\\alpha$-interp.), $\\chi^2={rB['chi2']:.1f}$")
    ax1.axhline(0, color='k', lw=0.5)
    ax1.axhspan(-1, 1, color='green', alpha=0.08)
    ax1.set_ylabel(r'Tension $T_i$ [$\sigma$]')
    ax1.set_ylim(-5, 1.5)
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.text(0.98, 0.95,
             r'$\Lambda$CDM (DESI-optimized)' +
             f'\n$\\Omega_m={rA["Om"]:.3f}$, $H_0={rA["H0"]:.1f}$',
             transform=ax1.transAxes, ha='right', va='top', fontsize=10)

    chi2_i = rA['Ti_baotr']**2
    colors = ['red' if c > 4 else 'steelblue' if c > 1
              else 'lightgrey' for c in chi2_i]
    ax2.bar(x, chi2_i, 0.6, color=colors, edgecolor='grey', lw=0.5)
    ax2.set_ylabel(r'$\chi^2_i$ (Method A)')
    ax2.set_ylim(0, 16)
    ax2.set_xticks(x)
    labels = [f'{z:.2f}' if z < 1 else f'{z:.3f}' for z in Z_BAOTR]
    labels[-1] = '2.225'
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_xlabel('Redshift $z$')
    ax2.text(0.98, 0.90,
             f'$\\chi^2_{{\\rm tot}}={rA["chi2_baotr"]:.1f}$',
             transform=ax2.transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig1_tension_bars.pdf', bbox_inches='tight')
    print("  Saved fig1_tension_bars.pdf")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Grid scan (shared by Figures 2 & 3)
# ══════════════════════════════════════════════════════════════════
NW = 55
W0_ARR = np.linspace(-2.5, 0.5, NW)
WA_ARR = np.linspace(-5.5, 2.5, NW)
G_C2D = np.full((NW, NW), np.nan)
G_C2B = np.full((NW, NW), np.nan)
SC_W0, SC_CD, SC_CB = [], [], []


def run_grid_scan():
    global G_C2D, G_C2B, SC_W0, SC_CD, SC_CB
    print("\nGrid scan (55x55)...")
    cnt = 0
    total = sum(1 for w0 in W0_ARR for wa in WA_ARR if w0 + wa < 0)
    for i, wa in enumerate(WA_ARR):
        for j, w0 in enumerate(W0_ARR):
            if w0 + wa >= 0:
                continue
            cnt += 1
            if cnt % 300 == 0:
                print(f"  {cnt}/{total}...")
            r = best_fit_desi(w0, wa)
            if r is not None:
                G_C2D[i, j] = r['chi2_3d']
                G_C2B[i, j] = r['chi2_baotr']
                SC_W0.append(w0)
                SC_CD.append(r['chi2_3d'])
                SC_CB.append(r['chi2_baotr'])
    print(f"  Done ({cnt} points)")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Trade-off plot
# ══════════════════════════════════════════════════════════════════
def figure2():
    print("\nFigure 2: trade-off plot...")
    fig, ax = plt.subplots(figsize=(8, 6.5))

    sc = ax.scatter(SC_CD, SC_CB, c=SC_W0, cmap='coolwarm', s=10,
                    alpha=0.55, vmin=-2.5, vmax=0.5, rasterized=True)
    plt.colorbar(sc, label=r'$w_0$', pad=0.02)

    markers = {'LCDM': ('k*', 18), 'CMB+PP&SH0ES': ('mo', 12),
               'CMB+DESI': ('rs', 12)}
    for name, (fmt, ms) in markers.items():
        rA = RES_A[name]
        ax.plot(rA['chi2_3d'], rA['chi2_baotr'], fmt, ms=ms,
                zorder=10, markeredgecolor='k', markeredgewidth=1.0,
                label=name.replace('&', r'\&'))

    ax.axhline(15, color='green', ls=':', alpha=0.6, lw=1.2,
               label=r'$\chi^2_{\rm BAOtr}=15$')
    ax.axvline(7, color='blue', ls=':', alpha=0.6, lw=1.2,
               label=r'$\chi^2_{\rm DESI}=7$')
    ax.set_xlabel(r'$\chi^2_{\rm DESI}$', fontsize=14)
    ax.set_ylabel(r'$\chi^2_{\rm BAOtr}$', fontsize=14)
    ax.set_title(r'No CPL model fits both datasets ($H_0$ free)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig2_tradeoff.pdf', bbox_inches='tight')
    print("  Saved fig2_tradeoff.pdf")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 3: Chi2 surfaces
# ══════════════════════════════════════════════════════════════════
def figure3():
    from matplotlib.lines import Line2D
    print("\nFigure 3: chi2 surfaces...")
    W0, WA = np.meshgrid(W0_ARR, WA_ARR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    cs1 = ax1.contourf(W0, WA, G_C2B,
                        levels=np.linspace(10, 100, 25),
                        cmap='RdYlGn_r', extend='both')
    plt.colorbar(cs1, ax=ax1, label=r'$\chi^2_{\rm BAOtr}$')
    ax1.set_title(r'$\chi^2_{\rm BAOtr}(w_0,w_a)$ — DESI-optimized')

    cs2 = ax2.contourf(W0, WA, G_C2D,
                        levels=np.linspace(0, 100, 25),
                        cmap='Reds', extend='max')
    plt.colorbar(cs2, ax=ax2, label=r'$\chi^2_{\rm DESI}$')
    ax2.set_title(r'$\chi^2_{\rm DESI}(w_0,w_a)$ — DESI-optimized')

    for ax in [ax1, ax2]:
        ax.plot(-1, 0, 'k*', ms=14, zorder=5)
        ax.plot(-0.694, -1.700, 'mo', ms=9, zorder=5, mec='k', mew=0.8)
        ax.plot(-0.660, -1.910, 'mD', ms=7, zorder=5, mec='k', mew=0.8)
        ax.plot(-0.480, -1.510, 'c^', ms=9, zorder=5, mec='k', mew=0.8)
        ax.plot(-0.420, -1.750, 'rs', ms=9, zorder=5, mec='k', mew=0.8)
        wl = np.linspace(-2.5, 0.3, 100)
        ax.plot(wl, -wl, 'k--', alpha=0.3, lw=1)
        ax.fill_between(wl, -wl, 2.5, color='grey', alpha=0.15)
        ax.set_xlabel(r'$w_0$')
        ax.set_ylabel(r'$w_a$')
        ax.set_xlim(-2.5, 0.3)
        ax.set_ylim(-5.0, 2.5)

    handles = [
        Line2D([0], [0], marker='*', color='k', ls='', ms=12,
               label=r'$\Lambda$CDM'),
        Line2D([0], [0], marker='o', color='m', ls='', ms=8, mec='k',
               label=r'CMB+PP\&SH0ES'),
        Line2D([0], [0], marker='D', color='m', ls='', ms=6, mec='k',
               label=r'CMB+PP\&SH0ES+BAOtr'),
        Line2D([0], [0], marker='^', color='c', ls='', ms=8, mec='k',
               label='CMB+SDSS'),
        Line2D([0], [0], marker='s', color='r', ls='', ms=8, mec='k',
               label='CMB+DESI'),
    ]
    ax1.legend(handles=handles, fontsize=7.5, loc='upper left',
               framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig3_chi2_surfaces.pdf', bbox_inches='tight')
    print("  Saved fig3_chi2_surfaces.pdf")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 4: DESI vs SDSS (Method B)
# ══════════════════════════════════════════════════════════════════
def figure4():
    print("\nFigure 4: DESI vs SDSS...")
    rB_desi = method_alpha(-1.0, 0.0, DESI_DATA)
    rB_sdss = method_alpha(-1.0, 0.0, SDSS_DATA)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(Z_BAOTR))
    w = 0.35

    ax.bar(x - w/2, rB_desi['Ti'], w, color='steelblue', alpha=0.85,
           label=(f'DESI anchors ($\\chi^2_B={rB_desi["chi2"]:.1f}$,'
                  f' $H_0={rB_desi["H0"]:.1f}$)'))
    ax.bar(x + w/2, rB_sdss['Ti'], w, color='indianred', alpha=0.75,
           label=(f'SDSS anchors ($\\chi^2_B={rB_sdss["chi2"]:.1f}$,'
                  f' $H_0={rB_sdss["H0"]:.1f}$)'))

    ax.axhline(0, color='k', lw=0.5)
    ax.axhspan(-1, 1, color='green', alpha=0.08)
    ax.set_ylabel(r'Tension $T_i$ [$\sigma$]')
    ax.set_ylim(-5, 1.5)
    ax.set_xticks(x)
    labels = [f'{z:.2f}' if z < 1 else f'{z:.3f}' for z in Z_BAOTR]
    labels[-1] = '2.225'
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Redshift $z$')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=10)
    ax.set_title(r'Method B ($\alpha$-interpolation): '
                 r'$\Lambda$CDM, DESI-optimized', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig4_sdss_comparison.pdf', bbox_inches='tight')
    print(f"  Saved fig4_sdss_comparison.pdf "
          f"(DESI={rB_desi['chi2']:.1f}, SDSS={rB_sdss['chi2']:.1f})")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure 5: Residuals relative to Planck LCDM
# ══════════════════════════════════════════════════════════════════
def figure5():
    print("\nFigure 5: residuals...")
    dash = (8, 4)

    # Parameters
    Om_pl, H0_pl = OM_PLANCK, H0_PLANCK
    Om_lcdm_opt, H0_lcdm_opt = RES_A['LCDM']['Om'], RES_A['LCDM']['H0']
    Om_cpl, H0_cpl = RES_A['CMB+DESI']['Om'], RES_A['CMB+DESI']['H0']

    # Chi2 values
    c2d_pl = chi2_3d_func(Om_pl, H0_pl, -1., 0.)
    c2b_pl = chi2_baotr(Om_pl, H0_pl, -1., 0.)['chi2']
    c2d_cpl = RES_A['CMB+DESI']['chi2_3d']
    c2b_cpl = RES_A['CMB+DESI']['chi2_baotr']
    c2B_lcdm = RES_B['LCDM']['chi2']
    c2B_cpl = RES_B['CMB+DESI']['chi2']

    # Dense redshift grid
    zd = np.linspace(0.05, 2.5, 300)
    zd = np.sort(np.unique(np.concatenate(
        [zd, np.array(sorted(DESI_DATA.keys())), Z_BAOTR])))

    # Planck LCDM reference
    dm_pl = np.array([DM_over_rd(z, Om_pl, H0_pl) for z in zd])
    dm_pl_baotr = np.array([DM_over_rd(z, Om_pl, H0_pl) for z in Z_BAOTR])

    # CPL raw curve
    dm_cpl = np.array([DM_over_rd(z, Om_cpl, H0_cpl, -0.42, -1.75)
                        for z in zd])

    # Alpha-warped curves
    aw_lcdm = alpha_warped_dense(Om_lcdm_opt, H0_lcdm_opt, -1., 0., zd)
    aw_cpl = alpha_warped_dense(Om_cpl, H0_cpl, -0.42, -1.75, zd)

    # DESI anchor points for plotting
    zp, dp, sp, dm_pl_d = [], [], [], []
    for z, (tp, v, sg) in sorted(DESI_DATA.items()):
        if tp == 'DM':
            zp.append(z); dp.append(v); sp.append(sg)
        else:
            dh = DH_over_rd(z, Om_pl, H0_pl)
            dm_v = np.sqrt(v**3 / (z * dh))
            sg_v = 1.5 * (dm_v / v) * sg
            zp.append(z); dp.append(dm_v); sp.append(sg_v)
        dm_pl_d.append(DM_over_rd(z, Om_pl, H0_pl))
    zp = np.array(zp); dp = np.array(dp)
    sp = np.array(sp); dm_pl_d = np.array(dm_pl_d)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Method A
    ax1.axhline(0, color='k', lw=2, label=(
        r'Planck $\Lambda$CDM ($\Omega_m\!=\!0.315$, $H_0\!=\!67.4$)'
        f'\n  ($\\chi^2_{{\\rm BAOtr}}={c2b_pl:.0f}$,'
        f' $\\chi^2_{{\\rm DESI}}={c2d_pl:.0f}$)'))
    ax1.plot(zd, dm_cpl - dm_pl, color='blue', lw=2, dashes=dash,
        label=(r'CPL DESI-opt. ($\Omega_m\!=\!0.35$, $H_0\!=\!63.8$)'
               f'\n  ($\\chi^2_{{\\rm BAOtr}}={c2b_cpl:.0f}$,'
               f' $\\chi^2_{{\\rm DESI}}={c2d_cpl:.0f}$)'))
    ax1.errorbar(zp, dp - dm_pl_d, yerr=sp, fmt='ro', ms=7,
                 capsize=3, zorder=5, label='DESI data')
    ax1.errorbar(Z_BAOTR, DM_BAOTR - dm_pl_baotr, yerr=SIG_BAOTR,
                 fmt='gs', ms=5, capsize=2, alpha=0.7,
                 label='BAOtr data', zorder=4)
    ax1.axhspan(-0.5, 0.5, color='grey', alpha=0.06)
    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel(r'$\Delta(D_{\rm M}/r_{\rm d})$ rel. to Planck $\Lambda$CDM')
    ax1.set_title('Method A: direct model comparison', fontsize=12)
    ax1.legend(fontsize=8, loc='lower left', framealpha=0.95)
    ax1.set_xlim(0, 2.55); ax1.set_ylim(-7, 3)
    ax1.axvspan(0, 0.295, color='red', alpha=0.04)

    # Right: Method B
    ax2.axhline(0, color='grey', lw=0.8, ls='-')
    ax2.plot([], [], color='grey', lw=0.8, ls='-',
             label=r'Planck $\Lambda$CDM (reference)')
    ax2.plot(zd, aw_lcdm - dm_pl, 'k-', lw=2,
        label=(r'$\Lambda$CDM DESI-opt. ($\alpha$-warped)'
               f'\n  ($\\chi^2_B={c2B_lcdm:.0f}$)'))
    ax2.plot(zd, aw_cpl - dm_pl, color='blue', lw=2, dashes=dash,
        label=(r'CPL DESI-opt. ($\alpha$-warped)'
               f'\n  ($\\chi^2_B={c2B_cpl:.0f}$)'))
    ax2.errorbar(zp, dp - dm_pl_d, yerr=sp, fmt='ro', ms=7,
                 capsize=3, zorder=5, label='DESI data')
    ax2.errorbar(Z_BAOTR, DM_BAOTR - dm_pl_baotr, yerr=SIG_BAOTR,
                 fmt='gs', ms=5, capsize=2, alpha=0.7,
                 label='BAOtr data', zorder=4)
    ax2.axhspan(-0.5, 0.5, color='grey', alpha=0.06)
    ax2.set_xlabel('Redshift $z$')
    ax2.set_ylabel(r'$\Delta(D_{\rm M}/r_{\rm d})$ rel. to Planck $\Lambda$CDM')
    ax2.set_title(r'Method B: $\alpha$-warped predictions', fontsize=12)
    ax2.legend(fontsize=8, loc='lower left', framealpha=0.95)
    ax2.set_xlim(0, 2.55); ax2.set_ylim(-7, 3)
    ax2.axvspan(0, 0.295, color='red', alpha=0.04)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig5_methods_comparison.pdf', bbox_inches='tight')
    print("  Saved fig5_methods_comparison.pdf")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    figure1()
    run_grid_scan()
    figure2()
    figure3()
    figure4()
    figure5()
    print("\nAll figures generated.")