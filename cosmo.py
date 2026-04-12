"""
cosmo.py — Core cosmology library for the BAOtr-DESI tension analysis.

Implements:
  - Flat CPL cosmology with radiation
  - CMB theta* constraint: H0(Omega_m, w0, wa)
  - DESI-optimized parameter determination
  - Method A: direct model comparison
  - Method B: alpha-interpolation
  - Data: DESI DR2, SDSS-IV, BAOtr compilation

Reference: Pantos & Perivolaropoulos (2026)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# Physical constants and Planck 2018 reference values
# ══════════════════════════════════════════════════════════════════
C_KMPS = 299792.458          # Speed of light [km/s]
H0_PLANCK = 67.36            # Planck 2018 H0 [km/s/Mpc]
RD = 147.09                  # Sound horizon at drag epoch [Mpc]
Z_STAR = 1089.92             # Redshift of last scattering
Z_EQ = 3387.0                # Matter-radiation equality redshift
OM_PLANCK = 0.3153           # Planck 2018 Omega_m


# ══════════════════════════════════════════════════════════════════
# Hubble rate
# ══════════════════════════════════════════════════════════════════
def E_z(z, Om, w0=-1.0, wa=0.0):
    """
    Normalised Hubble rate E(z) = H(z)/H0 for flat CPL + radiation.

    Parameters
    ----------
    z : float
        Redshift.
    Om : float
        Present-day matter density parameter.
    w0, wa : float
        CPL dark energy equation of state parameters.

    Returns
    -------
    float
        E(z).
    """
    Or = Om / (1.0 + Z_EQ)
    ODE = 1.0 - Om - Or
    zp1 = 1.0 + z
    if ODE <= 0 or zp1 <= 0:
        return 1e10
    try:
        de = ODE * zp1**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / zp1)
    except (OverflowError, FloatingPointError):
        return 1e10
    val = Om * zp1**3 + Or * zp1**4 + de
    return np.sqrt(val) if val > 0 else 1e10


# ══════════════════════════════════════════════════════════════════
# Distance integrals
# ══════════════════════════════════════════════════════════════════
def _integral_Ez(z, Om, w0=-1.0, wa=0.0):
    """Dimensionless integral int_0^z dz'/E(z')."""
    result, _ = quad(lambda zp: 1.0 / E_z(zp, Om, w0, wa),
                     0, z, limit=300)
    return result


def _integral_Ez_zstar(Om, w0=-1.0, wa=0.0):
    """Dimensionless integral to z_* (last scattering)."""
    result, _ = quad(lambda zp: 1.0 / E_z(zp, Om, w0, wa),
                     0, Z_STAR, limit=500)
    return result


# Precompute LCDM reference integral
_I_LCDM_ZSTAR = _integral_Ez_zstar(OM_PLANCK, -1.0, 0.0)


def DM_over_rd(z, Om, H0, w0=-1.0, wa=0.0):
    """Comoving distance D_M(z) / r_d."""
    return (C_KMPS / H0 / RD) * _integral_Ez(z, Om, w0, wa)


def DH_over_rd(z, Om, H0, w0=-1.0, wa=0.0):
    """Hubble distance D_H(z) / r_d = c / [H(z) * r_d]."""
    return C_KMPS / (H0 * E_z(z, Om, w0, wa) * RD)


def DV_over_rd(z, Om, H0, w0=-1.0, wa=0.0):
    """Volume-averaged distance D_V(z) / r_d."""
    dm = DM_over_rd(z, Om, H0, w0, wa)
    dh = DH_over_rd(z, Om, H0, w0, wa)
    return (z * dm**2 * dh) ** (1.0 / 3.0)


# ══════════════════════════════════════════════════════════════════
# CMB theta* constraint
# ══════════════════════════════════════════════════════════════════
def H0_from_theta_star(Om, w0=-1.0, wa=0.0):
    """
    Compute H0 such that D_M(z_*) = D_M^{LCDM}(z_*), i.e. the
    CMB angular scale theta_* is preserved.

    Calibrated: returns H0 = 67.36 at (Om=0.3153, w0=-1, wa=0).

    Parameters
    ----------
    Om : float
        Matter density parameter.
    w0, wa : float
        CPL parameters.

    Returns
    -------
    float
        H0 in km/s/Mpc.
    """
    I_star = _integral_Ez_zstar(Om, w0, wa)
    return H0_PLANCK * I_star / _I_LCDM_ZSTAR


# ══════════════════════════════════════════════════════════════════
# Observational data
# ══════════════════════════════════════════════════════════════════

# DESI DR2 BAO measurements (AbdulKarim et al. 2025)
# Format: {z_eff: (observable_type, value, sigma)}
# 'DV' = D_V/r_d (isotropic); 'DM' = D_M/r_d (anisotropic)
DESI_DATA = {
    0.295: ('DV', 7.942, 0.075),
    0.510: ('DM', 13.588, 0.167),
    0.706: ('DM', 17.351, 0.177),
    0.934: ('DM', 21.576, 0.152),
    1.321: ('DM', 27.601, 0.318),
    1.484: ('DM', 30.512, 0.760),
    2.330: ('DM', 38.988, 0.531),
}

# SDSS-IV consensus BAO (Alam et al. 2021)
SDSS_DATA = {
    0.380: ('DM', 10.23, 0.17),
    0.510: ('DM', 13.36, 0.21),
    0.700: ('DM', 17.86, 0.33),
    1.480: ('DM', 30.69, 0.80),
    2.334: ('DM', 37.30, 1.70),
}

# BAOtr compilation (Nunes et al. 2020; de Carvalho et al. 2016, 2018, 2020, 2021;
# Alcaniz et al. 2017)
# Columns: z, theta_BAO [deg], sigma_theta [deg]
_BAOTR_RAW = np.array([
    [0.110, 19.80, 3.26],
    [0.235,  9.06, 0.23],
    [0.365,  6.33, 0.22],
    [0.450,  4.77, 0.17],
    [0.470,  5.02, 0.25],
    [0.490,  4.99, 0.21],
    [0.510,  4.81, 0.17],
    [0.530,  4.29, 0.30],
    [0.550,  4.25, 0.25],
    [0.570,  4.62, 0.40],
    [0.590,  4.37, 0.35],
    [0.610,  3.86, 0.33],
    [0.630,  3.88, 0.42],
    [0.650,  3.54, 0.17],
    [2.225,  1.77, 0.31],
])

Z_BAOTR = _BAOTR_RAW[:, 0]
DM_BAOTR = 180.0 / (np.pi * _BAOTR_RAW[:, 1])        # D_M/r_d = 180/(pi*theta)
SIG_BAOTR = DM_BAOTR * (_BAOTR_RAW[:, 2] / _BAOTR_RAW[:, 1])  # error propagation

# Published CPL posterior centres from Xu et al. (2026)
PUBLISHED_MODELS = [
    ('LCDM',                -1.000,  0.000),
    ('CMB+PP&SH0ES',        -0.694, -1.700),
    ('CMB+PP&SH0ES+BAOtr',  -0.660, -1.910),
    ('CMB+SDSS',            -0.480, -1.510),
    ('CMB+DESI',            -0.420, -1.750),
]


# ══════════════════════════════════════════════════════════════════
# Chi-squared functions
# ══════════════════════════════════════════════════════════════════
def chi2_3d(Om, H0, w0, wa, anchor_data=None):
    """
    Chi-squared of the model against 3D BAO data.

    Parameters
    ----------
    Om, H0, w0, wa : float
        Model parameters.
    anchor_data : dict, optional
        BAO anchor dataset (default: DESI_DATA).

    Returns
    -------
    float
        chi2 value.
    """
    if anchor_data is None:
        anchor_data = DESI_DATA
    c2 = 0.0
    for z, (obs_type, val, sig) in anchor_data.items():
        if obs_type == 'DV':
            model_val = DV_over_rd(z, Om, H0, w0, wa)
        else:
            model_val = DM_over_rd(z, Om, H0, w0, wa)
        c2 += ((val - model_val) / sig) ** 2
    return c2


def chi2_baotr(Om, H0, w0, wa):
    """
    Chi-squared of the model against BAOtr data.

    Returns
    -------
    dict with keys:
        'chi2': total chi2
        'Ti': per-point normalised tensions
        'dm_model': model D_M/r_d at BAOtr redshifts
    """
    dm_model = np.array([DM_over_rd(z, Om, H0, w0, wa) for z in Z_BAOTR])
    Ti = (DM_BAOTR - dm_model) / SIG_BAOTR
    return {'chi2': np.sum(Ti**2), 'Ti': Ti, 'dm_model': dm_model}


# ══════════════════════════════════════════════════════════════════
# DESI-optimized parameter determination
# ══════════════════════════════════════════════════════════════════
def best_fit_desi(w0, wa, anchor_data=None, Om_range=(0.15, 0.55)):
    """
    At fixed (w0, wa), find Omega_m that minimises chi2 against the
    3D BAO data, with H0 determined by the theta* constraint.

    Parameters
    ----------
    w0, wa : float
        CPL parameters.
    anchor_data : dict, optional
        BAO anchor dataset (default: DESI_DATA).
    Om_range : tuple
        Search range for Omega_m.

    Returns
    -------
    dict or None
        Contains 'Om', 'H0', 'chi2_3d', 'chi2_baotr', 'Ti_baotr',
        'dm_model_baotr'. Returns None if no solution found.
    """
    if anchor_data is None:
        anchor_data = DESI_DATA

    def objective(Om):
        H0 = H0_from_theta_star(Om, w0, wa)
        if H0 < 40 or H0 > 100:
            return 1e6
        return chi2_3d(Om, H0, w0, wa, anchor_data)

    try:
        result = minimize_scalar(objective, bounds=Om_range,
                                 method='bounded',
                                 options={'xatol': 1e-5})
        Om = result.x
        H0 = H0_from_theta_star(Om, w0, wa)
        c2_3d = result.fun
        baotr = chi2_baotr(Om, H0, w0, wa)
        return {
            'Om': Om, 'H0': H0,
            'chi2_3d': c2_3d,
            'chi2_baotr': baotr['chi2'],
            'Ti_baotr': baotr['Ti'],
            'dm_model_baotr': baotr['dm_model'],
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# Method B: alpha-interpolation
# ══════════════════════════════════════════════════════════════════
def method_alpha(w0, wa, anchor_data=None, extrap='constant'):
    """
    Alpha-interpolation method (Method B).

    Anchors the prediction to the 3D BAO data via dilation
    parameters alpha_j, then interpolates to BAOtr redshifts.

    Parameters
    ----------
    w0, wa : float
        CPL parameters.
    anchor_data : dict, optional
        BAO anchor dataset (default: DESI_DATA).
    extrap : str
        Extrapolation scheme below lowest anchor:
        'constant', 'model', or 'linear'.

    Returns
    -------
    dict or None
        Contains 'chi2', 'Ti', 'dm_pred', 'alpha_anc', 'Om', 'H0'.
    """
    if anchor_data is None:
        anchor_data = DESI_DATA

    r = best_fit_desi(w0, wa, anchor_data)
    if r is None:
        return None
    Om, H0 = r['Om'], r['H0']

    # Build anchor arrays (convert BGS D_V to D_M)
    z_anc, dm_data, sig_data = [], [], []
    for z, (obs_type, val, sig) in sorted(anchor_data.items()):
        if obs_type == 'DV':
            dh = DH_over_rd(z, Om, H0, w0, wa)
            dm_val = np.sqrt(val**3 / (z * dh))
            dm_sig = 1.5 * (dm_val / val) * sig
        else:
            dm_val, dm_sig = val, sig
        z_anc.append(z)
        dm_data.append(dm_val)
        sig_data.append(dm_sig)

    z_anc = np.array(z_anc)
    dm_data = np.array(dm_data)
    sig_data = np.array(sig_data)

    # Alpha at anchors
    dm_model_anc = np.array([DM_over_rd(z, Om, H0, w0, wa) for z in z_anc])
    alpha_anc = dm_data / dm_model_anc

    # Interpolate alpha to BAOtr redshifts
    u_anc = np.log(1.0 + z_anc)
    u_baotr = np.log(1.0 + Z_BAOTR)
    alpha_interp = np.empty(len(Z_BAOTR))

    for i, u in enumerate(u_baotr):
        if u <= u_anc[0]:
            if extrap == 'constant':
                alpha_interp[i] = alpha_anc[0]
            elif extrap == 'model':
                alpha_interp[i] = 1.0
            elif extrap == 'linear':
                slope = (alpha_anc[1] - alpha_anc[0]) / (u_anc[1] - u_anc[0])
                alpha_interp[i] = alpha_anc[0] + slope * (u - u_anc[0])
        elif u >= u_anc[-1]:
            alpha_interp[i] = alpha_anc[-1]
        else:
            k = np.searchsorted(u_anc, u) - 1
            f = (u - u_anc[k]) / (u_anc[k+1] - u_anc[k])
            alpha_interp[i] = alpha_anc[k] + f * (alpha_anc[k+1] - alpha_anc[k])

    # Predicted D_M/r_d at BAOtr redshifts
    dm_model_baotr = np.array([DM_over_rd(z, Om, H0, w0, wa) for z in Z_BAOTR])
    dm_pred = alpha_interp * dm_model_baotr

    # Prediction uncertainty (propagated from anchors)
    sig_pred = np.zeros(len(Z_BAOTR))
    for i, u in enumerate(u_baotr):
        if u <= u_anc[0]:
            sig_pred[i] = dm_pred[i] * sig_data[0] / dm_data[0]
        elif u >= u_anc[-1]:
            sig_pred[i] = dm_pred[i] * sig_data[-1] / dm_data[-1]
        else:
            k = np.searchsorted(u_anc, u) - 1
            f = (u - u_anc[k]) / (u_anc[k+1] - u_anc[k])
            frac = np.sqrt((1 - f)**2 * (sig_data[k] / dm_data[k])**2
                           + f**2 * (sig_data[k+1] / dm_data[k+1])**2)
            sig_pred[i] = dm_pred[i] * frac

    # Tension
    sig_tot = np.sqrt(SIG_BAOTR**2 + sig_pred**2)
    Ti = (DM_BAOTR - dm_pred) / sig_tot

    return {
        'chi2': np.sum(Ti**2),
        'Ti': Ti,
        'dm_pred': dm_pred,
        'alpha_anc': alpha_anc,
        'Om': Om,
        'H0': H0,
    }


# ══════════════════════════════════════════════════════════════════
# Alpha-warped prediction on a dense redshift grid (for plotting)
# ══════════════════════════════════════════════════════════════════
def alpha_warped_dense(Om, H0, w0, wa, z_dense, anchor_data=None):
    """
    Compute the alpha-warped D_M/r_d prediction on a dense z grid.
    Used for plotting smooth curves that pass through anchor data.

    Returns
    -------
    numpy.ndarray
        Alpha-warped D_M/r_d at each z in z_dense.
    """
    if anchor_data is None:
        anchor_data = DESI_DATA

    z_anc, dm_anc = [], []
    for z, (obs_type, val, sig) in sorted(anchor_data.items()):
        if obs_type == 'DV':
            dh = DH_over_rd(z, Om, H0, w0, wa)
            dm_val = np.sqrt(val**3 / (z * dh))
        else:
            dm_val = val
        z_anc.append(z)
        dm_anc.append(dm_val)

    z_anc = np.array(z_anc)
    dm_anc = np.array(dm_anc)
    dm_model_anc = np.array([DM_over_rd(z, Om, H0, w0, wa) for z in z_anc])
    alpha_anc = dm_anc / dm_model_anc

    u_anc = np.log(1.0 + z_anc)
    u_dense = np.log(1.0 + z_dense)
    alpha_dense = np.empty(len(z_dense))

    for i, u in enumerate(u_dense):
        if u <= u_anc[0]:
            alpha_dense[i] = alpha_anc[0]
        elif u >= u_anc[-1]:
            alpha_dense[i] = alpha_anc[-1]
        else:
            k = np.searchsorted(u_anc, u) - 1
            f = (u - u_anc[k]) / (u_anc[k+1] - u_anc[k])
            alpha_dense[i] = alpha_anc[k] + f * (alpha_anc[k+1] - alpha_anc[k])

    dm_model_dense = np.array([DM_over_rd(z, Om, H0, w0, wa) for z in z_dense])
    return alpha_dense * dm_model_dense