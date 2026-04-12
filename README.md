# BAOtr–BAO 3D Tension

Analysis code for:

**"On the origin of the BAOtr - DESI tension"**
I. Pantos & L. Perivolaropoulos (2026)

## Overview

This repository contains the Python code to reproduce all figures and numerical tables in the paper. The analysis compares transversal (angular) BAO measurements (BAOtr) with 3D BAO measurements from DESI DR2 and SDSS-IV, testing whether the CPL dark-energy parametrisation can reconcile the two datasets.

## Files

| File | Description |
|------|-------------|
| `cosmo.py` | Core cosmology library: flat CPL cosmology with radiation, CMB θ* constraint, DESI-optimized parameter determination, Method A (direct comparison) and Method B (α-interpolation). Contains all observational data (DESI DR2, SDSS-IV, BAOtr). |
| `generate_tables.py` | Produces Tables 1, 5–8 of the paper: overcorrection diagnostics, CMB-consistent parameters, per-point tensions, χ² summary, and sensitivity tests. |
| `generate_figures.py` | Produces Figures 1–5 of the paper: tension bar charts, χ² trade-off plot, χ² surfaces in the (w₀, wₐ) plane, DESI vs SDSS comparison, and residual plots. |

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib

## Usage

```bash
# Generate all tables (prints to stdout)
python generate_tables.py

# Generate all figures (saved to figures/)
python generate_figures.py
```

## Data sources

- **DESI DR2 BAO**: Abdul-Karim et al. (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- **SDSS-IV consensus**: Alam et al. (2021), [arXiv:2007.08991](https://arxiv.org/abs/2007.08991)
- **BAOtr compilation**: Nunes et al. (2020), [arXiv:2008.13075](https://arxiv.org/abs/2008.13075)
- **CPL posteriors**: Xu et al. (2026)
- **Planck 2018**: Aghanim et al. (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)

## License

MIT
