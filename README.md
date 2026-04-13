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
| `generate_tables.py` | Produces all numerical tables of the paper. |
| `generate_figures.py` | Produces all figures of the paper. |

## Reproducibility

| Command | Output | Paper element |
|---------|--------|---------------|
| `python generate_tables.py` | stdout | Table 1: Overcorrection from rescaling |
| | | Table 5: CMB-consistent parameters |
| | | Table 6: Per-point baseline tension |
| | | Table 7: χ² summary for all models |
| | | Table 8: Sensitivity to extrapolation and BGS anchor |
| `python generate_figures.py` | `figures/fig1_tension_bars.pdf` | Figure 1: Per-point tension bars |
| | `figures/fig2_tradeoff.pdf` | Figure 2: χ²_DESI vs χ²_BAOtr trade-off |
| | `figures/fig3_chi2_surfaces.pdf` | Figure 3: χ² surfaces in (w₀, wₐ) plane |
| | `figures/fig4_sdss_comparison.pdf` | Figure 4: DESI vs SDSS comparison |
| | `figures/fig5_methods_comparison.pdf` | Figure 5: Method A vs Method B residuals |

Tables 2–4 (DESI DR2 data, SDSS-IV data, BAOtr compilation) are observational data tables reproduced directly from the cited references.

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
- **BAOtr compilation**: Nunes et al. (2020), [arXiv:2008.13075](https://arxiv.org/abs/2008.13075); individual measurements from Carvalho et al. (2016, 2020), de Carvalho et al. (2018, 2021), Alcaniz et al. (2017)
- **CPL posteriors**: Xu et al. (2026)
- **Planck 2018**: Aghanim et al. (2020), [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)

## License

MIT
