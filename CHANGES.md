# Changelog

## 1.2.0 (Unreleased)

Contributors to this version: Joseph Siddons (@jtsiddons), Richard Cornes (@rcornes), Steven Chan (@SCChan21).

### New features and enhancements

* Approximate normality transformation using the Box-Cox transformation, with
  function specifically for Weibull-distributed variables (such as wind speeds) (#138).

### Bug fixes

* Remove hard-coding of grid index name ("grid_idx") in `grid.Grid.map_observations` and
  `grid.map_to_grid`, dynamically set the name using the `grid_prefix` argument (#105).

### Documentation

* Update citation details, add `__bibtex__` for convenience (#118).

### Internal changes

* Adds a `Makefile` (#144).
* Build tool changed to `flit` (#144).

## 1.1.0 (2026-01-12)

Contributors to this version: Joseph Siddons (@jtsiddons, @josidd), Steven Chan (@SCChan21), Richard
Cornes (@rcornes)

### Announcements

* Now supports python 3.14 (#61).

### Breaking changes

* `stochastic.scipy_mv_normal_draw` renamed to `stochastic.draw_from_cov` (#74)

### New features and enhancements

* Can now compute ellipse covariance for 'unrotated' and 'isotropic' cases (#58)
* `grid.grid_from_resoluton` and `grid.Grid.from_resolution` have new `definition` argument
* Added `covariance_tools.validate_covariance` to validate shape and symmetry of covariance matrix,
  to within a tolerance (#74)
* `grid.grid_from_resoluton` and `grid.Grid.from_resolution` have new `definition` argument
  describing how the left boundary in each coordinate defines the grid, as the "left" edge of the
  grid in each direction (so that grid-boxes are defined by lower-bound + resolution / 2), or the
  "center" of the first grid-box (#33)
* New `grid.Grid` class that encapsulates the full process, including `distance_matrix`,
  `covariance_matrix`, and `kriging` methods. Also allows for _masking_ with the `add_mask` method,
  allowing for the kriging to be performed on a masked grid. Includes a `from_resolution`
  constructor method, replicating behaviour of `grid.grid_from_resoluton` (#33).
* Add `error_covariance.weighted_sum` function to compute weights for gridding using local inverse
  error correlation structure of grouped records (#42).
* Checks if matrix is symmetric in `stochastic.scipy_mv_normal_draw`; adds `sym_atol` argument for
  absolute tolerance in input covariance symmetry; if slightly asymmetric, use `(cov + cov.T) / 2`
  (#73)

### Bug fixes

* Fixed typo in `kriging.Kriging.constraint_mask` docstrings (#63)
* Return vector in `stochastic.scipy_mv_normal_draw` when back-up approach
  `stats.multivariate_normal.rvs` is used with `ndraws=1` (#73)

### Internal changes

* Clean-up misleading language in comments, warnings, parameter names (#82)
* Unit tests for `covariance_tools` and `variogram` modules now test that resulting matrices are
  symmetric (#59).

## 1.0.1 (2025-11-11)

Contributors to this version: Joseph Siddons (@jtsiddons, @josidd).

### Bug fixes

* Fix bug with `grid.cross_coords` that reversed coordinate names for transposed grid (#44).
* Fix bug with potential mixed duration units when computing mid-point of months (#44).

## 1.0.0 (2025-08-22)

Contributors to this version: Joseph Siddons (@josidd), Steven Chan (@stchan), Richard Cornes
(@ricorne).

### Announcements

* Released on PyPI
* License updated to Apache 2.0 (!77)
* Added components for constructing spatial covariances from ellipses following [Paciorek and
  Schervish (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2157553/pdf/nihms13857.pdf) and
  [Karspeck et al. (2012)](https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.900).

### Deprecations

* `kriging.kriging_simple` and `kriging.kriging_ordinary` are set to be deprecated in favour of
  `kriging.SimpleKriging` and `kriging.OrdinaryKriging` classes respectively. They will be removed
  in a future version (!62)

### Breaking changes

* `ellipse.covariance.EllipseCovarianceBuilder` argument `max_dist` is now optional and treated as
  infinite if not set (!85)
* `Kriging` class objects now have required `idx`, `obs`, and optional `error_cov` inputs, these
  inputs are no longer required for class methods (!75)
* `kriging.kriging` and `kriging.unmasked_kriging` wrapper functions are removed (!68)
* `covariance_tools.eof_chop` is removed. It does not return a valid covariance matrix (!68)
* Removed `variogram.LinearVariagram` and `variogram.PowerVariogram` (!69)
* `perturbation` module is renamed to `stochastic` (!65)
* All job-specific files are removed from the library (!56)

### New features and enhancements

* Updated docstrings for `glomar_gridding.covariance_tools.laloux_clip` (!90)
* Added additional example to the jupyter notebook (!89)
* Input `dtype`s are maintained through the `Kriging` classes (!87)
* Allow for option to select training data for ellipse parameter estimation using Euclidean degree
  distance, option to use Haversine distance as selection criteria is the default (!84)
* `NaN` values remaining in error covariance after filtering to observations and dropping `NaN`
  values on the diagonal are set to 0.0 in `glomar_gridding.kriging.Kriging` classes (!83)
* Added new functions `glomar_gridding.covariance_tools.laloux_clip` and
  `glomar_gridding.covariance_tools.explained_variance_clip` (!78)
* Add `simulated_obs` as attribute to `StochasticKriging` in `solve` method (!71)
* Added an example notebook (!46)
* Added documentation pdf (!46)
* Added `variogram.SphericalVariogram` (!69)
* `ellipse`, `ellipse_builder`, and `ellipse_covariance` are renamed to `ellipse.model`,
  `ellipse.estimate`, and `ellipse.covariance` respectively. Ellipse classes `EllipseModel`,
  `EllipseBuilder`, and `EllipseCovarianceBuilder` are available at the `ellipse` level (!66)
* Added a new class for performing a two-stage Kriging following [Morice et al.
  (2021)](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019JD032361) -
  `stochastic.StochasticKriging` (!65)
* Introduced new classes for Kriging - `kriging.SimpleKriging` and `kriging.OrdinaryKriging`,
  allowing for easy computation of uncertainty, and alpha values (!62)
* Add `covariance_tools` module for adjusting estimated covariance matrices to positive definite (!54)
* Improved performance of ellipse covariance with vectorised and batch-vectorised methods (!54)
* Add module for calculating spatial covariance matrix from ellipse parameters
  (`ellipse_covariance`) (!54)
* Add module for estimating ellipse parameters from observational datasets (`ellipse_builder`) (!54)
* Add module containing ellipse models (`ellipse`) (!54)
* Add new function to combine coordinates, for example for the index/coordinates for a distance
  matrix (!61)

### Bug fixes

* Accounted for `xarray.DataArray` in `variogram.SphericalVariogram` (!88)
* Corrected `glomar_gridding.covariance_tools.eigenvalue_clip` with `"Laloux"` method to compute new
  eigenvalues using correlation matrix rather than covariance matrix (!78)

### Internal changes

* Build method changed to [hatchling](https://hatch.pypa.io/1.9/config/build/)
* Additional unit test for ellipse-based covariance self-consistency (!80)
* Unit test for ellipse covariance from uniform parameters fixed, Chi squared test dropped in favour
  of correlation distance from https://doi.org/10.1109/VETECS.2005.1543265 (!80)
* Additional unit tests added (!68)
* Removed `requirements.txt` file. Dependencies are managed by `pyproject.toml` (!63)
* Added a GitLab runner pipeline (!64)

## 0.2.3 (2025-04-30)

Contributors to this version: Joseph Siddons (@josidd), Steven Chan (@stchan).

### New features and enhancements

* Add function to compute `constraint_mask` / alpha following Morice et al. (2021) (!58).

## 0.2.2 (2025-04-30)

Contributors to this version: Joseph Siddons (@josidd)

### New features and enhancements

* Added optional argument `mean` to `kriging.kriging_simple` (!47)

## 0.2.1 (2025-04-28)

Contributors to this version: Joseph Siddons (@josidd)

### New features and enhancements

* Add function to compute mid-point of a month (matching HadCRUT datetimes) (!48)
* `init_logging` now has a `level` argument (!50)
* Added script to combine LSAT and SST for HadCRUT reconstruction using weights file using polars to
  join (!40)
* Added `io.get_recurse` for scanning nested dictionaries by a key list (!38)

### Bug fixes

* Use "days since" as units for HadCRUT reconstruction outputs (!48)
* Correct local import of `noc_helpers` in `noc_runners` scripts (!45)

### Breaking changes

* `kriging.kriging_simple` and `kriging.kriging_ordinary` covariance arguments renamed (!49)
* `utils.get_git_commit` moved to `noc_helpers.get_git_commit`. Use of subprocess dropped (!44)
* Refactored HadCRUT runner script (!38)
    * Outputs are no-longer yearly files
    * Use xarray rather than open a netCDF file
    * Loop over all ensembles in script
    * Simplify arguments for script, now only require config, all parameters must be set in config
      file
    * Outputs include information for traceability (including git commit, user)

### Internal changes

* Add `__version__` (!55)
* `variogram.Variogram` is now an instance of `abc.ABC` abstract class, `.fit` is an abstract method (!51)
* Added changelog (!39)

## 0.2.0 (2025-02-18)

Contributors to this version: Joseph Siddons (@josidd), Steven Chan (@stchan), Agnieszka Faulkner
(@agfaul)

### Announcements

* Renamed library to `GloMarGridding`
* Significant re-factor of the entire library in merge request !30.

### New features and enhancements

* Added `perturbation` module for perturbing kriging output fields following Morice et al. (2021)
  (!30)
* Added `io` module for loading netCDF files, making use of format strings (!30)
* Computation of distance matrix from grid (!30)
* Added `noc_runners/noc_helpers` to contain shared job-specific functions (!30)
* Added `interpolation_covariance` module (!30)
* Added `climatology` module for joining climatology data to observational data (!30)
* Added `error_covariance` module for computing correlated, uncorrelated, and spatial components of
  error covariance matrices (functions moved from `observations`, `kriging` modules) (!30)
* Added `distances` module for computing haversine, euclidean distances for covariance matrices (!30)
* Added `mask` module for working with masks and grids, some functions moved from `observations` (!30)
* Added `grid` module for constructing, mapping to grids for outputs, allowing for consistent
  indexing (!30)

### Bug fixes

* Account for immutability of `np.diag` (!35)

### Internal changes

* Added `LICENSE` and `CONTRIBUTING` files (!30)
* Updated `README` to include set-up details (!30)
* Complete re-structure of library, modules now part of `glomar_gridding` module (!30)
* Add some unit tests for ordinary kriging (!30)
* Config files for main scripts now utilise yaml over ini (!30)
* Library dependencies can be managed by `pip` or `uv` (!30)
* Build of library now uses `pyproject.toml` (!30)
* Follow code standards using `ruff` for linting and formatting (!30)

### Breaking changes

* Remove specific `io` functions, added to `noc_runners/noc_helpers` (!30)
* Variogram functions re-factored to classes, class names follow CamelCase naming convention (!30)
* Re-name `covariance_variogram` -> `variogram`, remove unused functions (!30)
* Removal of `observations`, `observations_plus_qc`, `simple_plots`, `covariance`,
  `covariance_calculation` modules, code moved to other modules (!30)
* Modules moved to `glomar_gridding` (!30)
* Main scripts and config files now moved to `noc_runners` (!30)
* Use `polars` in place of `python` (!30)

## 0.1.0 (2024-12-09)

Contributors to this version: Agnieszka Faulkner (@agfaul), Steven Chan (@stchan), Joseph Siddons
(@josidd), Richard Cornes (@ricorne)

### New features and enhancements

* Allow computation of Euclidean (tunnel) distance (!26)
* Allow for removal of mean/median before kriging, adding back in after (!24)
* Added processing scripts for DCENT and HadCRUT (!20, !23, !33)
* Add function to correctly compute covariance from variogram
* Added processing scripts for TAO (!18)
* Add Matern Tau distance
* Add utils module
* Simplify adjustment of small negative eigenvalues (!29)
* Add metadata to output netCDF files (!13)
* Only perform one method of kriging in main scripts (!11)
* Allow for height adjustments to marine air temperature in mat main files
* Count number of observations by grid-box (!5)
* Add script for producing MAT datasets by month (!4, !16)

### Bug fixes

* Fix problems with longitude in DCENT processing (!34)
* Fix missing `remove_obs_mean` definition in DCENT processing (!31)
* Disable use of uncorrelated components in HadSST script (!36)
* Fix numerical issues when kriging with no error covariance (!28)
* Correct calculation of `dz` for ordinary kriging (!21)

### Internal changes

* Optimised reading of height adjustments by using polars and feather format (!12)
* Optimised computation of distance/weight components of error covariance (!1, !3)
* Use float32 for improved memory performance on JASMIN (!4)

### Breaking changes

* Simplified kriging functions by splitting out `kriging_ordinary` and `kriging_simple` (!27)
* Refactored code into python modules

## 0.0.1 (2023-08-16)

Contributors to this version: Agnieszka Faulkner (@agfaul)

### New features and enhancements

* Rewrote original Matlab code into python scripts for kriging
