<!--
SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based Versioning](https://jacobtomlinson.dev/effver/).
(Changes to features documented as "experimental" will not increment macro and meso version numbers.)

## [Unreleased]

Improved graphics and a corrected sampling of the slide track.
The slide surfaces change slightly, so this release gets a meso version bump (1.1.0).

### Added

- The slides `boxcar_blitz` and `brutal_bends` are now included in the package
  and can be loaded with `Slide.from_name("boxcar_blitz")`, without downloading TOML files.
  This guarantees that everyone using the same version of `soapboxslide` gets the same slides.
  To make this possible, `soapboxslide` is now a package instead of a single module.
  `Slide.from_file` remains available for loading your own slides.

### Changed

- Corrected the curvature estimate used to sample the slide track,
  and the last control point now coincides with the finish.
  This slightly changes the slide surfaces (up to ~0.7 m in altitude near the track),
  so results from earlier versions are not exactly reproducible.
- `Slide.get_hits` also accepts lists (or other array-like inputs) for `masses`, `points` and `stop`.
- The constants `MAX_TIME` (60 s) and `FAR_MARGIN` (5 m) are exported and documented,
  so that all simulations use the same limits for `EndState.TIMEOUT` and `EndState.FAR`.
- The `Trajectory` time validator now reports "Time must not decrease.",
  matching the check, which allows equal consecutive times.
- Removed the unused `cattrs` dependency.
- Optional dependencies `jax` and `autograd`, installed with e.g. `pip install soapboxslide[jax]`.
  Unit tests now verify that the slide works with plain NumPy, JAX and autograd,
  including first and second derivatives computed with automatic differentiation.
- Improved package metadata (keywords, classifiers, project URLs)
  and made the links in the README file absolute, so they also work on PyPI.

### Fixed

- The plotting code has been tweaked to make the depth perception more intuitive.
- Altitude colors in `Slide.plot` are now aligned with the plotted coordinates.
- Evaluating the slide far away from the track (more than ~50 m)
  no longer returns NaN values.
- `Trajectory.stop_time` is now always a `float` (or `None`),
  also after loading a trajectory with `Trajectory.from_file`.
- Typos in docstrings, error messages and the README file.
- Docstrings and README now match the code more closely.

## [1.0.2] - 2025-10-12

Bug fixes.

### Fixed

- The `Slide.get_hits` method now measures the distance from the center of mass to the waypoints.
- Document `Slide.plot` and `Slide.get_hits` in the README file.

## [1.0.1] - 2025-10-12

Small improvements.

### Changed

- Improved README.
- Improved clarity of first and last target.
- Simplify altitude colormap.
- Improve the brutal bends slide.

## [1.0.0] - 2025-10-11

Initial release, just to get things started.

[Unreleased]: https://github.com/molmod/soapboxslide
[1.0.2]: https://github.com/molmod/soapboxslide/releases/tag/v1.0.2
[1.0.1]: https://github.com/molmod/soapboxslide/releases/tag/v1.0.1
[1.0.0]: https://github.com/molmod/soapboxslide/releases/tag/v1.0.0
