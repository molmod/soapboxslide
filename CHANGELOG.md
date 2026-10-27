<!--
SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based Versioning](https://jacobtomlinson.dev/effver/).
(Changes to features documented as "experimental" will not increment macro and meso version numbers.)

## [Unreleased]

Improved graphics.

### Changed

- `Slide.get_hits` also accepts lists (or other array-like inputs) for `masses`, `points` and `stop`.
- The constants `MAX_TIME` (60 s) and `FAR_MARGIN` (5 m) are exported and documented,
  so that all simulations use the same limits for `EndState.TIMEOUT` and `EndState.FAR`.
- The `Trajectory` time validator now reports "Time must not decrease.",
  matching the check, which allows equal consecutive times.

### Fixed

- The plotting code has been tweaked to make the depth perception more intuitive.
- Altitude colors in `Slide.plot` are now aligned with the plotted coordinates.
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
