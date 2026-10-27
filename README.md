<!--
SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# A Computational Soap Box Slide

[![pytest](https://github.com/molmod/soapboxslide/actions/workflows/pytest.yaml/badge.svg)](https://github.com/molmod/soapboxslide/actions/workflows/pytest.yaml)
[![release](https://github.com/molmod/soapboxslide/actions/workflows/release.yaml/badge.svg)](https://github.com/molmod/soapboxslide/actions/workflows/release.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/molmod/soapboxslide/main.svg)](https://results.pre-commit.ci/latest/github/molmod/soapboxslide/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI Version](https://img.shields.io/pypi/v/soapboxslide)](https://pypi.org/project/soapboxslide/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/soapboxslide)](https://pypi.org/project/soapboxslide/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/soapboxslide)](https://pypistats.org/packages/soapboxslide)
[![GPL-3 License](https://img.shields.io/github/license/molmod/soapboxslide)](https://github.com/molmod/soapboxslide/blob/main/LICENSE)


This Python package is written by Toon Verstraelen for students of the course
["Scientific Computing for Physicists" (C004649)](https://studiekiezer.ugent.be/2026/studiefiche/en/C004649)
of the [B.Sc. Physics and Astronomy](https://studiekiezer.ugent.be/2026/bachelor-of-science-in-physics-and-astronomy)
at [Ghent University](https://www.ugent.be/).
It is distributed under the conditions of the [GPLv3 license](https://github.com/molmod/soapboxslide/blob/main/LICENSE).

## Quick Start

You can install the `soapboxslide` Python module with:

```bash
pip install soapboxslide
```

This module implements a 2D surface that resembles a curved slide, suitable for computational soapbox racing.
To give you a quick idea, the following figure is created with the [`plot.py`](https://github.com/molmod/soapboxslide/blob/main/plot.py) script in this repository, and visualizes the two slides included in the package, `boxcar_blitz` and `brutal_bends`:

![slides.jpg](https://raw.githubusercontent.com/molmod/soapboxslide/main/slides.jpg)

Your eyes may need some time to adapt to the correct depth perception: black ridges are high-altitude separations between the colored valleys.

Note that the starting point and the finish are denoted as green and red circles, respectively.
All other circles are waypoints that must be visited on the way down.

## Concept

The idea is for students to use the surface implemented in `soapboxslide` to simulate the dynamics of a particle (or a connected set of particles) sliding down.
In addition to correctly implementing the dynamics, students are instructed to find the physical parameters for which the point mass(es) reach the final target as quickly as possible, without missing any of the intermediate waypoints.

Two classes of physical models can be considered:

1. The most convenient is to assume a model of point particles strictly bound to the surface with holonomic constraints.
In this case, equations of motion can be derived using a Lagrangian, possibly with a generalized force to include non-conservative friction forces.

2. A more challenging scenario (not used for now, but closer in spirit to real soap box races) is to impose inequality constraints, allowing particles to fly over the surface.

## `Slide` Class Usage

You can load one of the slides included in the package by name
and calculate slide properties at a given point, e.g. $x=5$ and $y=38$, as follows:

```python
import numpy as np

from soapboxslide import Slide

slide = Slide.from_name("boxcar_blitz")
print(slide(np.array([5.0, 38.0])))
```

This will show three results:

```text
(array(2.04001764), array(0.20172108), array(14.75990762))
```

These three values have the following meaning (in meter):

1. The progress along the track.
2. The orthogonal(ish) deviation from the bottom of the track.
3. The altitude of the track.

The included slides are `"boxcar_blitz"` and `"brutal_bends"`.
Because they are part of the package, everyone using the same version of `soapboxslide` gets exactly the same slides.
Their definitions are [TOML](https://toml.io/en/) files in the
[`soapboxslide/`](https://github.com/molmod/soapboxslide/tree/main/soapboxslide) directory of the repository.
You can also design your own slide in a TOML file with the same structure, and load it with `Slide.from_file("my_slide.toml")`.

The `slide()` function is fully vectorized: one may also provide an array of points to calculate many altitudes efficiently.
(This is useful for plotting.)

The `slide()` function optionally supports alternative NumPy wrappers, such as those of [JAX](https://docs.jax.dev/) and its predecessor [Autograd](https://github.com/HIPS/autograd).
This allows for an efficient evaluation of analytical partial derivatives of the altitude.
These libraries are not installed by default, but can be included with
`pip install soapboxslide[jax]` or `pip install soapboxslide[autograd]`.
For example, a vectorized calculation of many gradients is implemented as follows:

```python
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from soapboxslide import Slide

# We recommend double precision:
jax.config.update("jax_enable_x64", True)

# Define the slide and its gradient
slide = Slide.from_name("boxcar_blitz")
alt_grad = jax.jit(jax.vmap(jax.grad(
    lambda pos: slide(pos, npw=jnp)[2]
)))

# Compute the gradient at several points
pos = np.array([
    [5.0, 38.0],
    [12.1, 7.3],
    [10.7, 25.5],
])
print(alt_grad(pos))
```

This will show three gradients, one for each row of `pos`:

```text
[[ 0.21286679  0.10150386]
 [-1.86091385  1.80566055]
 [-2.18744618 -2.91699718]]
```

In addition, a `Slide` instance has the following attributes:

- `width`: the width of the slide arena in meter.
- `height`: the height of the slide arena in meter.
- `waypoints`: A 2D NumPy array with three columns (and at least two rows)
  defining the shape and altitude of the slide track.
  These are also intended as points that must be visited by a particle or a system sliding down,
  to ensure that it followed a legitimate trajectory.
- `target_radius`: a required proximity between a particle (or the center of mass of a system of particles) and a waypoint, to mark this waypoint as properly visited.
  The distance is only measured in the $xy$-plane.

Note that the above surface plots show dotted circles centered on the waypoints, whose radius is the target radius.

The `Slide` class also features two potentially useful methods:

- `Slide.get_hits` can be used to check which waypoints were reached by a given trajectory.
- `Slide.plot` can be used to prepare a drawing of the slide on which one can overlay additional results.
  It supports several keyword arguments to customize the surface.

This is a minimal example using the `Slide.plot` method, to which one can easily add more matplotlib code:

```python
import matplotlib.pyplot as plt

from soapboxslide import Slide

# Prepare plot with slide surface
slide = Slide.from_name("boxcar_blitz")
fig, ax = plt.subplots(figsize=(8, 7))
slide.plot(fig, ax)

# Customize plot
ax.set(title="You can put here whatever you want")
ax.plot([5.0, 20.0, 9.0], [36.0, 1.0, 5.0], "g-", lw=4)

# Save
fig.savefig("myplot.jpg")
```

Consult the docstrings for more information on how to use the `get_hits` and `plot` methods.

## Storing and Sharing Trajectory Data

The `soapboxslide` module also implements a `Trajectory` class for storing the results of a numerical integration of one or more particles sliding over the surface.
This class performs an initial validation of the trajectory data upon construction.
It also features `to_file` and `from_file` methods with which trajectories can be dumped to and loaded from NPZ files.
This is useful if you want to share your trajectory with someone, e.g. for review, and to implement separate computation and visualization scripts (or notebooks).

To use the `Trajectory` class, create an instance as follows after completing the numerical integration of the equations of motion:

```python
traj = Trajectory(
    time=...,        # NDArray[float], shape = (ntime, )
    mass=...,        # NDArray[float], shape = (npoint, )
    gamma=...,       # NDArray[float], shape = (npoint, )
    pos=...,         # NDArray[float], shape = (ntime, npoint, 3), (x, y and z coordinates)
    vel=...,         # NDArray[float], shape = (ntime, npoint, 3), (x, y and z coordinates)
    grad=...,        # NDArray[float], shape = (ntime, npoint, 2), (h_x and h_y derivatives)
    hess=...,        # NDArray[float], shape = (ntime, npoint, 3), (h_xx, h_xy and h_yy derivatives)
    spring_idx=...,  # Optional, NDArray[int], shape = (nspring, 2), (each row is a pair of point indexes)
    spring_par=...,  # Optional, NDArray[float], shape = (nspring, 3), (force constant, rest length, damping coeff)
    end_state=...,   # One of the EndState members
    stop_time=...,   # Optional, time that last target was reached
    stop_pos=...,    # Optional, NDArray[float], shape = (npoint, 3), positions at stop time
    stop_vel=...,    # Optional, NDArray[float], shape = (npoint, 3), velocities at stop time
)
traj.to_file("traj.npz")
```

In this Python snippet, you need to replace all triple dots by arrays you defined or computed.
Some guidelines:

- If there are no springs, you can omit `spring_idx` and `spring_par`.
- If there is just one particle, several arrays will have a size 1 axis, i.e. `npoint=1`.
- You can first store all fields in a dictionary, e.g. `data = {"time": ..., ...}` and optionally add the `stop_*` fields to the dictionary. The trajectory is then created with `Trajectory(**data)`.
- The end state can be specified to indicate when the numerical integration has ended, using one of the `EndState` instances defined in the `soapboxslide` module:
  - `EndState.STOP`: last target was reached in time.
  - `EndState.CRASH`: the numerical integration failed (e.g. due to Inf or NaN results).
  - `EndState.FAR`: the particle moved more than `FAR_MARGIN` (5 m) out of the arena.
  - `EndState.TIMEOUT`: the maximum time `MAX_TIME` (60 s) was reached.

  Import these constants, e.g. `from soapboxslide import FAR_MARGIN, MAX_TIME`,
  instead of hard-coding the values, so that all simulations use the same limits.
- When creating a `Trajectory` instance, you will get a `TypeError` or `ValueError` if some of the data do not meet basic expectations.

The meaning of and requirements for all attributes are further elaborated in the docstrings in [`soapboxslide/__init__.py`](https://github.com/molmod/soapboxslide/blob/main/soapboxslide/__init__.py).
