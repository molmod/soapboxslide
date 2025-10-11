# A Computational Soap Box Slide

This Python package is written by Toon Verstraelen for students of the course
["Python for Scientists" (C004212)](http://studiekiezer.ugent.be/studiefiche/en/C004212/2025)
of the [B.Sc. Physics and Astronomy](https://studiekiezer.ugent.be/2026/bachelor-of-science-in-physics-and-astronomy)
at [Ghent University](https://www.ugent.be/).

## Quick Start

You can install the `soapboxslide` Python module with:

```bash
pip install soapboxslide
```

This module implements a 2D surface that resembles a curved slide, suitable for computational soapbox racing.
To give you a quick idea, the following figure is created with the [`plot.py`](plot.py) script in this repository,
and visualizes the slides [`boxcar_blitz.toml`](boxcar_blitz.toml) and [`brutal_bends.toml`](brutal_bends.toml) included in this repository:

![slides.jpg](slides.jpg)

Your eyes may need some time to adapt to the correct depth perception: black ridges are high-altitude separations between the colored valleys.

## Concept

The overall idea is that students use the surface implemented in `soapboxslide` to simulate the dynamics of a particle (or a connected set of particles) sliding down.
In addition to correctly implementing the dynamics, an additional challenge is do so as quickly as possible without missing any of the waypoints shown as dotted circles.

Two classes of physical models can be considered:

1. The most convenient is to assume a model of point particles strictly bound to the surface with holonomic constraints.
In this case, equations of motion can be derived using a Lagrangian, possibly with a generalized force to include non-conservative friction forces.

2. A more challenging scenario (not used for Py4Sci, but closer in spirit to soap box races) is to impose inequality constraints, allow particles to detach from the surface.

## Slide Class Usage

One can load a surface from a TOML file and calculate the altitude at a given point, e.g. x=5 and y=38, as follows:

```python
import numpy as np

from soapboxslide import Slide

slide = Slide.from_file("boxcar_blitz.toml")
print(slide(np.array([5.0, 38.0])))
```

This will show three results:

```text
(array(2.05331579), array(0.20172586), array(14.75904983))
```

These three values have the following meaning (in meter):

1. The progress along the track.
2. The orthogonal(ish) deviation from the bottom of the track.
3. The altitude of the track.

The `slide()` function is fully vectorized: one may also provide an array of points to calculate many altitudes efficiently.
(This is useful for plotting.)

The `slide()` function optionally supports alternative NumPy wrappers, such as those of [JAX](https://docs.jax.dev/) and its predecessor [Autograd](https://github.com/HIPS/autograd).
This allows for an efficient evaluation of analytical partial derivatives of the altitude.
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
slide = Slide.from_file("boxcar_blitz.toml")
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
[[ 0.21289734  0.10176584]
 [-1.85311237  1.84445056]
 [-2.16704263 -2.9280058 ]]
```

## Storing and Sharing Trajectory Data

The `soapboxslide` module also implements a `Trajectory` class for storing the results of a numerical integration of one or more particles sliding over the surface.
This class performs an initial validation on your trajectory data upon construction features `to_npz` and `from_npz` methods with which trajectories can be saved to and loaded from files.
This is useful if you want to share your trajectory with someone, e.g. for review, and to implement separate computation and visualization scripts (or notebooks).

To use the `Trajectory` class, create an instance as follows after completing the numerical integration of the equations of motion:

```python
traj = Trajectory(
    time=...,
    mass=...,
    gamma=...,
    pos=...,
    vel=...,
    grad=...,
    hess=...,
    spring_idx=...,
    spring_par=...,
    end_state=...,
    stop_time=...,
    stop_pos=...,
    stop_vel=...,
)
traj.to_file("traj.npz")
```

In this Python snippet, you need to replace all triple dots by arrays you defined or computed.
The meaning of and requirements for all attributes can be found in the docstrings in [`soapboxslide.py`](soapboxslide.py).
