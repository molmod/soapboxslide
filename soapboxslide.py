# Soap Box Slide is a computational take on soapbox racing.
# © 2025 Toon Verstraelen
#
# This file is part of Soap Box Slide.
#
# Soap Box Slide is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Soap Box Slide is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""A Computational Soapbox race."""

import tomllib
from enum import Enum

import attrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.typing import NDArray

__all__ = ("STANDARD_GRAVITY", "EndState", "Slide", "Trajectory")


MAX_TIME = 60.0
STANDARD_GRAVITY = 9.81

HILLSHADE = mcolors.LinearSegmentedColormap.from_list(
    "hillshade",
    [
        (0, 0, 0, 1.0),
        (0, 0, 0, 0.8),
        (0, 0, 0, 0.6),
        (0, 0, 0, 0.4),
        (0, 0, 0, 0.2),
        (0, 0, 0, 0.0),
        (1, 1, 1, 0.1),
        (1, 1, 1, 0.2),
        (1, 1, 1, 0.3),
        (1, 1, 1, 0.4),
        (1, 1, 1, 0.5),
    ],
)


def has_shape(shape):
    """Validator for shapes of arrays to be used with `attrs.field` definitions.

    Parameters
    ----------
    shape
        A tuple with the required shape. Each item specifies the requirements
        along one array dimension and can be one of the following:

        - `None`, meaning that along this dimension there are no restrictions.
        - An integer item requires a specific size.
        - A tuple item can be used to specify a range of a allowed sizes of
          the form `(begin, end)` where `begin` and `end` are integers or `None`
          and `end` is inclusive. `None` can be used to specify the absence
          of an upper or lower bound of the range.
        - A function that receives the instance, attribute, index and size being
          checked. This can be used to enforce consistency between different array
          attribtues.

    Returns
    -------
    validator
        A function that validates whether the array satisfies the given requirements.
        If not, the returned function raises a ValueError.
    """
    iter(shape)

    def validate(instance, attribute, value):
        if value.ndim != len(shape):
            raise ValueError(
                f"The array {attribute.name} has the wrong dimensionalty: "
                f"expected {len(shape)}, got {value.ndim}"
            )
        for i, req in enumerate(shape):
            if req is None:
                continue
            if isinstance(req, int):
                if value.shape[i] != req:
                    raise ValueError(
                        f"The array {attribute.name} has an incorrect size along axis {i}: "
                        f"expected {req}, got {value.shape[i]}"
                    )
            elif (
                isinstance(req, tuple)
                and len(req) == 2
                and all(isinstance(item, int) or item is None for item in req)
            ):
                if not (
                    (req[0] is None or value.shape[i] >= req[0])
                    and (req[1] is None or value.shape[i] <= req[1])
                ):
                    raise ValueError(
                        f"The array {attribute.name} has an incorect size along axis {i}: "
                        f"expected in range {req}, got {value.shape[i]}"
                    )
            elif callable(req):
                req(instance, attribute, i, value.shape[i])
            else:
                raise ValueError(
                    f"Cannot interpret size requirement of array {attribute.name} "
                    f"at index {i}: {req}"
                )

    return validate


def to_array(dtype, optional: bool = False):
    """Create an array converter for `attrs.field` definitions.

    Parameters
    ----------
    dtype
        The desired dtype to convert the array into.
    optional
        If set to True, `None` values are allowed and not converted to an array.

    Returns
    -------
    converter
        A function performing the conversion.
    """

    def convert(given):
        if given is None and optional:
            return None
        return np.asarray(given, dtype)

    return convert


@attrs.define
class Slide:
    """Altitude function and tracking coordinates of the (downhill) slide."""

    width: float = attrs.field(converter=float, kw_only=True)
    """Overall width of the slide arena."""

    height: float = attrs.field(converter=float, kw_only=True)
    """Overall height of the slide arena."""

    target_radius: float = attrs.field(converter=float, kw_only=True)
    """Radius of sphere round a way point for start and stop regions."""

    waypoints: NDArray = attrs.field(
        converter=to_array(float),
        validator=has_shape(((2, None), 3)),
        kw_only=True,
    )
    """Array with waypoints defining the slide geometry.

    This is an array with shape `(npoint, 3)`, where `npoint` is the number of waypoints.
    The collumns correspond to $x$, $y$ and $z$ coordinates of the bottom of slide.
    """

    # Internal pre-computed attributes, derived from waypoints.
    controls: NDArray = attrs.field(init=False)
    """Fine grained control points derived from the waypoints, used for actual computations."""

    path_lengths: NDArray = attrs.field(init=False)
    """The cumulative length of the slide (in the $xy$ plane) at each control point."""

    tangents: NDArray = attrs.field(init=False)
    """The tangent of the cubic spline at the control points."""

    def __attrs_post_init__(self):
        """Post-process the spline into a set of finer control points and related properties."""
        steps = np.arange(len(self.waypoints))
        path = sp.interpolate.CubicSpline(steps, self.waypoints, bc_type="natural")
        path_d1 = path.derivative()
        path_d2 = path_d1.derivative()

        # Sample the path with a step size approximately proportional
        # to the radius of curvature (regularized).
        us = [0.0]
        while True:
            u = us[-1]
            d1 = path_d1(u)[:2]
            d1_norm = np.linalg.norm(d1)
            d2 = path_d2(u)[:2]
            c = d2 / d1_norm**2 + d1 * np.dot(d1, d2) / d1_norm**4
            length = 0.5 / (0.25 + np.linalg.norm(c))
            next_u = us[-1] + length / d1_norm
            if next_u > steps[-1]:
                break
            us.append(next_u)

        # Pre-compute properties along the sample points
        self.controls = path(us)
        self.path_lengths = np.zeros(len(us))
        np.cumsum(
            np.linalg.norm(np.diff(self.controls[:, :2], axis=0), axis=1), out=self.path_lengths[1:]
        )
        derivs = path_d1(us)[:, :2]
        self.tangents = derivs / np.linalg.norm(derivs, axis=1).reshape(-1, 1)

    @classmethod
    def from_file(cls, path_toml: str):
        """Load the slide geometry from a TOML file."""
        with open(path_toml, "rb") as fh:
            data = tomllib.load(fh)
        return cls(**data)

    def __call__(self, points, *, npw=np):
        """Compute slide properties at the given (x, y) coordinates.

        Parameters
        ----------
        points
            An array of which the last index corresponds to the x and y coordinates (size 2).
            The evaluation is vectorized over all other indexes.
        npw
            A NumPy wrapper to use for array operations.
            This can be plain `numpy` if the only goal is to evaluate the altitude.
            Use `autograd.numpy` or `jax.numpy` when using automatic differentation
            or other features from the `autograd` or `jax` libraries, respectively.

        Returns
        -------
        track_x
            An estimate the distance along the slide,
            useful for tracking progress of the downhill trajectory.
        track_y
            An estimate of the distance orthogonal to the bottom of the slide,
            useful for tracking excursions.
        altitude
            The altitude of the slide at the given points.

        Notes
        -----
        Vectorized NumPy operations make use of autograd's NumPy wrapper,
        allowing for automatic differentiation of all results.
        """
        out_shape = points.shape[:-1]
        points_2d = points.reshape((-1, 2))
        track_x, track_y, altitude = self.compute(points_2d, npw=npw)
        return (track_x.reshape(out_shape), track_y.reshape(out_shape), altitude.reshape(out_shape))

    def compute(self, points, *, npw=np):
        """Low-level implementation of `__call__`.

        The differences with `__call__` is that `points` must be a 2D array,
        and the results are 1D arrays.
        """
        # Compute sample weights
        # - all squared distance between path points and points where functions must be computed.
        deltas = self.controls[:, :2] - points.reshape(-1, 1, 2)
        dists_sq = npw.einsum("pca,pca->pc", deltas, deltas)
        # - weights with shape (npoint, ncontrol), normalized each point over the controls.
        weights = npw.exp(-0.3 * dists_sq)
        weights /= weights.sum(axis=1).reshape(-1, 1)

        # Take weighted averages to construct slide track and altitude.
        lxs = npw.einsum("ca,pca->pc", self.tangents, deltas)
        deltas = deltas[:, :, ::-1]
        lys = npw.einsum("ca,pca,a->pc", self.tangents, deltas, np.array([-1, 1]))
        track_x = npw.einsum("pc,pc->p", lxs + self.path_lengths, weights)
        track_y = npw.sqrt(npw.einsum("pc,pc->p", lys**2, weights))
        altitude = npw.einsum("c,pc->p", self.controls[:, 2], weights) + 0.5 * track_y**2
        return (track_x, track_y, altitude)

    def plot(
        self,
        fig,
        ax,
        *,
        spacing: float = 0.1,
        alt_min: float = 0.0,
        alt_max: float = 25.0,
        nlevel: int = 26,
        cmap: str = "inferno_r",
        add_altitude_fill: bool = True,
        add_tracking: bool = True,
        add_hillshade: bool = True,
        add_altitude_lines: bool = False,
        add_spline: bool = False,
        add_targets: bool = True,
    ):
        """Make a plot of the slide surface viewed from the sky.

        Parameters
        ----------
        fig
            Matplotlib figure on which to draw.
        ax
            Matplotlib axes in which to draw.
        spacing
            The spacing between grid points for the contour plot.
        alt_min
            The lowest altitude for the colorscale.
        alt_max
            The highest altitude for the colorscale.
        nlevel
            The number of levels in the altitude color scale.
        cmap
            The colormap to use for the altitudes.
        add_altitude_full
            Plot the altitude with filled contours.
        add_tracking
            Plot the tracking coordinates.
        add_hillshade
            Overlay hillshading to enhance the depth perception.
        add_altitude_lines
            Plot the altitude with solid lines.
        add_spline
            Plot the spline from which the slide geometry is derived.
        add_targets
            Include starting, intermediate and stopping regions.
        """
        # Rectangular grid for contour plotting
        nx = round(self.width / spacing)
        ny = round(self.height / spacing)
        xspacing = self.width / nx
        yspacing = self.height / ny
        xg = np.arange(-1, nx + 2) * xspacing
        yg = np.arange(-1, ny + 2) * yspacing
        grid_points = np.empty((ny + 3, nx + 3, 2))
        grid_points[:, :, 0] = xg
        grid_points[:, :, 1] = yg.reshape(-1, 1)

        # Evaluate functions on grid points
        track_x, track_y, altitude = self(grid_points)

        # Plot the path, using a color code for the altitude
        if add_altitude_fill:
            cm = ax.contourf(
                xg,
                yg,
                altitude,
                levels=np.linspace(alt_min, alt_max, nlevel),
                cmap=cmap,
                extend="both",
            )
            fig.colorbar(cm, label="Altitude [m]", format=(lambda x, _: f"{x:.1f}"))
        if add_tracking:
            ax.contour(
                xg, yg, track_y, levels=np.arange(-3, 4), colors="k", linewidths=1, alpha=0.5
            )
            ax.contour(
                xg,
                yg,
                track_x,
                levels=np.arange(-10, self.path_lengths[-1] + 10),
                colors="k",
                linewidths=1,
                alpha=0.5,
            )
        if add_hillshade:
            # Compute hillshading using a finite difference filter
            altitude_dx = (altitude[1:-1, 2:] - altitude[1:-1, :-2]) / (2 * xspacing)
            altitude_dy = (altitude[2:, 1:-1] - altitude[:-2, 1:-1]) / (2 * yspacing)
            source = np.array([-1.0, 1.0, 0.0])
            source /= np.linalg.norm(source)
            cosine = (-source[0] * altitude_dx - source[1] * altitude_dy + source[2]) / np.sqrt(
                altitude_dx**2 + altitude_dy**2 + 1
            )
            ax.imshow(
                cosine,
                origin="lower",
                extent=(0, self.width, 0, self.height),
                vmin=-1,
                vmax=1,
                cmap=HILLSHADE,
                zorder=2,
            )
        if add_altitude_lines:
            ax.contour(
                xg,
                yg,
                altitude,
                levels=np.linspace(alt_min, alt_max, nlevel),
                colors="k",
                linewidths=1,
                alpha=0.5,
            )
        if add_spline:
            ax.scatter(
                self.controls[:, 0],
                self.controls[:, 1],
                c=self.controls[:, 2],
                vmin=alt_min,
                vmax=alt_max,
                cmap=cmap,
                edgecolors="k",
                zorder=3,
            )
            ax.plot(self.waypoints[:, 0], self.waypoints[:, 1], "ko", ms=10, zorder=3)
        if add_targets:
            colors = {0: "g", len(self.waypoints) - 1: "r"}
            for i, waypoint in enumerate(self.waypoints):
                color = colors.get(i, "w")
                alpha = 0.7 if color == "w" else 1.0
                ax.add_patch(
                    plt.Circle(
                        waypoint,
                        self.target_radius,
                        color="none",
                        ec=color,
                        lw=2,
                        ls=":",
                        zorder=4,
                        alpha=alpha,
                    )
                )
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    def get_hits(
        self, points: NDArray[float], stop: NDArray[float] | None = None, eps: float = 1e-6
    ) -> tuple[NDArray[bool], NDArray[int]]:
        """Return a boolean array with the targets that were hit.

        Parameters
        ----------
        points
            An array with shape `(nstep, ..., ncart)` with `ncart >= 2`.
            The first index represents a point of the trajectory and the last index
            corresponds to `x` and `y` coordinates, respectively.
            Intermediate indexes are allowed.
        stop
            Optionally, the stopping point identified by SciPy's `solve_ivp`.
            This is used for (double)checking the last waypoint.
        eps
            Error tolerance.

        Returns
        -------
        hits
            A boolean array with shape `(nwaypoint,)`. hit = `True`, miss = `False`.
        best
            An integer array with shape `(nwaypoint,)`.
            The indexes in the given `points` array that come closest to the corresponding waypoint.
        """
        hits = np.zeros(len(self.waypoints), dtype=bool)
        best = np.zeros(len(self.waypoints), dtype=int)
        nstep = points.shape[0]
        for i, waypoint in enumerate(self.waypoints[:, :2]):
            distances = np.linalg.norm(points[..., :2] - waypoint, axis=-1)
            distances = distances.reshape(nstep, -1).min(axis=1)
            best[i] = np.argmin(distances)
            if distances[best[i]] <= self.target_radius + eps:
                hits[i] = True
        if stop is not None:
            distance = np.linalg.norm(stop[..., :2] - self.waypoints[-1, :2], axis=-1)
            if distance.ndim > 0:
                distance = distance.min()
            if distance <= self.target_radius + eps:
                hits[-1] = True
        return hits, best


class EndState(Enum):
    """Flag used to indicate how the integration of the EOM ended."""

    STOP = 1
    """A proper end, by reaching the final target."""

    CRASH = 2
    """Some floating point issue that forced the integrator to give up."""

    FAR = 3
    """The integration lead to a position far (more than 5 m) outside the arena."""

    TIMEOUT = 4
    """The final target could not be reached in 60 seconds."""


def _validate_ntime(instance, attribute, index, size):
    if size != len(instance.time):
        raise ValueError(
            f"The number of time steps in the {attribute.name} array does not match "
            f"the length of the time array. Expected {len(instance.time)}, got {size}."
        )


def _validate_npoint(instance, attribute, index, size):
    if size != len(instance.mass):
        raise ValueError(
            f"The number of points in the {attribute.name} array does not match "
            f"the length of the mass array. Expected {len(instance.mass)}, got {size}."
        )


def _validate_nspring(instance, attribute, index, size):
    if size != len(instance.spring_idx):
        raise ValueError(
            f"The number of springs in the {attribute.name} array does not match "
            "the length number of springs in spring_idx. "
            f"Expected {len(instance.spring_idx)}, got {size}."
        )


@attrs.define
class Trajectory:
    """A container for trajectory objects. You must use this to create an NPZ file to be submitted.

    Note that this implementation already validate the shapes of the arrays.
    It is designed to work for both single and multiple points on the surface.

    Alle quantities must be stored in SI base units.
    """

    time: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape(((1, None),)),
        kw_only=True,
    )
    """1D array with the time steps at which the trajectory is recorded."""

    @time.validator
    def _validate_time(self, attribute, value):
        if (np.diff(value) < 0).any():
            raise ValueError("Time should always increase.")

    mass: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape(((1, None),)),
        kw_only=True,
    )
    """1D array with the mass of the point particles on the slide.

    In case of a single particle, this array just contains one mass.
    """

    @mass.validator
    def _validate_mass(self, attribute, value):
        if (value <= 0).any():
            raise ValueError("The point masses must be strictly positive.")

    gamma: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape((_validate_npoint,)),
        kw_only=True,
    )
    """1D array with the friction coefficients of the point particles on the slide.

    In case of a single particle, this array just contains one value.
    """

    @gamma.validator
    def _validate_gamma(self, attribute, value):
        if (value < 0).any():
            raise ValueError("The friction coefficients must be zero or positive.")

    pos: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape((_validate_ntime, _validate_npoint, 3)),
        kw_only=True,
    )
    """3D array with the positions of the points on the slide as a function of time.

    The three array dimensions correspond to time step, point particle and cartesian coordinate,
    respectively. Note that x, y and z coordinates must be stored.
    """

    vel: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape((_validate_ntime, _validate_npoint, 3)),
        kw_only=True,
    )
    """3D array with the velocities of the points on the slide as a function of time.

    The three array dimensions correspond to time step, point particle and cartesian coordinate,
    respectively. Note that x, y and z coordinates must be stored.
    """

    grad: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape((_validate_ntime, _validate_npoint, 2)),
        kw_only=True,
    )
    """3D array with the derivative of the altitude of the slide with respect to x and y.

    The three array dimensions correspond to time step, point particle and cartesian coordinate,
    respectively. Note that x and y derivatives must be stored.
    """

    hess: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape((_validate_ntime, _validate_npoint, 3)),
        kw_only=True,
    )
    """3D array with the second derivative of the altitude of the slide with respect to x and y.

    The three array dimensions correspond to time step, point particle and cartesian coordinate,
    respectively. Note that xx, xy and yy derivatives must be stored.
    """

    spring_idx: NDArray[int] = attrs.field(
        converter=to_array(int),
        validator=has_shape((None, 2)),
        factory=lambda: np.zeros((0, 2)),
        kw_only=True,
    )
    """2D array identifying the pairs of particles connected by springs.

    Each row corresponds to one spring and consists of two integer indexes referring
    to the two points being connected by a spring.
    """

    @spring_idx.validator
    def _validate_spring_idx(self, attribute, value):
        if (value < 0).any():
            raise ValueError("The spring indexes must be zero or positive.")
        if (value >= len(self.mass)).any():
            raise ValueError("Some spring indexes exceeed their maximum value (npoint - 1).")

    spring_par: NDArray[float] = attrs.field(
        converter=to_array(float),
        validator=has_shape((_validate_nspring, 3)),
        factory=lambda: np.zeros((0, 3)),
        kw_only=True,
    )
    """2D array with physical parameters of the springs.

    Each row corresponds to one spring and consists of three parameters:
    the force constant, the rest length and the spring damping coefficient, respectively.
    """

    @spring_par.validator
    def _validate_spring_par(self, attribute, value):
        if (value[:, 0] <= 0).any():
            raise ValueError("The force constants must be strictly positive.")
        if (value[:, 1] <= 0).any():
            raise ValueError("The rest lengths must be strictly positive.")
        if (value[:, 2] < 0).any():
            raise ValueError("The spring damping coefficients must be zero or positive.")

    end_state: EndState = attrs.field(converter=EndState, kw_only=True)
    """Specifies how the trajectory has ended.

    This can be one of the EndState enums, e.g. `EndState.STOP` if the integration was stopped
    because the last target was reached.
    """

    stop_time: float | None = attrs.field(kw_only=True, default=None)
    """If the last target was reached, specify its time here."""

    @stop_time.validator
    def _validate_stop_time(self, attribute, value):
        if value is not None and value < self.time[-1]:
            raise ValueError("The stop time cannot be earlier than the last trajectory point.")

    stop_pos: NDArray[float] = attrs.field(
        converter=to_array(float, optional=True),
        validator=attrs.validators.optional(has_shape((_validate_npoint, 3))),
        default=None,
        kw_only=True,
    )
    """The position(s) of the point(s) when the final target was reached."""

    @stop_pos.validator
    def _validate_stop_pos(self, attribute, value):
        if value is None and self.stop_time is not None:
            raise ValueError("The stop_time is specified without a stop_pos.")

    stop_vel: NDArray[float] = attrs.field(
        converter=to_array(float, optional=True),
        validator=attrs.validators.optional(has_shape((_validate_npoint, 3))),
        default=None,
        kw_only=True,
    )
    """The velocities(s) of the point(s) when the final target was reached."""

    @stop_vel.validator
    def _validate_stop_vel(self, attribute, value):
        if value is None and self.stop_time is not None:
            raise ValueError("The stop_time is specified without a stop_vel.")

    def to_file(self, path_npz):
        data = attrs.asdict(self)
        data["end_state"] = self.end_state.value
        data = {key: value for key, value in data.items() if value is not None}
        np.savez_compressed(path_npz, **data, allow_pickle=False)

    @classmethod
    def from_file(cls, path_npz):
        data = np.load(path_npz)
        return cls(**data)
