# SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for Soap Box Slide."""

import re
from importlib.resources import as_file, files

import attrs
import numpy as np
import pytest
import scipy as sp
from matplotlib.figure import Figure

from soapboxslide import EndState, Slide, Trajectory

SLIDE_NAMES = ["boxcar_blitz", "brutal_bends"]


def test_readme_example():
    # The example in README.md. This guards against accidental changes to the surface.
    # Deliberate changes to the surface must update the README and this test together.
    slide = Slide.from_name("boxcar_blitz")
    result = slide(np.array([5.0, 38.0]))
    np.testing.assert_allclose(result, [2.04001764, 0.20172108, 14.75990762], rtol=1e-7)


@pytest.mark.parametrize("name", SLIDE_NAMES)
def test_from_name(name):
    # A slide loaded by name must be identical to the same TOML file loaded with from_file.
    slide_name = Slide.from_name(name)
    with as_file(files("soapboxslide") / f"{name}.toml") as path_toml:
        slide_file = Slide.from_file(path_toml)
    for attr in attrs.fields(Slide):
        assert np.array_equal(getattr(slide_name, attr.name), getattr(slide_file, attr.name))


@pytest.mark.parametrize("name", ["boxcar_blitz.toml", "Boxcar_Blitz", "../pyproject", ""])
def test_from_name_unknown(name):
    # The error message lists exactly the included slides.
    with pytest.raises(ValueError, match=re.escape(str(SLIDE_NAMES))):
        Slide.from_name(name)


@pytest.mark.parametrize("name", SLIDE_NAMES)
@pytest.mark.filterwarnings("error")
def test_far_from_track(name):
    # Far from all control points, the weights must not underflow to zero (0/0 = NaN).
    slide = Slide.from_name(name)
    result = slide(np.array([[200.0, -150.0], [-1e3, 1e3]]))
    assert all(np.isfinite(item).all() for item in result)


@pytest.fixture(params=["numpy", "jax", "autograd"])
def npw_backend(request):
    """Return a NumPy wrapper and its grad and hessian transformations (None for NumPy).

    JAX and autograd tests are skipped if these libraries are not installed.
    """
    if request.param == "numpy":
        yield np, None, None
    elif request.param == "jax":
        jax = pytest.importorskip("jax")
        # Double precision is needed for a meaningful comparison.
        # The original setting is restored afterwards, to avoid side effects on other tests.
        x64_orig = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        # Without jit, JAX is too slow because it dispatches every operation separately.
        yield (
            pytest.importorskip("jax.numpy"),
            lambda func: jax.jit(jax.grad(func)),
            lambda func: jax.jit(jax.hessian(func)),
        )
        jax.config.update("jax_enable_x64", x64_orig)
    elif request.param == "autograd":
        autograd = pytest.importorskip("autograd")
        yield pytest.importorskip("autograd.numpy"), autograd.grad, autograd.hessian


# Points near the start, near the track, between parts of the track and outside the arena.
NPW_POINTS = np.array([[5.0, 38.0], [12.1, 7.3], [10.7, 25.5], [30.0, 20.0], [-3.0, 44.0]])


@pytest.mark.parametrize("name", SLIDE_NAMES)
def test_npw_values(name, npw_backend):
    npw, _, _ = npw_backend
    slide = Slide.from_name(name)
    expected = slide(NPW_POINTS)
    # Use a 3D input array to also test the vectorization over multiple axes.
    points = npw.array(NPW_POINTS.reshape(5, 1, 2))
    for item, item_expected in zip(slide(points, npw=npw), expected, strict=True):
        assert item.shape == (5, 1)
        np.testing.assert_allclose(np.asarray(item)[:, 0], item_expected, rtol=1e-12)


@pytest.mark.parametrize("npw_backend", ["jax", "autograd"], indirect=True)
@pytest.mark.parametrize("name", SLIDE_NAMES)
def test_npw_derivatives(name, npw_backend):
    npw, grad, hessian = npw_backend
    slide = Slide.from_name(name)
    alt_grad = grad(lambda point: slide(point, npw=npw)[2])
    alt_hess = hessian(lambda point: slide(point, npw=npw)[2])
    eps = 1e-5
    for point in NPW_POINTS:
        # Compare to central finite differences of the altitude (computed with plain NumPy)
        # and of the gradient (computed with automatic differentiation).
        grad_fd = np.zeros(2)
        hess_fd = np.zeros((2, 2))
        for i in range(2):
            delta = np.zeros(2)
            delta[i] = eps
            grad_fd[i] = (slide(point + delta)[2] - slide(point - delta)[2]) / (2 * eps)
            hess_fd[i] = (np.asarray(alt_grad(point + delta)) - alt_grad(point - delta)) / (2 * eps)
        np.testing.assert_allclose(alt_grad(point), grad_fd, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(alt_hess(point), hess_fd, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("name", SLIDE_NAMES)
def test_controls_endpoints(name):
    slide = Slide.from_name(name)
    np.testing.assert_allclose(slide.controls[0], slide.waypoints[0])
    np.testing.assert_allclose(slide.controls[-1], slide.waypoints[-1])


def test_controls_straight_nonuniform():
    # Unevenly spaced waypoints on a straight line: the spline speed varies,
    # but the curvature is zero, so each step in the spline parameter u
    # corresponds to a (first-order) length of 0.5 / 0.25 = 2 m.
    waypoints = np.array([[x, 2.0, 0.0] for x in [0.0, 2.0, 5.0, 9.0, 14.0, 20.0]])
    slide = Slide(width=20.0, height=4.0, target_radius=1.0, waypoints=waypoints)
    np.testing.assert_allclose(slide.controls[:, 1], 2.0)
    # Recover u at each control from the x coordinate.
    path = sp.interpolate.CubicSpline(np.arange(len(waypoints)), waypoints[:, 0], bc_type="natural")
    us = np.array([path.solve(x)[0] for x in slide.controls[:, 0]])
    speeds = path.derivative()(us[:-2])
    np.testing.assert_allclose(np.diff(us)[:-1], 2.0 / speeds)


@pytest.mark.parametrize("name", SLIDE_NAMES)
@pytest.mark.filterwarnings("error")
def test_plot(name, tmp_path):
    slide = Slide.from_name(name)
    # Use Figure instead of pyplot, so the Agg backend is used without global state.
    fig = Figure()
    ax = fig.add_subplot()
    # A coarse grid keeps the test fast. All optional layers are enabled to cover all branches.
    slide.plot(fig, ax, spacing=0.5, add_altitude_lines=True, add_spline=True)
    fig.savefig(tmp_path / "slide.png")


def test_plot_image_extent():
    slide = Slide.from_name("boxcar_blitz")
    fig = Figure()
    ax = fig.add_subplot()
    spacing = 0.5
    slide.plot(fig, ax, spacing=spacing, add_altitude_fill=True, add_hillshade=True)
    nx = round(slide.width / spacing)
    ny = round(slide.height / spacing)
    dx = slide.width / nx
    dy = slide.height / ny
    # Pixel centers must coincide with the grid points 0, dx, ..., width (same for y).
    expected = [-0.5 * dx, slide.width + 0.5 * dx, -0.5 * dy, slide.height + 0.5 * dy]
    assert len(ax.images) == 2
    for image in ax.images:
        assert image.get_array().shape == (ny + 1, nx + 1)
        np.testing.assert_allclose(image.get_extent(), expected)


@pytest.fixture
def straight_slide():
    return Slide(
        width=20,
        height=10,
        target_radius=1.0,
        waypoints=[[0, 5, 2], [10, 5, 1], [20, 5, 0]],
    )


def test_get_hits_straight(straight_slide):
    # A single point moving along the slide, slightly off-center.
    xs = np.linspace(0, 20, 21)
    points = np.zeros((21, 1, 3))
    points[:, 0, 0] = xs
    points[:, 0, 1] = 5.5
    hits, best = straight_slide.get_hits([1.0], points)
    assert hits.tolist() == [True, True, True]
    assert best.tolist() == [0, 10, 20]


def test_get_hits_miss(straight_slide):
    # A parallel trajectory too far from the waypoints.
    points = np.zeros((21, 1, 3))
    points[:, 0, 0] = np.linspace(0, 20, 21)
    points[:, 0, 1] = 7.0
    hits, best = straight_slide.get_hits([1.0], points)
    assert hits.tolist() == [False, False, False]
    assert best.tolist() == [0, 10, 20]


def test_get_hits_center_of_mass(straight_slide):
    # Neither point passes through the targets, but their center of mass does.
    points = np.zeros((21, 2, 3))
    points[:, :, 0] = np.linspace(0, 20, 21).reshape(-1, 1)
    points[:, 0, 1] = 2.0
    points[:, 1, 1] = 6.0
    hits, _ = straight_slide.get_hits([1.0, 3.0], points)
    assert hits.tolist() == [True, True, True]
    # With swapped masses, the center of mass is too far from the targets.
    hits, _ = straight_slide.get_hits([3.0, 1.0], points)
    assert hits.tolist() == [False, False, False]


def test_get_hits_stop(straight_slide):
    # The trajectory ends before the last target, which is only reached at the stop point.
    points = np.zeros((16, 1, 3))
    points[:, 0, 0] = np.linspace(0, 15, 16)
    points[:, 0, 1] = 5.0
    hits, _ = straight_slide.get_hits([1.0], points)
    assert hits.tolist() == [True, True, False]
    hits, _ = straight_slide.get_hits([1.0], points, stop=[[19.5, 5.0, 0.0]])
    assert hits.tolist() == [True, True, True]


def make_trajectory_kwargs():
    """Return keyword arguments for a valid trajectory with two points and one spring."""
    return {
        "time": [0, 1, 2],
        "mass": [1, 1.2],
        "gamma": [0.3, 0.0],
        "pos": [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 2, 0], [1, 2, 0]],
        ],
        "vel": [
            [[0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 1, 0]],
        ],
        "grad": [
            [[0.5, 0.1], [1.2, 0.3]],
            [[0.6, 0.2], [1.3, 0.4]],
            [[0.7, 0.3], [1.4, 0.5]],
        ],
        "hess": [
            [[0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 1, 0]],
        ],
        "spring_idx": [[0, 1]],
        "spring_par": [[100, 0.5, 1.2]],
        "end_state": EndState.STOP,
        "stop_time": 30.0,
        "stop_pos": [[10, 0, 0], [10, 1, 0]],
        "stop_vel": [[0, 0, 0], [0, 0, 0]],
    }


def test_npz_trajectory(tmp_path):
    traj = Trajectory(**make_trajectory_kwargs())
    path = tmp_path / "trajectory.npz"
    traj.to_file(path)
    loaded_traj = Trajectory.from_file(path)

    for attr in attrs.fields(Trajectory):
        val_orig = getattr(traj, attr.name)
        val_loaded = getattr(loaded_traj, attr.name)
        assert np.array_equal(val_orig, val_loaded)
        assert type(val_orig) is type(val_loaded)


def test_trajectory_repeated_time():
    # Equal consecutive times are allowed, e.g. when concatenating integration segments.
    kwargs = make_trajectory_kwargs()
    kwargs["time"] = [0, 1, 1]
    Trajectory(**kwargs)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"time": [0, 2, 1]}, "Time must not decrease"),
        ({"mass": [1, 0]}, "strictly positive"),
        ({"gamma": [0.3, -0.1]}, "zero or positive"),
        ({"gamma": [0.3]}, "number of points in the gamma array"),
        ({"pos": [[[0, 0, 0], [1, 0, 0]]]}, "number of time steps in the pos array"),
        ({"grad": np.zeros((3, 2, 3))}, "incorrect size along axis 2"),
        ({"hess": np.zeros((3, 2))}, "wrong dimensionality"),
        ({"spring_idx": [[0, 2]]}, "exceed their maximum"),
        ({"spring_idx": [[-1, 1]]}, "zero or positive"),
        ({"spring_par": [[0, 0.5, 1.2]]}, "force constants"),
        ({"spring_par": [[100, 0.5, 1.2], [100, 0.5, 1.2]]}, "number of springs"),
        ({"end_state": 99}, "not a valid EndState"),
        ({"stop_time": 1.5}, "stop time cannot be earlier"),
        ({"stop_pos": None}, "without a stop_pos"),
        ({"stop_vel": None}, "without a stop_vel"),
    ],
)
def test_trajectory_invalid(changes, match):
    kwargs = make_trajectory_kwargs()
    kwargs.update(changes)
    with pytest.raises(ValueError, match=match):
        Trajectory(**kwargs)
