# SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for Soap Box Slide."""

from pathlib import Path

import attrs
import numpy as np
import pytest
from matplotlib.figure import Figure

from soapboxslide import EndState, Slide, Trajectory

ROOT = Path(__file__).parent
SLIDE_FILES = ["boxcar_blitz.toml", "brutal_bends.toml"]


def test_readme_example():
    # The example in README.md. This guards against accidental changes to the surface.
    # Deliberate changes to the surface must update the README and this test together.
    slide = Slide.from_file(ROOT / "boxcar_blitz.toml")
    result = slide(np.array([5.0, 38.0]))
    np.testing.assert_allclose(result, [2.05331579, 0.20172586, 14.75904983], rtol=1e-7)


@pytest.mark.parametrize("name", SLIDE_FILES)
@pytest.mark.filterwarnings("error")
def test_plot(name, tmp_path):
    slide = Slide.from_file(ROOT / name)
    # Use Figure instead of pyplot, so the Agg backend is used without global state.
    fig = Figure()
    ax = fig.add_subplot()
    # A coarse grid keeps the test fast. All optional layers are enabled to cover all branches.
    slide.plot(fig, ax, spacing=0.5, add_altitude_lines=True, add_spline=True)
    fig.savefig(tmp_path / "slide.png")


def test_plot_image_extent():
    slide = Slide.from_file(ROOT / "boxcar_blitz.toml")
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
