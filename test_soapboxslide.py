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
"""Unit tests for Soap Box Slide."""

import attrs
import numpy as np

from soapboxslide import EndState, Trajectory


def test_npz_trajectory(tmpdir):
    traj = Trajectory(
        time=[0, 1, 2],
        mass=[1, 1.2],
        gamma=[0.3, 0.0],
        pos=[
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 2, 0], [1, 2, 0]],
        ],
        vel=[
            [[0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 1, 0]],
        ],
        grad=[
            [[0.5, 0.1], [1.2, 0.3]],
            [[0.6, 0.2], [1.3, 0.4]],
            [[0.7, 0.3], [1.4, 0.5]],
        ],
        hess=[
            [[0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 1, 0]],
        ],
        spring_idx=[[0, 1]],
        spring_par=[[100, 0.5, 1.2]],
        end_state=EndState.STOP,
        stop_time=30.0,
        stop_pos=[[10, 0, 0], [10, 1, 0]],
        stop_vel=[[0, 0, 0], [0, 0, 0]],
    )
    path = tmpdir.join("trajectory.npz")
    traj.to_file(path)
    print(path)
    loaded_traj = Trajectory.from_file(path)

    for attr in attrs.fields(Trajectory):
        val_orig = getattr(traj, attr.name)
        val_loaded = getattr(loaded_traj, attr.name)
        assert np.array_equal(val_orig, val_loaded)
