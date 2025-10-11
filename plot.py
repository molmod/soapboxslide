#!/usr/bin/env python3
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
"""Plot the two slides included in the repository."""

import matplotlib.pyplot as plt

from soapboxslide import Slide


def main():
    fig, axs = plt.subplots(2, 1, figsize=(7, 12))
    paths_toml = ["boxcar_blitz.toml", "brutal_bends.toml"]
    for ax, path_toml in zip(axs, paths_toml, strict=True):
        slide = Slide.from_file(path_toml)
        slide.plot(fig, ax)
        ax.set_title(path_toml)
    fig.savefig("slides.jpg")


if __name__ == "__main__":
    main()
