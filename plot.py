#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
# SPDX-License-Identifier: GPL-3.0-or-later
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
