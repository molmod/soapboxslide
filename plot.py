#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot the two slides included in the soapboxslide package."""

import matplotlib.pyplot as plt

from soapboxslide import Slide


def main():
    fig, axs = plt.subplots(2, 1, figsize=(7, 12))
    names = ["boxcar_blitz", "brutal_bends"]
    for ax, name in zip(axs, names, strict=True):
        slide = Slide.from_name(name)
        slide.plot(fig, ax)
        ax.set_title(name)
    fig.savefig("slides.jpg")


if __name__ == "__main__":
    main()
