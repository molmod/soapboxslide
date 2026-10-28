#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Toon Verstraelen <Toon.Verstraelen@UGent.be>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot the two slides included in the soapboxslide package.

This script uses JAX to speed up the evaluation of the slide surface.
JAX is not installed with soapboxslide by default, see README.md.
"""

import jax
import matplotlib.pyplot as plt

from soapboxslide import Slide

# We recommend double precision:
jax.config.update("jax_enable_x64", True)


def main():
    fig, axs = plt.subplots(2, 1, figsize=(7, 12))
    names = ["boxcar_blitz", "brutal_bends"]
    for ax, name in zip(axs, names, strict=True):
        slide = Slide.from_name(name)
        # The lambda is needed because jax.jit cannot be applied to a Slide instance directly.
        slide.plot(fig, ax, func=jax.jit(lambda points, slide=slide: slide(points)))
        ax.set_title(name)
    fig.savefig("slides.jpg")


if __name__ == "__main__":
    main()
