#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np


def load_spectrum(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("%s must contain at least two numeric columns" % path)
    return data[:, 0], data[:, 1]


def interpolate(reference_energy, reference_value, target_energy):
    return np.interp(target_energy, reference_energy, reference_value)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyATB COHP spectrum with the standalone COHP script output."
    )
    parser.add_argument("pyatb", help="Path to Out/COHP/COHP.dat")
    parser.add_argument("standalone", help="Path to standalone broadened COHP spectrum")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()

    pyatb_energy, pyatb_value = load_spectrum(Path(args.pyatb))
    standalone_energy, standalone_value = load_spectrum(Path(args.standalone))
    standalone_on_grid = interpolate(standalone_energy, standalone_value, pyatb_energy)

    diff = pyatb_value - standalone_on_grid
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    passed = np.allclose(pyatb_value, standalone_on_grid, rtol=args.rtol, atol=args.atol)

    print("points: %d" % pyatb_energy.size)
    print("max_abs_diff: %.12g" % max_abs)
    print("rms_diff: %.12g" % rms)
    print("rtol: %.3g" % args.rtol)
    print("atol: %.3g" % args.atol)
    if not passed:
        print("comparison: FAILED")
        return 1
    print("comparison: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
