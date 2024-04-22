import argparse
from cmcrameri import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.stats


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input',
                   help='Path of the input file.')
    
    p.add_argument('--pairwise', action='store_true')

    p.add_argument('--wm', action='store_true')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    stats = np.loadtxt(args.input)

    with open(args.input) as f:
        names = f.readline().strip('\n').strip('#').split(' ')[1:-1]
    if "" in names:
        names.remove("")
    if '' in names:
        names.remove('')

    if args.pairwise:
        new_names = []
        for name in names:
            new_names.append(name.split("_")[0])

        diag = np.take(np.diagonal(stats, offset=1), np.array([0, 2, 4, 6, 8, 10]))
        names = np.take(np.asarray(new_names), np.array([0, 2, 4, 6, 8, 10]))

        interpret = np.empty(len(diag), dtype=object)
        interpret[diag >= 0.7] = "strong"
        interpret[diag < 0.7] = "moderate"
        interpret[diag < 0.4] = "low"

        print(np.round(diag, decimals=2))
        print(interpret)
        print(names)

    if args.wm:
        stats = stats[-1]

        interpret = np.empty(len(stats), dtype=object)
        interpret[stats >= 0.7] = "strong"
        interpret[stats < 0.7] = "moderate"
        interpret[stats < 0.4] = "low"

        for i in range(len(names)):
            print(names[i], " : ", interpret[i], " --- ",
                  np.round(stats[i], decimals=2))


if __name__ == "__main__":
    main()
