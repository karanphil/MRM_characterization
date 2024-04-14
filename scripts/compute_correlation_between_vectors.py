import argparse
import numpy as np
import pandas as pd


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument('in_stats', nargs=2,
                   help='List of all stats directories.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    stats = []
    for in_stat in args.in_stats:
        tmp_stat = np.loadtxt(in_stat)
        stats.append(tmp_stat)

    to_analyse = np.asarray(stats)
    dataset = pd.DataFrame(data=to_analyse.T)
    corr = dataset.corr()
    print(corr[0][1])


if __name__ == "__main__":
    main()
