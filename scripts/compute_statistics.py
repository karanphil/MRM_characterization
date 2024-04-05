import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.stats


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_dir',
                   help='Path of the output directory.')
    
    p.add_argument('--in_results', nargs='+', required=True,
                   help='List of all results directories.')
    
    p.add_argument('--measures', nargs='+', required=True,
                   help='List of all measures to analyse.')
    
    p.add_argument('--names', nargs='+',
                   help='List of names.')
    
    p.add_argument('--whole_wm', default=[],
                   help='Path to the whole WM characterization.')
    
    p.add_argument('--suffix', default='', type=str)

    p.add_argument('--is_bundles', action='store_true')

    p.add_argument('--along_measures', action='store_true')

    p.add_argument('--crossing', action='store_true')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    min_nb_voxels = args.min_nb_voxels
    names = []
    if args.names:
        names = args.names

    out_dir = Path(args.out_dir)

    results = []
    result_MCP = None
    for i, result in enumerate(args.in_results):
        if args.names:
            if str(Path(result).parts[-2]) in names:
                print("Loading: ", result)
                results.append(np.load(result))
        else:
            if str(Path(result).parent.name) == 'MCP':
                result_MCP = result
            else:
                print("Loading: ", result)
                results.append(np.load(result))
                names.append(str(Path(result).parent.name))

    if result_MCP:
        print("Loading: ", result_MCP)
        results.append(np.load(result_MCP))
        names.append(str(Path(result_MCP).parent.name))

    if args.whole_wm:
        print("Loading: ", args.whole_wm)
        whole_wm = np.load(args.whole_wm)
        results.append(whole_wm)
        names.append("WM")

    nb_bins = len(results[0]['Angle_min'])
    nb_results = len(results)
    nb_measures = len(args.measures)

    if args.crossing:
        for measure in args.measures:
            print(measure)
            coeff_vars = np.zeros((4, nb_bins))
            for j, frac in enumerate(range(4)):
                to_analyse = np.zeros((nb_results, nb_bins))
                for i, result in enumerate(results):
                    to_analyse[i] = np.diagonal(result[measure][frac])
                    to_analyse[i, np.diagonal(result['Nb_voxels'][frac]) < min_nb_voxels] = np.nan
                coeff_vars[j] = scipy.stats.variation(to_analyse, axis=0)
            coeff_var = np.nanmean(coeff_vars, axis=0)

            out_path = out_dir / (measure + '_' + args.suffix + '_variation.txt')
            np.savetxt(out_path, [np.nanmean(coeff_var) * 100])
        return 0

    if not args.along_measures:
        for measure in args.measures:
            print(measure)
            to_analyse = np.zeros((nb_results, nb_bins))
            bundle_names = ''
            for i, result in enumerate(results):
                to_analyse[i] = result[measure]
                to_analyse[i, result['Nb_voxels'] < min_nb_voxels] = np.nan
                bundle_names += names[i] + ' '

            if args.is_bundles:
                dataset = pd.DataFrame(data=to_analyse.T)
                corr = dataset.corr()
                out_path = out_dir / (measure + '_' + args.suffix + '_correlation.txt')
                np.savetxt(out_path, corr, header=bundle_names)

                variation_matrix = np.zeros((nb_results, nb_results))
                pair_array = np.zeros((2, nb_bins))
                bundles_idx = np.flip(np.arange(1, nb_results, 1))
                for b_idx in bundles_idx:
                    for next_b_idx in range(b_idx):
                        pair_array[0] = to_analyse[b_idx]
                        pair_array[1] = to_analyse[next_b_idx]
                        variation_matrix[b_idx, next_b_idx] = np.nanmean(scipy.stats.variation(pair_array, axis=0))
                        variation_matrix[next_b_idx, b_idx] = variation_matrix[b_idx, next_b_idx]

                out_path = out_dir / (measure + '_' + args.suffix + '_variation.txt')
                np.savetxt(out_path, variation_matrix * 100, header=bundle_names)

            else:
                coeff_var = scipy.stats.variation(to_analyse, axis=0)
                out_path = out_dir / (measure + '_' + args.suffix + '_variation.txt')
                np.savetxt(out_path, [np.nanmean(coeff_var) * 100])

    if args.along_measures:
        all_corrs = np.zeros(nb_results)
        bundle_names = ''
        for j, result in enumerate(results):
            bundle_names += names[j] + ' '
            to_analyse = np.zeros((nb_measures, nb_bins))
            for i, measure in enumerate(args.measures):
                to_analyse[i] = result[measure]
                to_analyse[i, result['Nb_voxels'] < min_nb_voxels] = np.nan
            if args.is_bundles:
                dataset = pd.DataFrame(data=to_analyse.T)
                corr = dataset.corr()
                all_corrs[j] = corr[0][1]

        out_path = out_dir / (args.suffix + 'intra_measure_correlation.txt')
        np.savetxt(out_path, all_corrs, header=bundle_names)


if __name__ == "__main__":
    main()
