# wdir="/home/local/USHERBROOKE/karp2601/Samsung/data/MT_Diffusion/myelo_inferno";
# wdir="/home/local/USHERBROOKE/karp2601/data/stockage/MT_Diffusion/myelo_inferno";
wdir="/home/pkaran/Samsung/data/MT_Diffusion/myelo_inferno";
source_dir="source";
# source_dir="Research/source";
cd ${wdir}/ihMT;
all_subs=(sub-026*);
cd $wdir;
bin_width=5;
for sub in "${all_subs[@]}";
    do echo $sub;
    bundles_path=${wdir}/bundles/$sub/masks;
    bundles=$bundles_path/*.nii.gz;
    for bundle in $bundles;
        do bundle_name=$(basename -- "${bundle%%.*}");
        echo $bundle_name;
        outdir="intra_subject_w_b1/characterization/bundles_${bin_width}deg_bins/${sub}/${bundle_name}";
        mkdir -p $outdir;
        python ~/${source_dir}/MRM_characterization/scripts/scil_characterize_orientation_dependence.py FODF_metrics/${sub}/new_peaks/peaks.nii.gz  FODF_metrics/${sub}/new_peaks/peak_values.nii.gz DTI_metrics/${sub}/${sub}__dti_fa.nii.gz FODF_metrics/${sub}/new_peaks/nufo.nii.gz wm_mask/${sub}/${sub}__wm_mask.nii.gz $outdir --measures ihMT/${sub}/${sub}__MTR_b1_warped.nii.gz ihMT/${sub}/${sub}__ihMTR_b1_warped.nii.gz ihMT/${sub}/${sub}__MTsat_b1_warped.nii.gz ihMT/${sub}/${sub}__ihMTsat_b1_warped.nii.gz --in_e1 DTI_metrics/${sub}/${sub}__dti_evecs_v1.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --in_roi $bundle --save_plots --bin_width_1f $bin_width --min_nb_voxels 1 --compute_three_fiber_crossings;
    done;
done;