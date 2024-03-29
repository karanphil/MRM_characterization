# wdir="/home/pkaran/Samsung/data/MT_Diffusion/myelo_inferno";
wdir="/home/local/USHERBROOKE/karp2601/Samsung/data/MT_Diffusion/myelo_inferno";
# wdir="/home/local/USHERBROOKE/karp2601/data/stockage/MT_Diffusion/myelo_inferno";
# source_dir="source";
source_dir="Research/source";
cd ${wdir}/DTI_metrics;
all_ses=(sub-026*);
cd $wdir;
bin_width=5;
for ses in "${all_ses[@]}";
    do echo $ses;
    outdir="sub-026/orientation_dependence/whole_wm_5deg_bins/${ses}";
    mkdir -p $outdir;
    python ~/${source_dir}/MRM_characterization/scripts/scil_characterize_orientation_dependence.py FODF_metrics/${ses}/new_peaks/peaks.nii.gz  FODF_metrics/${ses}/new_peaks/peak_values.nii.gz DTI_metrics/${ses}/${ses}__dti_fa.nii.gz FODF_metrics/${ses}/new_peaks/nufo.nii.gz wm_mask/${ses}/${ses}__wm_mask.nii.gz $outdir --measures ihMT/${ses}/${ses}__MTR_b1_warped.nii.gz ihMT/${ses}/${ses}__ihMTR_b1_warped.nii.gz ihMT/${ses}/${ses}__MTsat_b1_warped.nii.gz ihMT/${ses}/${ses}__ihMTsat_b1_warped.nii.gz --in_e1 DTI_metrics/${ses}/${ses}__dti_evecs_v1.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --save_npz_files --bin_width_1f $bin_width --min_nb_voxels 1 --compute_three_fiber_crossings;
done;