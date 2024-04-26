wdir="/home/pkaran/Samsung/data/MT_Diffusion/rotated_myelo_inferno";
# wdir="/home/local/USHERBROOKE/karp2601/Samsung/data/MT_Diffusion/myelo_inferno";
source_dir="source";
# source_dir="Research/source";
cd $wdir;
bin_width=5;
bin_width_dir="${bin_width}_degree_bins";
subject="sub-001_ses-2";
for bundle in rbx_flow/results_rbx/${subject}/Masks/*;
    do bundle_name=$(basename -- "${bundle%%.*}");
    if [ 1 == 1 ];
    #if [ "$bundle_name" = "AF_L" ];
        then echo $bundle_name;
        outdir="orientation_analysis/${subject}/characterization_per_bundle/${bin_width_dir}/${bundle_name}"
        mkdir -p $outdir;
        python ~/${source_dir}/MRM_characterization/scripts/scil_characterize_orientation_dependence.py new_peaks/${subject}/peaks.nii.gz  new_peaks/${subject}/peak_values.nii.gz tractoflow/output/results/${subject}/DTI_Metrics/${subject}__fa.nii.gz new_peaks/${subject}/nufo.nii.gz tractoflow/output/results/${subject}/Segment_Tissues/${subject}__mask_wm.nii.gz $outdir --measures ihmt_flow/output_b1_corr_no_modif/results/${subject}/Compute_ihMT/Register_ihMT_maps/${subject}__MTR_b1_warped.nii.gz ihmt_flow/output_b1_corr_no_modif/results/${subject}/Compute_ihMT/Register_ihMT_maps/${subject}__ihMTR_b1_warped.nii.gz ihmt_flow/output_b1_corr_no_modif/results/${subject}/Compute_ihMT/Register_ihMT_maps/${subject}__MTsat_b1_warped.nii.gz ihmt_flow/output_b1_corr_no_modif/results/${subject}/Compute_ihMT/Register_ihMT_maps/${subject}__ihMTsat_b1_warped.nii.gz --in_e1 tractoflow/output/results/${subject}/DTI_Metrics/${subject}__evecs_v1.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --save_npz_files --in_roi $bundle --bin_width_1f $bin_width --min_nb_voxels 1;
    fi;
done;