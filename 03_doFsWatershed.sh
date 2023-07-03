#!/bin/bash
# ======================================================================================================================
# run freesurfer, the watershed algorithm (needed by mne to compute headmodel)
# $1 subject id

# e.g. /mnt/storage/tier2/robbot/ProjectData/MEGRID/prg/03_doFsWatershed.sh 19820925PEGA

# ======================================================================================================================

function fs_watershed () {

    bidsPath='/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/'
    export SUBJECTS_DIR='/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/freesurfer/'

    # check if this subject has been already processed using freesurfer
    if [[ ! -d "$SUBJECTS_DIRsub-$1" ]]
        then # if not we process it

        # run freesurfer reconstruction
        recon-all -i $bidsPath/sub-$1/ses-mri/anat/sub-$1_ses-mri_T1w.nii.gz -subjid sub-$1 -sd $SUBJECTS_DIR -cw256 -all

        # make the hippocampus segmentation from aseg a label
        for roi_name in 'Left-Hippocampus' 'Right-Hippocampus'
            do
            # get freesurfer color
            roi_color=$(grep $roi_name $FREESURFER_HOME/FreeSurferColorLUT.txt | cut -c -3)
		        hemi=$(echo ${roi_name:0:1} | tr [:upper:] [:lower:])
		        roi=$(cut -d '-' -f2 <<< "$roi_name" | tr [:upper:] [:lower:])

            # convert the mri to label
            mri_vol2label --c "$SUBJECTS_DIR/sub-$1/mri/aparc+aseg.mgz" \
                          --l "$SUBJECTS_DIR/sub-$1/label/${hemi}h.aseg.$roi.label" \
                          --id $roi_color
        done
    fi

    # check if this subject has been already processed using watershed algorhitm
    if [[ ! -d "$SUBJECTS_DIR/sub-$1/bem/watershed" ]]
        then
        # compute watershed
        # https://mne.tools/stable/generated/commands.html#mne-watershed-bem
        mne watershed_bem -s sub-$1 -d "$SUBJECTS_DIR" --overwrite
        
        # move the files where mne expects them
        files=$(find "$SUBJECTS_DIR/sub-$1/bem/watershed" -name "sub-$1*" -type f)
        while IFS= read -r file
            do
                # get the surface name from the filename
                surface_name=${file#"$SUBJECTS_DIR/sub-$1/bem/watershed/sub-$1"_}
                    # and remove the "_surface" suffix
                surface_name=${surface_name%"_surface"}
                cp -f "${file}" "$SUBJECTS_DIR/sub-$1/bem/${surface_name}.surf"
        done <<< "$files"
        
        # create high density meshes for coregistration
        mne make_scalp_surfaces -s sub-$1 -d $SUBJECTS_DIR -o -f
    fi # end if

}

# ======================================================================================================================

# activate the mne environment https://github.com/conda/conda/issues/7980
# get the conda base path
CONDA_BASE=$(conda info --base)
# source the conda path
source $CONDA_BASE/etc/profile.d/conda.sh
# activate the environment
conda activate gft2

# activate freesurfer
export FREESURFER_HOME=/opt/freesurfer-7.1.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# run the main function
fs_watershed $1
