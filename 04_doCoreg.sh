#!/bin/bash

# ======================================================================================================================
# run coregistration subject by subject
# @author: giuliano giari, giuliano.giari@gmail.com
# ======================================================================================================================

function doCoreg () {

    bidsPath="/Volumes/ProjectData/MEGRID/"

    # get fif filename by finding the sub_id in the fif directory; we are using the maxfiltered data, they will thus all have the same head position
    # to make this work double quote should end before * otherwise they will be
    # considered as string instead of wildcard
    fifFile=$(find $bidsPath"derivatives/max/"$1"_"$2*"run-02_proc-tsss_meg.fif")
    
    # we check if this subject has been already processed, i.e. the transformation matrix file is present
    if [[ ! -f  $bidsPath"derivatives/src/"$1"_"$2"_trans.fif" ]]; then
        # if not we process it
        flag="y"
    else # ask wheter to redo the coregistration
        read -n 1 -rep "Do you want to redo the coregistration for "$1" session "$2"? [y/n]" flag
    fi
    
    # check if this subject has to be processed
    if [ "$flag" = "y" ]; then
        # but first check if this subject has the mri
        if [[ ! -d "$SUBJECTS_DIR$1" ]]; then
            # if not we have to reshape fsaverage to match individual's headshape
            # https://www.slideshare.net/mne-python/mnepython-scale-mri
            # open coregistration gui
            mne coreg -s sub-fsaverage -d $SUBJECTS_DIR -f $fifFile

            # compute watershed
            # https://mne.tools/stable/generated/commands.html#mne-watershed-bem
            mne watershed_bem -s $1 -d $SUBJECTS_DIR --overwrite

            # move the files where mne expects them
            files=$(find "$SUBJECTS_DIR$1/bem/watershed/" -name $1* -type f )
            while IFS= read -r file; do
                # get the surface name from the filename
                surface_name=${file#$SUBJECTS_DIR"/"$1"/bem/watershed/"$1"_"}
                # and remove the "_surface" suffix
                surface_name=${surface_name%"_surface"}
                cp -f "${file}" $SUBJECTS_DIR"/"$1"/bem/"${surface_name}".surf"
            done <<< "$files"

        else
            # we do the normal coregistration
            # open coregistration gui
            mne coreg -s $1 -d $SUBJECTS_DIR -f $fifFile
        fi

        # rename to add the session name
        mv $bidsPath"derivatives/src/"$1"-trans.fif" $bidsPath"derivatives/src/"$1"_"$2"_trans.fif"
    fi

}

# ======================================================================================================================

# activate the mne environment https://github.com/conda/conda/issues/7980
# get the conda base path
CONDA_BASE=$(conda info --base)
# source the conda path
source $CONDA_BASE/etc/profile.d/conda.sh
# activate the environment
conda activate gft2

# set paths
export SUBJECTS_DIR="/Volumes/ProjectData/MEGRID/freesurfer/"

# check if fsaverage is present, otherwise download it
if [ ! -d "${SUBJECTS_DIR}sub-fsaverage" ]; then
  python -c "from mne.coreg import create_default_subject; create_default_subject()"
fi

# define the list of subjects that need to be processed
subj_list=$(cut -f1 /Volumes/ProjectData/MEGRID/participants.tsv | tail -n +2)

# run the function
for sub_id in ${subj_list[@]}
    do
    ses_list=$(find /Volumes/Projects/MEGRID/derivatives/log/$sub_id/ -maxdepth 1 -type d -name ses* -exec basename {} \;)
	  for ses_id in ${ses_list[@]}
      do
      if [ $ses_id != "ses-anat" ]; then
        doCoreg $sub_id $ses_id
      fi
      done
done

conda activate base
