#!/bin/bash

# This epilog script runs on each node and copies the output from the designated
# output directory to the home directory. It requires some environment variables
# to be set.

if [ ! -d $UTOPIA_CLUSTER_MODE_OUT_DIR ]; then
    echo "$SLURMD_NODENAME: No custom output directory available; nothing to copy."
    exit 0
fi

# -- Parse arguments
sync_mode=${1:-direct}
shift


# -- Try to get a more recent rsync
module try-load rsync


# -- Compile source and target paths
# old, unspecific approach: operates on whole output directory (slow scanning for many files)
# src_dir=$UTOPIA_CLUSTER_MODE_OUT_DIR/
# dest_dir=$UTOPIA_OUTPUT/

# new, more specific approach: reduces number of files that need scanning
model_out_dir=$UTOPIA_CLUSTER_MODE_OUT_DIR/$UTOPIA_MODEL_NAME
run_dir_name=$(ls -1a $model_out_dir/ | grep job${SLURM_JOB_ID})
src_dir=$model_out_dir/$run_dir_name
dest_dir=$UTOPIA_OUTPUT/$UTOPIA_MODEL_NAME/$run_dir_name

echo "$SLURMD_NODENAME: Copying output (sync mode: $sync_mode, using $(which rsync)) ..."
echo "  source:       $src_dir"
echo "  destination:  $dest_dir"
echo ""


# -- Transfer the data
# ... either directly or by first putting it into a tarball
if [ $sync_mode == "direct" ]; then
    rsync -ahz --info=progress2 --info=name0 --info=stats2 $src_dir/* $dest_dir

elif [ $sync_mode == "archive" ]; then
    # Locally build an archive from the source output directory
    tarball_dir=$TMPDIR/_tmp
    mkdir -p $tarball_dir
    tarball_name=output_from_node_${SLURMD_NODENAME}.tar  # NOTE If changing this name, adjust unpacking script!
    tarball_path=$tarball_dir/$tarball_name

    echo "$SLURMD_NODENAME: Creating tarball at $tarball_path ..."
    tar -c -f $tarball_path -C $src_dir .

    # Now transfer it
    echo "$SLURMD_NODENAME: Transferring tarball ..."
    mkdir -p $dest_dir
    rsync -ah --info=progress2 --info=name0 --info=stats2 $tarball_dir/* $dest_dir/

else
    echo "Invalid sync mode: ${sync_mode}! Possible values: direct, archive"
    exit 1
fi

echo "$SLURMD_NODENAME: Finished copying output."
