#!/bin/sh
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cbas-start
#SBATCH --output=job-start.out
#SBATCH --error=job-start.err
#SBATCH --account=standby
#SBATCH --time=0-1:0:00

# Points (start stop step) - must match "--array=0-<nun>" above
export POINTS="15 17 1"

# NOTE Changes lines below as needed to use correct environment
module load anaconda
source activate cbas

# Threads per run - must match "--cpus-per-task=<num>" above
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Set locations of Python script and CBAero
REPO="$HOME/CBAero_Surrogate"
CBAERO="$HOME/CBAero_Surrogate/cbaero"

# Add appropriate CBAero executable to path
PATH="$CBAERO/cbaero.5.3.7_binaries/bin.linux.redhat.7.9:$PATH"

# Set location of tables
export CBAERO_TABLES="$CBAERO/tables"

# Model file
MODEL="/home/rcampbel/CBAS/cbaero/models/HL20_Models/hl20/hl20.cbaero"

# Define parameters
PARAMS="--min_alpha=0.0 --max_alpha=35.0 --min_mach=1.2 --max_mach=22.0 --min_q=0.0035 --max_q=10.0"  # noqa E501

# File options
FILE_OPTS=" --save_as=results.out "

# Call CBAero to create OTIS files for one set of points
python $REPO/cbaerosurrogate.py $MODEL $POINTS $PARAMS $FILE_OPTS --run_type=pstart --task_id=$SLURM_ARRAY_TASK_ID

# Last run in job array? Create new job to build the model using all OTIS files (skipping CBAero)
# Note: new job will wait for this job array to complete.
if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
    sbatch     --dependency=afterok:$SLURM_JOB_ID     --cpus-per-task=4     --job-name=cbaero-end     --output=$SLURM_SUBMIT_DIR/job-end.out --error=$SLURM_SUBMIT_DIR/job-end.err     --account=standby     --time=0-00:20:00     --wrap="python $REPO/cbaerosurrogate.py $MODEL $POINTS $PARAMS $FILE_OPTS --run_type=pend"
fi

# Record elapsed time
sacct --format="Elapsed" -j $SLURM_JOB_ID
