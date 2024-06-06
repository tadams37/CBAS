# model.py - Create CBAS job script
import os
import sys
import pickle
import logging

from gui import view
from gui.config import SETTINGS_FILE, SCRIPT_NAME

model = sys.modules[__name__]


def start():
    """Prep model."""
    logging.info('Model start')


def get_settings():
    """Read settings file, if available, ensure paths exist."""

    if os.path.isfile(SETTINGS_FILE):

        try:

            with open(SETTINGS_FILE, "rb") as file:
                cbaero_path, tables_path, run_path = pickle.load(file)

            if isinstance(cbaero_path, str) and not os.path.exists(cbaero_path):
                cbaero_path = None

            if isinstance(tables_path, str) and not os.path.exists(tables_path):
                tables_path = None

            if isinstance(run_path, str) and not os.path.exists(run_path):
                run_path = None

        except Exception:
            logging.error('ERROR: Unable to save settings to file!')
    else:
        cbaero_path = None
        tables_path = None
        run_path = None

    return (cbaero_path, tables_path, run_path)


def save_settings():
    """Save new settings to file."""
    try:
        with open(SETTINGS_FILE, "wb") as file:
            logging.info(f'Saving new settings to "{SETTINGS_FILE}"')
            pickle.dump([view.cbaero_path.selected,
                         view.tables_path.selected,
                         view.run_path.selected],
                        file)
    except Exception:
        logging.error('ERROR: Unable to save settings to file!')


def generate_job_script():
    "Write a triple-quoted (multi-line) f-string, with param values inserted, to a new shell script file."""

    array_length = len(range(int(view.train_pts_start.value),
                             int(view.train_pts_stop.value),
                             int(view.train_pts_step.value)))

    with open(model.SCRIPT_NAME, "w") as file:
        logging.info(f'Writing job script to "{SCRIPT_NAME}"...')
        file.write(f"""
#!/bin/sh
#SBATCH --array=0-{array_length}
#SBATCH --cpus-per-task={view.job_cpus.value}
#SBATCH --job-name=cbas-start
#SBATCH --output=job-start.out
#SBATCH --error=job-start.err
#SBATCH --account={view.job_queue.value}
#SBATCH --time={view.job_days.value}-{view.job_hrs.value}:{view.job_mins.value}:00

# Points (start stop step) - must match "--array=0-<nun>" above
export POINTS="{view.train_pts_start.value} {view.train_pts_stop.value} {view.train_pts_step.value}"

# NOTE Changes lines below as needed to use correct environment
module load anaconda
source activate cbaero

# Threads per run - must match "--cpus-per-task=<num>" above
export OMP_NUM_THREADS={view.job_cpus.value}
export MKL_NUM_THREADS={view.job_cpus.value}

# Set locations of Python script and CBAero
REPO="$HOME/CBAero_Surrogate"
CBAERO="$HOME/CBAero_Surrogate/cbaero"

# Add appropriate CBAero executable to path
PATH="$CBAERO/cbaero.5.3.7_binaries/bin.linux.redhat.7.9:$PATH"

# Set location of tables
export CBAERO_TABLES="$CBAERO/tables"

# Model file
MODEL="{view.model_path.value}"

# Define parameters
PARAMS="--min_alpha={view.a_min_txt.value} --max_alpha={view.a_max_txt.value} --min_mach={view.m_min_txt.value} --max_mach={view.m_max_txt.value} --min_q={view.q_min_txt.value} --max_q={view.q_max_txt.value}"  # noqa E501

# File options
FILE_OPTS="--cokrig_file={view.cokrig_path.value} --save_as={view.save_fname.value}"

# Call CBAero to create OTIS files for one set of points
python $REPO/cbaerosurrogate.py $MODEL $POINTS $PARAMS $FILE_OPTS --run_type=pstart --task_id=$SLURM_ARRAY_TASK_ID

# Last run in job array? Create new job to build the model using all OTIS files (skipping CBAero)
# Note: new job will wait for this job array to complete.
if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
    sbatch \
    --dependency=afterok:$SLURM_JOB_ID \
    --cpus-per-task={view.job_cpus.value} \
    --job-name=cbaero-end \
    --output=$SLURM_SUBMIT_DIR/job-end.out --error=$SLURM_SUBMIT_DIR/job-end.err \
    --account={view.job_queue.value} \
    --time=0-00:20:00 \
    --wrap="python $REPO/cbaerosurrogate.py $MODEL $POINTS $PARAMS $FILE_OPTS --run_type=pend"
fi

# Record elapsed time
sacct --format="Elapsed" -j $SLURM_JOB_ID
""")

        logging.info('Done.')
