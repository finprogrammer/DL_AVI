#!/bin/bash
# Parallel Jobs
N_JOBS=10
ARGS="-P$N_JOBS --header :"

# Uncomment this line for dry run
#ARGS="--dry-run "$ARGS

# Experiment parameters
PROJECT='CNN_Experiments'

PROBLEMS=("Screw" "Sheet_Metal_Package" "Winding_Head" "Cable" "Cover")
MAX_EPOCHS=(50)

parallel $ARGS \
    sbatch \
        --job-name=$PROJECT \
        $(echo --export=problem={problem},\
        epochs={max_epochs} | tr -d '[:space:]')\
        run-job.sh \
            ::: max_epochs "${MAX_EPOCHS[@]}" \
            ::: problem "${PROBLEMS[@]}" \
