#!/bin/bash
source /miniconda/etc/profile.d/conda.sh
conda activate base
exec "$@"
