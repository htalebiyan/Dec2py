#!/bin/bash
#SBATCH --job-name=Test_INDP
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=ht20@rice.edu
#SBATCH --mail-type=ALL

export GUROBI_HOME="/home/ht20/Gurobi8"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="${GUROBI_HOME}/gurobi.lic"

python /home/ht20/Codes/Run_INDP.py