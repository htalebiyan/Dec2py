#!/bin/bash
#SBATCH --job-name=Auctions_synthetic_Random
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem-per-cpu=2000m
#SBATCH --time=24:00:00
#SBATCH --mail-user=ht20@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --profile=ALL
  
echo "My job ran on:"
echo $SLURM_NODELIST
if [[ -d $SHARED_SCRATCH/$USER && -w $SHARED_SCRATCH/$USER ]]
then
    cd $SHARED_SCRATCH/$USER
    /path/to/myprogram
fi

echo $(cat /proc/cpuinfo)

export GUROBI_HOME="/home/ht20/Gurobi8"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="${GUROBI_HOME}/gurobi.lic"

python /home/ht20/Codes/Run_INDP.py