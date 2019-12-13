#!/bin/bash
#SBATCH --job-name=Test_INDP
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:30:00
#SBATCH --mail-user=ht20@rice.edu
#SBATCH --mail-type=ALL
 
srun Run_INDP