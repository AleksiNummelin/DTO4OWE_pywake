#!/bin/bash
#SBATCH --account=project_2019907
#SBATCH --time=18:00:00
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=hugemem
#SBATCH --cpus-per-task=1
#SBATCH --job-name=pywake1

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

module load python-geo
cd /users/nummelin/Projects/DTO4OWE

# python pywake_DTO4OWE_ClimateDT.py year month day
python pywake_DTO4OWE_ClimateDT.py 2012 1 0
#done
