#!/bin/bash
#SBATCH --job-name=expand_sc
#SBATCH --account=indikar0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=151G
#SBATCH --time=24:00:00
#SBATCH --output=logs_expand/curvature_%j.out
#SBATCH --error=logs_expand/curvature_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jduhamel@umich.edu

my_job_header

mkdir -p logs_expand

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/indikar_root/indikar1/jduhamel/conda-envs/pore_c

mkdir -p outputs_expand

PYTHON="/scratch/indikar_root/indikar1/jduhamel/conda-envs/pore_c/bin/python"

"$PYTHON" gather_matrix.py

echo "Complete"