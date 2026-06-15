#!/bin/bash
#SBATCH --job-name=curvature_sc_chr
#SBATCH --account=indikar0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/curvature_%j.out
#SBATCH --error=logs/curvature_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jduhamel@umich.edu

mkdir -p logs

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /home/jduhamel/.conda/envs/pore_c

INPUT="/scratch/indikar_root/indikar1/jduhamel/population_mESC_1000000_features_inter.h5ad"
EDGE_OUT="outputs/edges_curvature_inter_only.csv"
NODE_OUT="outputs/nodes_curvature_inter_only.csv"

mkdir -p outputs

/home/jduhamel/.conda/envs/pore_c/bin/python curvature.py \
    "$INPUT" \
    "$EDGE_OUT" \
    "$NODE_OUT" \