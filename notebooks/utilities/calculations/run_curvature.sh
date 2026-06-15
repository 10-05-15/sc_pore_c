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

my_job_header

mkdir -p logs

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/indikar_root/indikar1/jduhamel/conda-envs/pore_c

mkdir -p outputs

INPUT_DIR="/nfs/turbo/umms-indikar/shared/projects/poreC/pipeline_outputs/higher_order/by_chromosome"
PYTHON="/scratch/indikar_root/indikar1/jduhamel/conda-envs/pore_c/bin/python"

for chr in {1..19} X; do
    INPUT="${INPUT_DIR}/singlecell_mESC_1000000_chr${chr}.h5ad"
    if [[ ! -f "$INPUT" ]]; then
        echo "Skipping chr${chr}: $INPUT not found"
        continue
    fi
    EDGE_OUT="outputs/singlecell_edges_curvature_chr_${chr}.csv"
    NODE_OUT="outputs/singlecell_nodes_curvature_chr_${chr}.csv"
    echo "Processing chr${chr}..."
    "$PYTHON" curvature.py "$INPUT" "$EDGE_OUT" "$NODE_OUT"
done