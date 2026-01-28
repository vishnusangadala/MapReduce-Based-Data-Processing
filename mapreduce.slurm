#!/bin/bash
#SBATCH --job-name=mapreduce_p1
#SBATCH --partition=short          # change ONLY if 'short' is not valid on your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/AMS598/class2025/Sangadala_Vishnu/project1/mapreduce_%j.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Sangadala_Vishnu/project1/mapreduce_%j.err

# -------------------- ENV --------------------
module purge
module load python/3.10 || module load anaconda/2023.09
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -------------------- PATHS ------------------
BASE=/gpfs/projects/AMS598/class2025/Sangadala_Vishnu/project1
INPUT=/gpfs/projects/AMS598/projects2025_data/project1_data
INTERMEDIATE=$BASE/intermediate
OUTPUT=$BASE/output/final_counts.tsv
PYTHON_FILE=$BASE/hpc_mapreduce_count.py

mkdir -p "$INTERMEDIATE" "$BASE/output"
cd "$BASE"

# -------------------- RUN --------------------
echo "Job started on $(hostname) at $(date)"
echo "Input:        $INPUT"
echo "Intermediate: $INTERMEDIATE"
echo "Output:       $OUTPUT"

srun python3 "$PYTHON_FILE" \
    --input "$INPUT" \
    --work "$INTERMEDIATE" \
    --output "$OUTPUT" \
    --mappers 4 \
    --reducers 4

echo "Job finished at $(date)"
