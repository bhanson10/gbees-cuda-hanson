#!/bin/bash
#SBATCH --job-name="gbees"
#SBATCH --output="./outputs/gbees.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --ntasks=1      
#SBATCH --nodes=1               
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=184G
#SBATCH --account=csd916
#SBATCH --no-requeue
#SBATCH -t 12:00:00

module purge
module load gpu/0.15.4
module load cuda/11.0.2
module load slurm

make clean
make
./GBEES