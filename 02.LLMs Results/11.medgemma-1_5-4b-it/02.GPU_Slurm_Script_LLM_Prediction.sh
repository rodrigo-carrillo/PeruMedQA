#!/bin/bash
#SBATCH --job-name=Examenes_Resi                       # Name of the job
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --partition=l40s-8-gm384-c192-m1536            # Partition name a100-8-gm320-c96-m1152
#SBATCH --output=output_file_2.out                     # Name of the output file
#SBATCH --error=error_file_2.err                       # Name of the error file
#SBATCH --gpus=2                                       # Enter no.of gpus needed
#SBATCH --mem=48G                                      # Memory Needed
#SBATCH --cpus-per-gpu=32
#SBATCH --time=165:00:00                               # Time requested
#SBATCH --mail-type=begin                              # send mail when job begins
#SBATCH --mail-type=end                                # send mail when job ends
#SBATCH --mail-type=fail                               # send mail if job fails
#SBATCH --mail-user=rmcarri@emory.edu                  # Replace mail

export TRANSFORMERS_CACHE=/scratch/rmcarri/Examenes_Residentado_Peru_medgemma-4b-it/huggingface
export HF_HOME=/scratch/rmcarri/Examenes_Residentado_Peru_medgemma-4b-it/huggingface
export HF_DATASETS_CACHE=/scratch/rmcarri/Examenes_Residentado_Peru_medgemma-4b-it/huggingface
export XDG_CACHE_HOME=/scratch/rmcarri/Examenes_Residentado_Peru_medgemma-4b-it/.cache

export TMPDIR=/scratch/rmcarri/Examenes_Residentado_Peru_medgemma-4b-it/tmp
mkdir -p $TMPDIR

conda init bash > /dev/null 2>&1
source ~/.bashrc
conda activate LLMs_HuggingFace   # Activate conda environment

echo "TMPDIR = $TMPDIR"
echo "HF_HOME = $HF_HOME"
python3 -c "import tempfile; print('Python tempdir:', tempfile.gettempdir())"
nvidia-smi --query-gpu=timestamp,index,name,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv -l 5 > gpu_log.csv &
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python3 LLMs_Answers.py