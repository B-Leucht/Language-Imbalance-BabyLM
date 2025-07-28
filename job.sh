#!/bin/bash
#SBATCH --job-name=babylm
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH --gpus=1
 
module load 2023
module load JupyterNotebook/7.0.2-GCCcore-12.3.0

python -m venv babyenv
source babyenv/bin/activate

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers@v4.53.0 datasets accelerate dotenv

PORT=`shuf -i 5000-5999 -n 1`
LOGIN_HOST=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
BATCH_HOST=$(hostname)
 
echo "To connect to the notebook type the following command from your local terminal:"
echo "ssh -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"
echo
echo "After connection is established in your local browser go to the address:"
echo "http://localhost:${PORT}"
 
jupyter notebook --no-browser --port $PORT

