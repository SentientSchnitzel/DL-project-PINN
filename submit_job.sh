#!/bin/sh
#BSUB -q gpuv100
#BSUB -J wave_eq_pinn
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1"
#BSUB -M 3GB
### wall-time 8 hr
#BSUB -W 8:00 
#BSUB -u simondanielschneider@gmail.com # change this to your own :)
#BSUB -N

### Doesnt overwrite, just makes nuew. 
#BSUB -o j_outputs/Output_%J.out 
#BSUB -e j_errors/Error_%J.err

source .venv/bin/activate
module load cuda/11.6

### Write which script to run below
python3 train.py