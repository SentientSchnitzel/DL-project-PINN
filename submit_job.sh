#!/bin/sh 

# Capture the exp ID passed as an argument
# exp_id=$1

#BSUB -q gpuv100
#BSUB -J wave_eq_pinn
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1"
#BSUB -M 3GB
### wall-time 2 hr
#BSUB -W 8:00 
#BSUB -u s194055@student.dtu.dk # change this to your own :)
#BSUB -N

### Doesnt overwrite, just makes nuew. 
#BSUB -o j_outputs/Output_%J.out 
#BSUB -e j_errors/Error_%J.err

### run shell script to activate virtual environment
source act_venv.sh

### Write which script to run below
python3 wave_eq_ffn.py