#!/bin/bash
#SBATCH --job-name=classif-ddpm-2class-30grp-test # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=12:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=class-ddpm%j.out # output file name
#SBATCH --error=class-ddpm%j.err  # error file name
#SBATCH --qos=qos_gpu-t4

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

# python -u main.py --mode train --dataset dataset_rh_2class-jeanzay \
# --labels pipelines    --batch_size 1 --data_dir data --sample_dir samples-2classes \
# --save_dir models-2classes --test_iter 190 --n_classes 2

/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/classifier_conditional_ddpm/main.py \
   --mode transfer --dataset dataset_rh_2class_spm-jeanzay --labels pipelines \
   --batch_size 1 --data_dir data \
   --sample_dir samples-2classes-30grp-spm --save_dir models-2classes-30grp-spm \
   --test_iter 490 --n_classes 2

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/classifier_conditional_ddpm/src/main.py \
#    --mode train --dataset dataset_rh_2class_30grp_spm-jeanzay --labels pipelines \
#    --batch_size 8 --data_dir data --n_classes 2 \
#    --n_epoch 500 --lrate 1e-5 --sample_dir samples-2classes-30grp-spm --save_dir models-2classes-30grp-spm