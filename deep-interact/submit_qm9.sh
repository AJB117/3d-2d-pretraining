#!/bin/bash -l

# --- Resource related ---
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=30000
#SBATCH -c 6

# --- Task related ---
#SBATCH --output="/home/zqe3cg/3d-2d-pretraining/logfiles/qm9_r2.log"
#SBATCH --error="/home/zqe3cg/3d-2d-pretraining/logfiles/qm9_r2.err"
#SBATCH --job-name="qm9_r2"

echo "Hostname -> $HOSTNAME"

source /etc/profile.d/modules.sh

source $SCRATCH_DIR/.virtualenvs/3d-pretraining/bin/activate

echo "which python -> $(which python)"
nvidia-smi

echo "STARTIME $i $(date)"

cd ..;
PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --task=r2 --input_data_dir=data --dataset=QM9 --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 1 --num_workers 6 --mode method --model_3d SchNet --model_2d GIN --num_interaction_blocks 6 --output_model_name schnet_gin_6_layers_emb300_mean_mean_cat_cat_lnorm_lre4_interatomic_edge_pred_20_r2 --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 --layer_norm --residual --initialization glorot_normal --input_model_file ./deep-interact/assets/deep_interact_interatomic_edge_pred_complete.pth

echo "ENDTIME $i $(date)"