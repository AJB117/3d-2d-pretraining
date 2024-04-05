#!/bin/bash -l

# --- Resource related ---
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=30000
#SBATCH -c 6

# --- Task related ---
#SBATCH --output="/home/zqe3cg/3d-2d-pretraining/logfiles/qm9_l10.log"
#SBATCH --error="/home/zqe3cg/3d-2d-pretraining/logfiles/qm9_l10.err"
#SBATCH --job-name="qm9_l10"

# targets=( mu alpha homo lumo r2 zpve u0 u298 h298 g298 cv gap_02 )
echo "Hostname -> $HOSTNAME"

source /etc/profile.d/modules.sh

source $SCRATCH_DIR/.virtualenvs/3d-pretraining/bin/activate

echo "which python -> $(which python)"
nvidia-smi

echo "STARTIME $i $(date)"

PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --task=gap --input_data_dir=data --dataset=QM9 --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 1 --num_workers 8 --mode method --model_3d SchNet --model_2d GIN --num_interaction_blocks 6 --output_model_name schnet_gin_6_layers_emb300_mean_mean_cat_cat_lnorm_lre4_interatomic_edge_pred_20_gap --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 --layer_norm --residual --initialization glorot_normal --input_model_file ./deep-interact/assets/deep_interact_interatomic_edge_pred_complete.pth

# done

echo "ENDTIME $i $(date)"
# PYTHONPATH='.' python runners/finetune_QM9_deep_interact.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 0 --num_workers 8 --mode method --model_3d SchNet --model_2d GIN --num_interaction_blocks 6 --output_model_name deep_interact_schnet_gin_6_layers
# done

# method
# PYTHONPATH='.' python runners/finetune_QM9.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --use_2d --epochs 1 --output_model_dir ./method/assets  --input_model_file ./method/assets/model_complete_use.pth --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 1 --num_workers 4 --mode method --use_3d --model_3d EGNN

# baseline egnn
# PYTHONPATH='.' python runners/finetune_QM9.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --use_2d --epochs 1 --output_model_dir ./method/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 1 --num_workers 4 --mode baseline --use_3d --model_3d EGNN
