#!/bin/bash -l

# --- Resource related ---
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=30000
#SBATCH -c 8

# --- Task related ---
#SBATCH --output="/home/zqe3cg/3d-2d-pretraining/logfiles/pcqm4mv2_l8_50.log"
#SBATCH --error="/home/zqe3cg/3d-2d-pretraining/logfiles/pcqm4mv2_l8_50.err"
#SBATCH --job-name="pcqm4mv2_l8_50"

echo "Hostname -> $HOSTNAME"

source /etc/profile.d/modules.sh

source $SCRATCH_DIR/.virtualenvs/3d-pretraining/bin/activate

echo "which python -> $(which python)"
nvidia-smi

echo "STARTIME $i $(date)"

cd ..;
schedulers=( CosineAnnealingLR )
# schedulers=( ReduceLROnPlateau CosineAnnealingLR CosineAnnealingWarmRestarts )

if [[ $1 == "PCQM4Mv2" ]]; then
    input_data_dir=$SCRATCH_DIR/data
    dataset=PCQM4Mv2
    batch_size=512
else
    input_data_dir=$SCRATCH_DIR/data/molecule_datasets
    dataset=QM9
    batch_size=128
fi

epochs=$2

for i in "${!schedulers[@]}";
do
    PYTHONPATH='.' python runners/pretrain_deep_interact.py --input_data_dir=$input_data_dir --dataset=$dataset --model_3d=SchNet --model_2d GIN --epochs $epochs --output_model_dir ./deep-interact/assets --lr_scheduler ${schedulers[$i]} --batch_size $batch_size --verbose --device 0 --mode method --num_workers 8 --process_num $i --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --emb_dim 300 --layer_norm --residual --num_interaction_blocks 6 --output_model_name pretrain_${dataset}_${epochs}_L6_deep_interact_interatomic-25_bond-angle_edge-pred-50_edge-class --pretrain_2d_tasks interatomic_dist bond_angle --pretrain_3d_tasks edge_existence edge_classification --wandb --no_verbose --pretrain_interatomic_samples 25 --pretrain_neg_link_samples 50
done
# edge_pred_# where # is the number of sampled negative links
# interatomic vs interatomic_# where # is the number of sampled distances

# wait

python3 aggregate.py
rm *_config.csv

echo "ENDTIME $i $(date)"