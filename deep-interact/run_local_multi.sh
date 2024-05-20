cd ..;
schedulers=( CosineAnnealingLR )
# schedulers=( ReduceLROnPlateau CosineAnnealingLR CosineAnnealingWarmRestarts )

if [[ $1 == "PCQM4Mv2" || $1 == "PCQM4Mv2-pretraining-centrality" ]]; then
    input_data_dir=/home/psoga/Documents/research/3d-2d-pretraining/data
    dataset=$1
elif [[ $1 == "QM9" ]]; then
    input_data_dir=data/molecule_datasets
    dataset=QM9
elif [[ $1 == "QM9_dihedral_spd" ]]; then
    input_data_dir=/home/psoga/Documents/research/3d-2d-pretraining/data/
    dataset=QM9_dihedral_spd
fi

batch_size=$2
pretrain_2d_tasks="interatomic_dist bond_angle dihedral_angle"
pretrain_3d_tasks="edge_classification spd centrality_ranking"
# pretrain_3d_tasks="spd edge_classification betweenness_ranking"

pretrain_2d_balances="1 1 1"
pretrain_3d_balances="1 1 1"

epochs=$3

# input_model_file=./deep-interact/assets/pretrain_QM9_50_e50_l8_bsz1024_rep_bond-mix_embs-tanh_dihedral-shallow_predict-swish-interatomic-bangle-dangle-spd-edgeclass-centrality-247-diff_interactor_complete_complete.pth
# input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-centrality_e50_l8_bsz1024_Swish_mixembs-interatomic-bangle-dangle-247-diff_interactor_complete.pth
# input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-centrality_e50_l8_bsz1024_Swish_tanh_dihedral-interatomic-bangle-dangle-247-diff_interactor_complete.pth
# input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-centrality_e50_l8_bsz1024_Swish_rep_bond-interatomic-bangle-dangle-247-diff_interactor_complete.pth
# input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-centrality_e100_l8_bsz1024_Swish_losses_at_end-interatomic-bangle-dangle-spd-edgeclass-centrality-247-diff_interactor_complete.pth
# input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-betweenness_e50_l8_bsz1024_Swish_interatomic-bangle-dangle-spd-edgeclass-betweenness-247-diff_interactor_complete.pth
blocks=8

for i in "${!schedulers[@]}";
do
    PYTHONPATH='.' python3 runners/pretrain_multi_interact.py --input_data_dir=$input_data_dir --dataset=$dataset --model_3d=SchNet --model_2d GIN --epochs $epochs --output_model_dir ./deep-interact/assets --lr_scheduler ${schedulers[$i]} --batch_size $batch_size --verbose --device 0 --mode method --num_workers 4 --process_num $i --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --emb_dim 300 --num_interaction_blocks $blocks --output_model_name pretrain_${dataset}_${epochs}_e50_l8_bsz1024_step_5-7 \
    --pretrain_2d_tasks $pretrain_2d_tasks --pretrain_3d_tasks $pretrain_3d_tasks --verbose \
    --task gap \
    --diff_interactor_per_block \
    --pretrain_3d_task_indices 2 5 7 --pretrain_2d_task_indices 2 5 7 \
    --residual \
    --pretrain_2d_balances 1.0 1.0 1.0 --pretrain_3d_balances 1.0 1.0 1.0 \
    --interactor_activation Swish --classify_dihedrals --use_shallow_predictors \
    --loss_pattern "A AB ABC"
done
    # --mix_embs_pretrain
    # --use_tanh_dihedral
