interactor_type=$1
device=$2

pretraining_dataset=scratch
finetuning_dataset=QM9
model_2d=GIN
# model_2d=Transformer

blocks=8
bsz=$3

# input_model_file=./deep-interact/assets/transformer_schnet_L8_deep_interact_interatomic_bond-angle_edge-pred-50_edge-class_corrected_complete.pth
# input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-centrality_50_e50_l8_bsz1024_pretrain-shallow-discount_2d-classify_dihedrals-spd_as_2_complete.pth
# input_model_file=./deep-interact/assets/spd_as_2-nodiscount.pth
# input_model_file="scratch"
input_model_file="./deep-interact/assets/pretrain_PCQM4Mv2_e50_l8_bsz1024_${interactor_type}_complete.pth"
# input_model_file="./deep-interact/assets/pretrain_QM9_e100_l8_bsz1024_mlp_elemprod_test_complete_final.pth"
output_model_name=finetune_${finetuning_dataset}_from_${pretraining_dataset}_gin_schnet_L${blocks}_test_outer

PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --input_data_dir=data --dataset=QM9 --epochs 1000 --input_model_file $input_model_file --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size $bsz --verbose --device $device --num_workers 12 --mode method --model_3d SchNet --model_2d $model_2d --gnn_type $model_2d --num_interaction_blocks $blocks --output_model_name $output_model_name --dropout_ratio 0.0 --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 --initialization glorot_normal --input_model_file $input_model_file --save_config --verbose --diff_interactor_per_block --task=gap --interactor_type $interactor_type --interactor_activation Swish --residual --decay 0.0  --JK last