pretraining_dataset=PCQM4Mv2
finetuning_dataset=MoleculeNet

input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2_50_L8_deep_interact_interatomic-25_bond-angle_edge-pred-50_edge-class_complete.pth
output_model_name=finetune_${finetuning_dataset}_from_${pretraining_dataset}_transformer_50_L8_deep_interact_interatomic-25_bond-angle_edge-pred-50_edge-class_complete
blocks=8
bsz=64

PYTHONPATH='.' python3 generate_base_config.py --input_data_dir=/scratch/zqe3cg/data/molecule_datasets/MoleculeNet --dataset=MoleculeNet --epochs 100 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size $bsz --no_verbose --device 0 --num_workers 8 --use_2d_only --model_2d Transformer --gnn_type Transformer --num_interaction_blocks 8 --output_model_name $output_model_name --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --dropout_ratio 0 --final_pool cat --mode 2d --emb_dim 300 --lr 1e-4 --layer_norm --initialization glorot_normal --input_model_file $input_model_file --save_config --residual --no_verbose \
| (read p1; python3 generate_configs.py --config_file_base $p1 --hparam dataset --hparam_choices bace sider muv tox21 hiv toxcast bbbp clintox --experiment_name $output_model_name) \
| (read p2; python3 generate_scripts.py --sweep_dir $p2)