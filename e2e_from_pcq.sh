pretraining_dataset=PCQM4Mv2
finetuning_dataset=QM9

input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2_50_L8_deep_interact_interatomic-25_bond-angle_edge-pred-50_edge-class_complete.pth
output_model_name=finetune_${finetuning_dataset}_from_${pretraining_dataset}_50_L8_deep_interact_interatomic-25_bond-angle_edge-pred-50_edge-class_complete
blocks=8
bsz=128

PYTHONPATH='.' python3 generate_base_config.py --input_data_dir=data --dataset=QM9 --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size $bsz --verbose --device 0 --num_workers 8 --mode method --model_3d SchNet --model_2d GIN --num_interaction_blocks $blocks --output_model_name ${output_model_name}.pth --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 --layer_norm --residual --initialization glorot_normal --input_model_file $input_model_file --save_config --no_verbose \
| (read p1; python3 generate_configs.py --config_file_base $p1 --experiment_name $output_model_name) \
| (read p2; python3 generate_scripts.py --sweep_dir $p2)
