pretraining_dataset=PCQM4Mv2
finetuning_dataset=QM9

blocks=8
bsz=128
model_2d=GIN


input_model_file=./deep-interact/assets/pretrain_PCQM4Mv2-pretraining-centrality_e50_l8_bsz1024_pretrain-shallow-discount_2d-classify_dihedrals_complete.pth
output_model_name=finetune_${finetuning_dataset}_from_${pretraining_dataset}_gin_schnet_L${blocks}_e50_l8_bsz1024_pretrain-shallow-discount_2d-classify_dihedrals_complete

tasks=( gap cv alpha mu r2 u298 g298 u0 homo lumo zpve h298 )

for i in "${!tasks[@]}";
do
	PYTHONPATH='.' python3 runners/finetune_QM9_deep_interact.py --input_data_dir=data --dataset=QM9 --epochs 1000 --input_model_file $input_model_file --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size $bsz --device $((i % 3)) --num_workers 8 --model_3d SchNet --model_2d $model_2d --gnn_type $model_2d --num_interaction_blocks $blocks --output_model_name $output_model_name --dropout_ratio 0 --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 1e-4 --initialization glorot_normal --input_model_file $input_model_file --save_config --diff_interactor_per_block --task=${tasks[i]} --residual --interactor_activation Swish  \
	    --pretrain_3d_task_indices 2 4 7 --pretrain_2d_task_indices 2 4 7 --interact_every_block  1> ${output_model_name}_${tasks[i]}.log 2> ${output_model_name}_${tasks[i]}.err & \
done
