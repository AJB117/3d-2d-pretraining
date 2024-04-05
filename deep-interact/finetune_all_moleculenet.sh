datasets=( bbbp clintox hiv muv tox21 toxcast sider bace )

# for i in "${!schedulers[@]}";
# do
cd ..;
for i in "${!datasets[@]}";
do
  PYTHONPATH='.' python3 runners/finetune_MoleculeNet_deep_interact.py --input_data_dir=~/data/molecule_datasets/MoleculeNet --dataset=${datasets[i]} --epochs 200 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 64 --verbose --device $((i % 6)) --num_workers 4 --model_3d SchNet --model_2d GIN --num_interaction_blocks 6 --output_model_name gin_6_layers_emb300_mean_mean_cat_cat_lnorm_lr5e4_b64_interatomic_bond-angle_edge-pred-50_edge-class_tune_${datasets[i]} --interaction_rep_2d mean --interaction_rep_3d mean --interaction_agg cat --final_pool cat --mode method --emb_dim 300 --lr 5e-4 --layer_norm --residual --initialization glorot_normal --input_model_file ./deep-interact/assets/deep_interact_interatomic_bond-angle_edge-pred-50_edge-class_complete.pth --wandb --use_2d_only 1> /dev/null &
done

