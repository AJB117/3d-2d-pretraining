cd ..;

PYTHONPATH='.' python runners/finetune_QM9_deep_interact.py --task=gap --input_data_dir=/home/patrick/3d-2d-pretraining/data/molecule_datasets --dataset=QM9 --epochs 1000 --output_model_dir ./deep-interact/assets  --lr_scheduler CosineAnnealingLR --batch_size 128 --verbose --device 0 --num_workers 4 --mode SchNet --model_3d SchNet --output_model_name SchNet --emb_dim 300 --SchNet_num_filters 300 --interaction_rep_3d "";
