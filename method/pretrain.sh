cd ..;
PYTHONPATH='.' python runners/pretrain.py --input_data_dir=/home/patrick/3d-2d-pretraining/data --dataset=QM9 --task=gap --model_3d=EGNN --epochs 1 --output_model_dir ./egnn_qm9/assets --lr_scheduler ReduceLROnPlateau --batch_size 128 --verbose --device 1 --mode EGNN --require_3d --num_workers 4

