cd ..;
PYTHONPATH='.' python runners/finetune_PCQM4Mv2.py --input_data_dir=/home/patrick/3d-2d-pretraining/data --dataset=PCQM4Mv2 --task=gap --model_3d=EGNN --epochs 200 --output_model_dir ./egnn_pcqm4mv2/assets --lr_scheduler ReduceLROnPlateau --batch_size 128 --verbose --device 0 --mode EGNN --require_3d --num_workers 6

