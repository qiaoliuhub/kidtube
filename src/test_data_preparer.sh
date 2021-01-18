#! /bin/bash

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false MASTER_ADDR=localhost \
python3 data_preparer.py \
--video_details_dir ../Data/video_details_csv \
--video_transcript_dir ../Data/video_caption_pickle \
--gpus 1 --max_epochs 10 --accelerator ddp --num_sanity_val_steps 0
# --num_workers 1