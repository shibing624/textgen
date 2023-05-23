/apdcephfs_cq3/share_2973545/data/env/py38/bin/python3 training_chatglm.py \
  --max_seq_length 823 \
  --max_length 175 \
  --num_epochs 3 \
  --train_file /apdcephfs_cq3/share_2973545/curvasong/dataset/title_rewrite/online_service/all_data.txt \
  --model_name /apdcephfs_cq3/share_2973545/data/models/THUDM-chatglm-6b \
  --do_train \
  --output_dir /apdcephfs_cq3/share_2973545/curvasong/ad_rewrite/finetune_output/online_service \
  
