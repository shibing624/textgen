#torchrun --nproc_per_node 4 finetune_searchad.py \
/apdcephfs_cq3/share_2973545/data/env/py38/bin/python3 -m torch.distributed.launch --nproc_per_node=4 finetune_searchad.py \
  --num_epochs 3 \
  --batch_size 8 \
  --model_type llama \
  --model_name /apdcephfs_cq3/share_2973545/data/models/shibing624/chinese-alpaca-plus-7b-hf \
  --do_train \
  --do_predict \
  --output_dir /apdcephfs/private_curvasong/ad_llm/llama_sogou_ad/output \
  
