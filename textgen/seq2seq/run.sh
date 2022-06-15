# train
python train.py --learning_rate=3e-5 --do_train --do_predict --evaluation_strategy steps --predict_with_generate --n_val 10 --data_dir data --output_dir zh_mbart --model_name_or_path sshleifer/tiny-mbart --tgt_lang zh_CN --src_lang zh_CN

# predict
python run_eval.py zh_mbart data/val.source mbart_val_generate_result.txt --reference_path data/val.target --score_path rouge.json --task summarization --device cuda --bs 32