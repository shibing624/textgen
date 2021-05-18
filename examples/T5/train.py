import logging

import pandas as pd
import torch
import sys

sys.path.append('../..')
from textgen.t5 import T5Model, T5Args

use_cuda = torch.cuda.is_available()

train_data = [
    ["one", "1"],
    ["two", "2"],
]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = [
    ["three", "3"],
    ["four", "4"],
]

eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = T5Args()
model_args.max_seq_length = 196
model_args.train_batch_size = 8
model_args.eval_batch_size = 8
model_args.num_train_epochs = 1
model_args.evaluate_during_training = False
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_return_sequences = 1
# model_args.wandb_project = "MT5 mixed tasks"

model = T5Model("mt5", "google/mt5-base", args=model_args, use_cuda=use_cuda)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Optional: Evaluate the model. We'll test it properly anyway.
results = model.eval_model(eval_df, verbose=True)
print(results)


print(model.predict(["four", "five"]))