# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import json
import os

from tqdm.auto import tqdm
from textgen.question_answering import QuestionAnsweringModel

# Create dummy data to use for training.
train_data = [
    {
        "context": "This is the first context",
        "qas": [
            {
                "id": "00001",
                "is_impossible": False,
                "question": "Which context is this?",
                "answers": [{"text": "the first", "answer_start": 8}],
            }
        ],
    },
    {
        "context": "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty "
                   "prohibiting the hunting of right and gray whales,"
                   " and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species "
                   "were relatively rare—and little opposition was raised",
        "qas": [
            {
                "id": "00002",
                "is_impossible": False,
                "question": "What was the cost to society?",
                "answers": [{"text": "low cost", "answer_start": 225}],
            },
            {
                "id": "00003",
                "is_impossible": False,
                "question": "What was the name of the 1937 treaty?",
                "answers": [{"text": "Bald Eagle Protection Act", "answer_start": 167}],
            },
            {"id": "00004", "is_impossible": True, "question": "How did Alexandar Hamilton die?", "answers": [], },
        ],
    },
]  # noqa: ignore flake8"


# Save as a JSON file
os.makedirs("data", exist_ok=True)
with open("data/train.json", "w") as f:
    json.dump(train_data, f)


print("data processed.")

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    # "evaluate_during_training": False,
    # "evaluate_during_training_steps": 10000,
    # "train_batch_size": 1,
    # "num_train_epochs": 1,
    # 'wandb_project': 'test-new-project',
    # "use_early_stopping": True,
    # "n_best_size": 3,
    # "fp16": False,
    # "no_save": True,
    # "manual_seed": 4,
    # "max_seq_length": 128,
    # "lazy_loading": True,
    # "use_multiprocessing": True,
}

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel("bert", "bert-base-cased", args=train_args,
                               use_cuda=False
                               # use_cuda=True, cuda_device=0
                               )

# Train the model with JSON file
model.train_model("data/train.json")
# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, text = model.eval_model("data/train.json")

print(result)
print(text)

print("-------------------")
# Making predictions using the model.
to_predict = [
    {
        "context": "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,\
            and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
        "qas": [{"question": "What was the name of the 1937 treaty?", "id": "0"}],
    }
]

print(model.predict(to_predict, n_best_size=3))
