# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: adjust for chinese tokenizer
"""
import os
import pickle
from multiprocessing import Pool

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from rouge import Rouge
from loguru import logger


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    if args.preprocess_inputs:
        return tokenizer.prepare_seq2seq_batch(
            src_texts=[
                prefix + ": " + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ],
            tgt_texts=dataset["target_text"],
            max_length=args.max_seq_length,
            max_target_length=args.max_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
    else:
        return tokenizer.prepare_seq2seq_batch(
            src_texts=[
                prefix + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ],
            tgt_texts=dataset["target_text"],
            max_length=args.max_seq_length,
            max_target_length=args.max_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )


def load_hf_dataset(data, tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=True,
    )

    dataset.set_format(type="pt", columns=["input_ids", "attention_mask"])

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_data(data):
    prefix, input_text, target_text, tokenizer, args = data

    # Add EOS again if truncated?
    if args.preprocess_inputs:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[prefix + ": " + input_text],
            tgt_texts=[target_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    else:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[prefix + ": " + input_text],
            tgt_texts=[target_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    return (input_ids, attention_mask, labels)


class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s" % args.cache_dir)

            data = [
                (prefix, input_text, target_text, tokenizer, args)
                for prefix, input_text, target_text in zip(
                    data["prefix"], data["input_text"], data["target_text"]
                )
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def dynamic_lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def f1_sim(text_a, text_b):
    """F1相似度
    说明：算出两个文本的最长公共子序列长度，然后乘2并处以两者
    长度之和。
    脚本见：https://github.com/CLUEbenchmark/pCLUE/blob/main/evaluate_pclue.py
    计算pCLUE任务总分，及子分数
    """
    if not text_a and not text_b:
        return 0.
    lcs_len = dynamic_lcs(text_a, text_b)
    return 2. * lcs_len / (len(text_a) + len(text_b))


def rouge_l_zh(target, pred):
    """计算Rouge-l得分，Rouge-l指标常用于评估自动文本摘要及翻译任务
    target: 真实标签
    pred: 预测标签"""

    if not (isinstance(target, str) or isinstance(pred, str)):
        logger.error("target或pred为非字符串！请检查!")
        return 0
    rouge = Rouge()
    scores = rouge.get_scores(" ".join(list(pred)), " ".join(list(target)))
    score = scores[0]["rouge-l"]
    return score["f"]


if __name__ == '__main__':
    a = '123444'
    b = '23411'
    print(f1_sim(a, b))
    print(dynamic_lcs(a, b))
