# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer https://github.com/ThilinaRajapakse/simpletransformers
"""

import os
import random

import numpy as np
import torch
from transformers import (
    CTRLConfig,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMConfig,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetConfig,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
from transformers import BertTokenizerFast

from textgen.config.model_args import LanguageGenerationArgs
from textgen.language_generation.language_generation_utils import PREPROCESSING_FUNCTIONS
from loguru import logger

has_cuda = torch.cuda.is_available()
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


class LanguageGenerationModel:
    def __init__(
            self,
            model_type,
            model_name,
            args=None,
            use_cuda=has_cuda,
            cuda_device=-1,
            model=None,
            tokenizer=None,
            **kwargs,
    ):

        """
        Initializes a LanguageGenerationModel model.

        Args:
            model_type: The type of model (gpt2, ctrl, openai-gpt, xlnet, transfo-xl, xlm)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
            "ctrl": (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer),
            "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "xlnet": (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
            "transfo-xl": (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
            "xlm": (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LanguageGenerationArgs):
            self.args = args

        self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"
        logger.debug(f"Device: {self.device}")

        self.args.model_name = model_name
        self.args.model_type = model_type

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer is None:
            # Special tokenizer for chinese gpt2 model
            if self.args.model_name in ['ckiplab/gpt2-base-chinese']:
                tokenizer_class = BertTokenizerFast

            if self.args.tokenizer_name:
                self.tokenizer = tokenizer_class.from_pretrained(
                    self.args.tokenizer_name
                )
            else:
                self.tokenizer = tokenizer_class.from_pretrained(
                    model_name, **kwargs
                )
                self.args.tokenizer_name = model_name
        else:
            self.tokenizer = tokenizer

        if self.args.config_name:
            self.config = config_class.from_pretrained(
                self.args.config_name
            )
        else:
            self.config = config_class.from_pretrained(
                model_name, **kwargs
            )
        if model is None:
            self.model = model_class.from_pretrained(
                model_name,
                config=self.config,
                **kwargs,
            )
        else:
            self.model = model

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def generate(self, prompt=None, args=None, verbose=False, add_cls_head=False, keep_prompt=True,
                 split_on_space=False):
        """
        Generate text using a LanguageGenerationModel

        Args:
            prompt (optional): A prompt text for the model. If given, will override args.prompt
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            verbose (optional): If verbose, generated text will be logged to the console.
            add_cls_head (optional): If True, add cls before prompt.
            keep_prompt (optional): If True, keep prompt in outputs.
            split_on_space (optional): If True, input is english string, if False, input is chinese string.
        Returns:
            generated_sequences: Sequences of text generated by the model.
        """  # noqa: ignore flake8"

        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        if args:
            self.args.update_from_dict(args)

        if prompt:
            self.args.prompt = prompt
        elif not self.args.prompt:
            self.args.prompt = input("Model prompt >>> ")

        prompt_text = self.args.prompt
        args = self.args

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(
                args, model, tokenizer, prompt_text
            )
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
                add_space_before_punct_symbol=True,
            )
        elif add_cls_head:
            encoded_prompt = [tokenizer.cls_token_id] + tokenizer.encode(
                prompt_text, add_special_tokens=False
            )
            encoded_prompt = torch.tensor([encoded_prompt], dtype=torch.long)
        else:
            encoded_prompt = tokenizer.encode(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )
        encoded_prompt = encoded_prompt.to(device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.max_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            if verbose:
                logger.info("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()
            # Decode text
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            prompt_len = len(tokenizer.decode(encoded_prompt[0], skip_special_tokens=True))
            # Split on space, False for chinese string
            gen_text = text[prompt_len:]
            if not split_on_space:
                gen_text = ''.join(gen_text.split(' '))
            if keep_prompt:
                total_sequence = prompt_text + gen_text
            else:
                total_sequence = gen_text
            generated_sequences.append(total_sequence)
            if verbose:
                logger.info(total_sequence)

        return generated_sequences

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = LanguageGenerationArgs()
        args.load(input_dir)
        return args
