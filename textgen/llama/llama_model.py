# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

modified from https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""
import math
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_int8_training,
    get_peft_model_state_dict,
)
from tqdm.auto import tqdm
from transformers import GenerationConfig, DataCollatorForSeq2Seq
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers.trainer import TRAINING_ARGS_NAME

from textgen.config.model_args import LlamaArgs
from textgen.llama.llama_utils import (
    load_hf_instruction_dataset,
    LlamaInstructionDataset,
    LlamaPretrainingDataset,
    load_hf_pretraining_dataset,
)

has_cuda = torch.cuda.is_available()
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
}


class LlamaModel:
    def __init__(
            self,
            model_type="llama",
            model_name="shibing624/chinese-alpaca-plus-7b-hf",
            peft_name=None,
            args=None,
            use_cuda=has_cuda,
            cuda_device=-1,
            **kwargs,
    ):

        """
        Initializes a LlamaModel model.

        Args:
            model_type: The type of model (llama)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            peft_name (optional): Peft model name
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (int, optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"
        model_type = model_type.lower()
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LlamaArgs):
            self.args = args

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if torch.cuda.is_available() > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        self.device_map = "auto"
        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
                    self.device_map = {"": int(cuda_device)}
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = "cpu"
                self.device_map = {"": "cpu"}
        logger.debug(f"Device: {self.device}")
        if not use_cuda:
            self.args.fp16 = False
            self.args.int8 = False
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = world_size != 1
        if self.ddp:
            self.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

        self.results = {}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if model_name is None:
            model_name = self.args.model_name_or_path
        config = AutoConfig.from_pretrained(model_name, **kwargs)

        self.model = model_class.from_pretrained(
            model_name,
            config=config,
            load_in_8bit=self.args.int8,
            torch_dtype=(torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else torch.float32)),
            device_map=self.device_map,
        )

        self.tokenizer_class = tokenizer_class
        if self.args.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(self.args.tokenizer_name)
        else:
            self.tokenizer = tokenizer_class.from_pretrained(model_name)
            self.args.tokenizer_name = self.args.model_name

        self.args.model_type = model_type
        if model_name is None:
            self.args.model_name = "Llama_from_scratch"
        else:
            self.args.model_name = model_name

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.resize_model_embeddings(len(self.tokenizer))

        self.peft_name = peft_name
        if self.args.use_peft:
            self.load_peft_model()

    def resize_model_embeddings(self, tokenizer_vocab_size):
        """Resizes model embeddings to match the tokenizer vocab size."""
        model_vocab_size = self.model.get_input_embeddings().weight.size(0)
        if model_vocab_size != tokenizer_vocab_size:
            logger.debug(
                f"Resize model embeddings to fit tokenizer, "
                f"Vocab of the base model: {model_vocab_size}, "
                f"Vocab of the tokenizer: {tokenizer_vocab_size}"
            )
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            assert self.model.get_input_embeddings().weight.size(0) == len(self.tokenizer)
            logger.debug(f"Model token embeddings updated, size: {len(self.tokenizer)}")

    def find_all_linear_names(self, int4=False, int8=False):
        cls = torch.nn.Linear
        if int4 or int8:
            import bitsandbytes as bnb
            if int4:
                cls = bnb.nn.Linear4bit
            elif int8:
                cls = bnb.nn.Linear8bitLt
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def train_model(
            self,
            train_data,
            output_dir=None,
            args=None,
            eval_data=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 3 columns - `instruction`, `input`, `output`.
                        - `instruction`: The instruction text. (E.g. `"correct the following:"`)
                        - `input`: The input text sequence. `instruction` is automatically prepended to form the full input. (<instruction> `\n` <input>)
                        - `output`: The target sequence
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            verbose (optional): If True, all of the warnings related to data processing will be printed. 
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)
        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir
        if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )
        # update model train config
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        if not self.ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            self.model.is_parallelizable = True
            self.model.model_parallel = True
        self.model.config.use_cache = False
        resume_from_checkpoint = self.args.resume_from_checkpoint
        if 'all' in self.args.lora_target_modules:
            self.args.lora_target_modules = self.find_all_linear_names(self.args.int4, self.args.int8)

        # setup peft
        if self.args.use_peft:
            peft_type = self.args.peft_type.upper()
            logger.info(f"Using PEFT type: {peft_type}")
            # add peft config
            if peft_type == 'LORA':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )
            elif peft_type == 'ADALORA':
                from peft import AdaLoraConfig

                peft_config = AdaLoraConfig(
                    init_r=self.args.adalora_init_r,
                    r=self.args.lora_r,
                    beta1=self.args.lora_beta,
                    beta2=self.args.lora_beta,
                    tinit=self.args.adalora_tinit,
                    tfinal=self.args.adalora_tfinal,
                    deltaT=self.args.adalora_delta_t,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                )
            elif peft_type == 'PROMPT_TUNING':
                from peft import PromptTuningConfig

                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                )
            elif peft_type == 'P_TUNING':
                from peft import PromptEncoderConfig

                peft_config = PromptEncoderConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size
                )
            elif peft_type == 'PREFIX_TUNING':
                from peft import PrefixTuningConfig

                peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size,
                    prefix_projection=True,
                )
                self.model.gradient_checkpointing_disable()
            else:
                logger.warning(f"Wrong type of peft. Set to default lora")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )

            if self.args.int8:
                self.model = prepare_model_for_int8_training(self.model)

            old_checkpoint_dir = None
            if resume_from_checkpoint:
                # Check the available weights and load them
                checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
                if not os.path.exists(checkpoint_name):
                    checkpoint_name = os.path.join(
                        resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
                    resume_from_checkpoint = (
                        False  # So the trainer won't try loading its state
                    )
                # The two files above have a different name depending on how they were saved, but are actually the same.
                if os.path.exists(checkpoint_name):
                    logger.info(f"Restarting from {checkpoint_name}")
                    old_checkpoint_dir = os.path.dirname(checkpoint_name)
                else:
                    logger.warning(f"Checkpoint {checkpoint_name} not found")

            if old_checkpoint_dir:
                self.model = PeftModel.from_pretrained(self.model, old_checkpoint_dir)
                self.model = self.model.merge_and_unload()
            else:
                logger.info(f"Initializing LoRA model from scratch")
                self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        else:
            logger.warning("Now full model params fine-tune, which is slow, set `use_peft=True` for lora fine-tune.")
        os.makedirs(output_dir, exist_ok=True)

        # load dataset
        train_dataset = self.load_and_cache_examples(train_data)
        if verbose:
            logger.debug(f"train_dataset len: {len(train_dataset)}, train_dataset[0]: {train_dataset[0]}")
        eval_dataset = None
        if eval_data is not None:
            eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)
            if verbose:
                logger.debug(f"eval_dataset len: {len(eval_dataset)}, eval_dataset[0]: {eval_dataset[0]}")

        # start train
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.args.logging_steps,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            save_steps=self.args.save_steps,
            optim=self.args.optimizer,
            save_strategy=self.args.save_strategy,
            evaluation_strategy='steps' if eval_data is not None else 'no',
            eval_steps=self.args.eval_steps if eval_data is not None else None,
            load_best_model_at_end=True if eval_data is not None else False,
            ddp_find_unused_parameters=False if self.ddp else None,
            save_total_limit=self.args.save_total_limit,
            fp16=self.args.fp16,
            remove_unused_columns=self.args.remove_unused_columns,
            report_to=self.args.report_to,
            overwrite_output_dir=self.args.overwrite_output_dir,
            no_cuda=True if self.device == "cpu" else False,
            **kwargs
        )
        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        if training_args.local_rank <= 0:
            logger.info(f"Training/evaluation parameters {training_args}")

        # Initialize our Trainer
        if self.args.is_pretraining:
            # For pretraining, we use a special data collator that simply regroups
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=self.args.pad_to_multiple_of,
                return_tensors="pt",
                padding=True
            )
        else:
            # For fine-tuning, padding to max length
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                padding="max_length",
                max_length=self.args.max_seq_length + self.args.max_length
            )
        if self.args.use_peft:
            trainer = SavePeftModelTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if eval_data is not None else None,
                args=training_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
        else:
            trainer = Trainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if eval_data is not None else None,
                args=training_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        if self.args.enable_torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        if self.args.int8 or self.args.int4:
            old_state_dict = self.model.state_dict
            self.model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
            ).__get__(self.model, type(self.model))

        logger.info("*** Train ***")
        (global_step, training_loss, metrics) = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.handle_metrics("train", metrics, output_dir)
        self.results.update(metrics)
        self.save_model(model=self.model)

        if eval_data is not None:
            logger.info("*** Evaluate ***")
            if self.args.fp16:
                self.model.half()
            metrics = trainer.evaluate(metric_key_prefix="eval")
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
            logger.debug(f"eval metrics: {metrics}")
            self.handle_metrics("eval", metrics, output_dir)
            self.results.update(metrics)

        if verbose and training_args.local_rank <= 0:
            logger.debug(f"metrics: {self.results}")
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_name, output_dir
                )
            )
        return global_step, training_loss

    @staticmethod
    def handle_metrics(split, metrics, output_dir):
        """
        Log and save metrics

        Args:
        - split: one of train, val, test
        - metrics: metrics dict
        - output_dir: where to save the metrics
        """

        logger.info(f"***** {split} metrics *****")
        for key in sorted(metrics.keys()):
            logger.info(f"  {key} = {metrics[key]}")
        output_file = os.path.join(output_dir, f"{split}_results.txt")
        with open(output_file, "w") as writer:
            for key in sorted(metrics.keys()):
                writer.write("{} = {}\n".format(key, str(metrics[key])))

    def load_peft_model(self):
        """Load peft model"""
        if self.peft_name:
            if os.path.isdir(self.peft_name) and os.path.exists(
                    os.path.join(self.peft_name, "tokenizer_config.json")):
                update_tokenizer = True
            else:
                update_tokenizer = False
            if "ziqingyang/chinese" in self.peft_name or update_tokenizer:
                self.tokenizer = LlamaTokenizer.from_pretrained(self.peft_name)
                self.resize_model_embeddings(len(self.tokenizer))
            self.model = PeftModel.from_pretrained(
                self.model,
                self.peft_name,
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
                device_map=self.device_map,
            )
            logger.info(f"Loaded peft model from {self.peft_name}")
        else:
            # Load peft model from output_dir
            peft_path = os.path.join(self.args.output_dir, self.args.peft_bin_name)
            if peft_path and os.path.exists(peft_path):
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.args.output_dir,
                    torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
                    device_map=self.device_map,
                )
                logger.info(f"Loaded peft model from {peft_path}")

    @torch.no_grad()
    def predict(self, sentences: List[str], keep_prompt: bool = False, max_length: int = None, **kwargs):
        """
        Performs predictions on a list of text.

        Args:
            sentences: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.
            keep_prompt: Whether to keep the prompt in the generated text.
            max_length: The maximum length of the generated text.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        if self.device == 'cpu':
            self.model.float()
        if self.args.fp16:
            self.model.half()
        self.model.eval()

        all_outputs = []
        # Batching
        for batch in tqdm(
                [
                    sentences[i: i + self.args.eval_batch_size]
                    for i in range(0, len(sentences), self.args.eval_batch_size)
                ],
                desc="Generating outputs",
                disable=self.args.silent,
        ):
            inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to(self.device)
            generation_config = GenerationConfig(
                max_new_tokens=max_length if max_length else self.args.max_length,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                do_sample=self.args.do_sample,
                repetition_penalty=self.args.repetition_penalty,
                length_penalty=self.args.length_penalty,
                num_beams=self.args.num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=self.args.num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            outputs = self.model.generate(**inputs, generation_config=generation_config)
            for idx, (prompt_text, generated_sequence) in enumerate(zip(batch, outputs.sequences)):
                # Decode text
                text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                prompt_len = len(prompt_text)
                gen_text = text[prompt_len:]
                if keep_prompt:
                    total_sequence = prompt_text + gen_text
                else:
                    total_sequence = gen_text
                all_outputs.append(total_sequence)
        return all_outputs

    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None,
             keep_prompt: bool = False, max_length: int = 128, **kwargs):
        """
        Chat with the model
        :param query:
        :param history:
        :param keep_prompt:
        :param max_length:
        :param kwargs:
        :return: response, history
        """
        if history is None:
            history = []
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        response = self.predict([prompt], keep_prompt=keep_prompt, max_length=len(prompt) + max_length, **kwargs)[0]
        history = history + [(query, response)]
        return response, history

    def load_and_cache_examples(
            self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a LlamaDataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        if self.args.is_pretraining:
            if self.args.use_hf_datasets:
                dataset = load_hf_pretraining_dataset(data, tokenizer, self.args, mode)
                return dataset
            elif args.dataset_class:
                CustomDataset = args.dataset_class
                return CustomDataset(tokenizer, args, data, mode)
            else:
                return LlamaPretrainingDataset(
                    tokenizer,
                    self.args,
                    data,
                    mode,
                )
        else:
            if self.args.use_hf_datasets:
                dataset = load_hf_instruction_dataset(data, tokenizer, self.args, mode)
                return dataset
            elif args.dataset_class:
                CustomDataset = args.dataset_class
                return CustomDataset(tokenizer, args, data, mode)
            else:
                return LlamaInstructionDataset(
                    tokenizer,
                    self.args,
                    data,
                    mode,
                )

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        """Save the model and the tokenizer."""
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            # save model
            self.save_model_args(output_dir)

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = LlamaArgs()
        args.load(input_dir)
        return args


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
