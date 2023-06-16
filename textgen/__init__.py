# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

__version__ = '1.0.0'

from textgen.augment.text_augment import TextAugment

from textgen.config.model_args import LanguageGenerationArgs
from textgen.language_generation.language_generation_model import LanguageGenerationModel

from textgen.config.model_args import LanguageModelingArgs
from textgen.language_modeling.language_modeling_model import LanguageModelingModel

from textgen.config.model_args import SongNetArgs
from textgen.language_modeling.songnet_model import SongNetModel
from textgen.language_modeling.songnet_utils import SongNetTokenizer, snapshot_download

from textgen.config.model_args import Seq2SeqArgs
from textgen.seq2seq.seq2seq_model import Seq2SeqModel
from textgen.seq2seq.conv_seq2seq_model import ConvSeq2SeqModel
from textgen.seq2seq.bart_seq2seq_model import BartSeq2SeqModel

from textgen.config.model_args import T5Args
from textgen.t5.t5_model import T5Model

from textgen.config.model_args import CopyT5Args
from textgen.t5.copyt5_model import CopyT5Model
from textgen.t5.copyt5_utils import ZHTokenizer

from textgen.unsup_generation.tgls_model import TglsModel

from textgen.config.model_args import ChatGlmArgs
from textgen.chatglm.chatglm_model import ChatGlmModel

from textgen.config.model_args import GptArgs
from textgen.gpt.gpt_model import GptModel

