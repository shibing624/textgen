# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
import argparse
from transformers import BertTokenizerFast
import sys

sys.path.append('../..')
from textgen.language_generation import LanguageGenerationModel
from textgen.language_modeling import LanguageModelingModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/zh_article.txt', type=str, help='Training data file')
    parser.add_argument('--test_file', default='../data/zh_article.txt', type=str, help='Test data file')
    parser.add_argument('--model_type', default='gpt2', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='ckiplab/gpt2-base-chinese', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/couplet-fine-tuned/', type=str,
                        help='Model output directory')
    parser.add_argument('--max_seq_length', default=50, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Training...')
        train_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "block_size": 512,
            "max_seq_length": args.max_seq_length,
            "learning_rate": 5e-6,
            "train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": args.num_epochs,
            "mlm": False,
            "output_dir": args.output_dir,
        }

        model = LanguageModelingModel(args.model_type, args.model_name, args=train_args)
        model.train_model(args.train_file, eval_file=args.test_file)
        print(model.eval_model(args.test_file))

    if args.do_predict:
        logger.info('Predict...')
        # Use fine-tuned model
        tokenizer = BertTokenizerFast.from_pretrained(args.output_dir)
        model = LanguageGenerationModel(args.model_type, args.output_dir, args={"max_length": args.max_seq_length},
                                        tokenizer=tokenizer)

        """
        王语嫣知道表哥神智已乱，富贵梦越做越深，不禁凄然。

        段誉见到阿碧的神情，怜惜之念大起，只盼招呼她和慕容复回去大理，妥为安顿，却见她瞧着慕容复的眼色中柔情无限，
        而慕容复也是一副志得意满之态，心中登时一凛：“各有各的缘法，慕容兄与阿碧如此，我觉得他们可怜，其实他们心中，
        焉知不是心满意足？我又何必多事？”轻轻拉了拉王语嫣的衣袖，做个手势。

        众人都悄悄退了开去。但见慕容复在土坟上南面而坐，口中兀自喃喃不休。
        """
        prompts = [
            "王语嫣知道表哥神智已乱",
            "段誉见到阿碧的神情，怜惜之念大起，只盼",
            "众人都悄悄退了开去。但见慕容复在土坟上南面而坐"
        ]
        for prompt in prompts:
            # Generate text using the model. Verbose set to False to prevent logging generated sequences.
            generated = model.generate(prompt, verbose=False)
            generated = generated[0]
            print("inputs:", prompt)
            print("outputs:", generated)


if __name__ == '__main__':
    main()
