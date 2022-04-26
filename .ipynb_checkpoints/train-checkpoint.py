
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     Setup     --
#···············································································
import argparse
import logging
import math
import os
import random
from functools import partial
from packaging import version

# Import from third party libraries
import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
import transformers

from transformer_mt import utils


# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()

bleu = datasets.load_metric("sacrebleu")

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --      Command Line Args     --
#···············································································
def parse_args():
    parser = argparse.ArgumentParser(description="Train machine translation model")

    # required args (?)
    parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            )
    # multiple src and target langs 
    parser.add_argument(
            "--source_langs",
            nargs="+",
            default="en"
            )
    parser.add_argument(
            "--target_langs",
            nargs="+",
            default="fr"
            )

    # non-required args
    parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="T5ForConditionalGeneration"
            )
    parser.add_argument(
            "--model_subset",
            type=str,
            default="t5-small"
            )

    # T5 config (optional)
    parser.add_argument(
            "--vocab_size",
            type=int,
            default=32128,
            help="Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling T5Model. Default=32128"
            )
    parser.add_argument(
            "--d_model",
            type=int,
            default=512,
            help="Size of the encoder layers and the pooler layer. Default=512"
            )
    parser.add_argument(
            "--d_kv",
            type=int,
            default=64,
            help=" Size of the key, query, value projections per attention head. d_kv has to be equal to d_model // num_heads. Default=64"
            )
    parser.add_argument(
            "--d_ff",
            type=int,
            default=2048,
            help="Size of the intermediate feed forward layer in each T5Block. Default=2048"
            )
    parser.add_argument(
            "--num_layers",
            type=int,
            default=6,
            help="Number of hidden layers in the Transformer encoder. Default=6"
            )
    parser.add_argument(
            "--num_decoder_layers",
            type=int,
            default=6,
            help=" Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not set (6)"
            )
    parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="Number of attention heads for each attention layer in the Transformer encoder. Default=8"
            )
    parser.add_argument(
            "--relative_attention_num_buckets",
            type=int,
            default=32,
            help="The number of buckets to use for each attention layer. Default=32"
            )

    parser.add_argument(
            "--relative_attention_max_distance",
            type=int,
            default=128,
            help="The maximum distance of the longer sequences for the bucket separation. Default=128"
            )

    parser.add_argument(
            "--dropout_rate",
            type=float,
            default=0.1,
            help="The ratio for all dropout layers. Default=0.1"
            )
    parser.add_argument(
            "--layer_norm_eps",
            type=float,
            default=1e-6,
            help="The epsilon used by the layer normalization layers. Default=1e-6"
            )
    parser.add_argument(
            "--initializer_factor",
            type=float,
            default=1,
            help="A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing). Default=1"
            )
    parser.add_argument(
            "--feed_forward_proj",
            type=str,
            default="relu",
            help="Type of feed forward layer to be used. Should be one of 'relu' or 'gated-gelu'. T5v1.1 uses the 'gated-gelu' feed forward projection. Original T5 uses 'relu'."
            )


    # other args
    parser.add_argument(
            "--tokenizer",
            type=str,
            default="AutoTokenizer"
            )
    parser.add_argument(
            "--max_seq_length",
            type=int,
            default=128
            )
    parser.add_argument(
            "--collator",
            type=str,
            default="DataCollatorForSeq2Seq"
            )
    parser.add_argument(
            "--dataset_name",
            type=str,
            default="wmt14",
            )
    parser.add_argument(
            "--lang_pairs",
            nargs="+",
            default="fr-en"
            )
    parser.add_argument(
            "--dataset_split",
            type=str,
            default=None,
            help="get a subset of the main datset for temporary use"
            )
    parser.add_argument(
           "--preprocessing_num_workers",
           type=int,
           default=8
           )
    parser.add_argument(
           "--overwrite_cache",
           type=bool,
           default=None
           )
    
    # training args
    parser.add_argument(
            "--device",
            default="cuda" if torch.cuda.is_available() else "cpu"
            )
    parser.add_argument(
            "--batch_size",
            type=int,
            default=8
            )
    parser.add_argument(
            "--learning_rate",
            type=float,
            default=3e-4
            )
    parser.add_argument(
            "--weright_decay",
            type=float,
            default=0.0
            )

    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=5000,
        help="Perform evaluation every n network updates.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Compute and log training batch metrics every n steps.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=transformers.SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--generation_type",
        choices=["greedy", "beam_search"],
        default="beam_search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help=("Beam size for beam search generation. "
              "Decreasing this parameter will make evaluation much faster, "
              "increasing this (until a certain value) would likely improve your results."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--wandb_project", 
        default="transformer_mt",
        help="wandb project name to log metrics to"
    )

    args = parser.parse_args()

    return args

 
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


#TODO: add more languages, move to .utils file
code_to_lang = {
        'fr' : 'French',
        'en' : 'English', 
        'ru' : 'Russian'
        }
def set_prefix(ex):
    src, tgt = ex.keys()
    src = code_to_lang[src]
    tgt = code_to_lang[tgt]
    return f"translate from {src} to {tgt}: "


#TODO: wrap in partial in main()
def preprocess_fn(
        examples,
        source_lang,
        target_lang,
        max_seq_length,
        tokenizer,
        both_dirs=True
        ):

    inputs = []
    targets = []

    # preprocess all langs in all directions
    if both_dirs==True:
        inputs.extend([set_prefix(ex) + ex[source_lang] for ex in examples['translation']])
        targets.extend([ex[target_lang] for ex in examples['translation']])

        inputs.extend([set_prefix(ex) + ex[target_lang] for ex in examples['translation']])
        targets.extend([ex[source_lang] for ex in examples['translation']])

    model_inputs=tokenizer(inputs, max_length=max_seq_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_seq_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
    
    return model_inputs


#TODO: MODEL EVAL


def main():
    # parse args
    args = parse_args()
    logger.info(f'Starting script with arguments: {args}')

    #wandb.init(project=args.wandb_project, config=args)

    # Load Data 
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    raw_datasets = []
    for lang_pair in args.lang_pairs:
        raw_datasets.extend(load_dataset('wmt14', lang_pair, split=args.dataset_split))

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, truncation=True)












