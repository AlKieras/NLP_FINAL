


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports     --
#···············································································
from attrdict import AttrDict
import toml
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

#from transformer_mt import utils


from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    # DownloadColumn(),
    # "•",
    # TransferSpeedColumn(),
    # "•",
    TimeRemainingColumn(),
)

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
# {{{                             --     Preprocessing     --
#···············································································
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

def preprocess_fn(
        examples,
        source_lang,
        target_lang,
        max_seq_length,
        tokenizer,
        ):

    inputs = []
    targets = []

    inputs.extend([set_prefix(ex) + ex[source_lang] for ex in examples['translation']])
    targets.extend([ex[target_lang] for ex in examples['translation']])

    inputs.extend([set_prefix(ex) + ex[target_lang] for ex in examples['translation']])
    targets.extend([ex[source_lang] for ex in examples['translation']])

    model_inputs=tokenizer(inputs, max_length=max_seq_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_seq_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']

    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────




## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     Eval     --
#···············································································
def evaluate_model(
        model,
        dataloader,
        *,
        tokenizer,
        device,
        max_seq_length,
        beam_size,
        ):
    n_generated_tokens = 0
    model.eval()
    for batch in tqdm(dataloader, desc='Evaluation'):
        with torch.inference_mode():
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            generated_ids = model.generate(
                    input_ids=input_ids,
                    num_beams=beam_size,
                    attention_mask=attention_mask
                    )

            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            decoded_labels = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

            for pred in decoded_preds:
                n_generated_tokens += len(tokenizer(pred)['input_ids'])

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            bleu.add_batch(predictions=decoded_preds, references=decoded_labels)
        
    model.train()
    eval_metric = bleu.compute()
    evaluation_results = {
            'bleu' : eval_metric['score'],
            'generation_length' : n_generated_tokens / len(dataloader.dataset)
            }

    return evaluation_results, input_ids, decoded_preds, decoded_labels



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────





## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     main     --
#···············································································
def main():
    # load config
    args = toml.load('config.toml')
    _init = AttrDict(args['init'][0])
    _train = AttrDict(args['train'][0]) 
    _t5 = AttrDict(args['t5'][0])

    logger.info(f'Starting script with arguments: {args}')

    wandb.init(project=_init.wandb_project, config=args)

    # load data
    os.makedirs(_init.output_dir, exist_ok=True)

    # load datasets
    raw_datasets = []
    for lang_pair in _init.lang_pairs:
        raw_datasets.append(load_dataset('wmt14', lang_pair, split=eval(_init.split)))

    if 'multi-lang-tokenizer' in os.listdir(_init.output_dir):
        tokenizer_path = os.path.join(_init.output_dir, 'multi-lang-tokenizer')
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(_init.model_checkpoint, truncation=True)
    model = transformers.T5ForConditionalGeneration.from_pretrained(_init.model_checkpoint)

    column_names = raw_datasets[0]['train'].column_names

    # preparing model
    train_langs = []
    test_langs = []
    model_outputs = []
    for i, lang_pair in enumerate(_init.lang_pairs):
        src, tgt = lang_pair.split('-')
        preprocess_fn_wrapped = partial(
                preprocess_fn,
                source_lang=src,
                target_lang=tgt,
                max_seq_length=_init.max_seq_length,
                tokenizer=tokenizer,
                )
        model_output = raw_datasets[i].map(
                preprocess_fn_wrapped,
                batched=True,
                num_proc=_init.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc=f'Tokenizing {lang_pair} dataset',
                )
        if _init.split == 'None':
            train_langs.append(model_output['train'])
            test_langs.append(model_output['test'])
        else:
            model_outputs.append(model_output)

    logger.info(f'Saving tokenizer to {_init.output_dir}/multi-lang-tokenizer')
    tokenizer.save_pretrained(os.path.join(_init.output_dir, 'multi-lang-tokenizer'))

    if _init.split == 'None':
        if len(_init.lang_pairs) > 1:
            train_dataset = datasets.concatenate_datasets([*train_langs])
            eval_dataset = datasets.concatenate_datasets([*test_langs])
        else:
            train_dataset = train_langs[0]
            eval_dataset = test_langs[0]
    else:
        all_langs = datasets.concatenate_datasets([*model_outputs])
        all_langs = all_langs.train_test_split(test_size=0.2)
        train_dataset, eval_dataset = all_langs['train']

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
        logger.info(f"Decoded labels: {tokenizer.decode(train_dataset[index]['labels'])}")
        logger.info("\n")

    # data loading and shuffling
    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=_train.batch_size,
            )
    eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=_train.batch_size
            )

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=_train.learning_rate,
            weight_decay=_train.weight_decay
            )

    num_update_steps_per_epoch = len(train_dataloader)
    if _train.max_train_steps == 0:
        max_train_steps = int(_train.num_train_epochs) * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(int(_train.max_train_steps) / num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=_train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=_train.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {_train.num_train_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps))

    # Log a pre-processed training example to make sure the pre-processing does not have bugs in it
    # and we do not input garbage to our model
    batch = next(iter(train_dataloader))

    _labs= batch['labels']
    _labs[_labs == -100] = tokenizer.pad_token_id
    logger.info("Look at the data that we input into the model, check that it looks as expected: ")
    for index in random.sample(range(len(batch)), 2):
        logger.info(f"Decoded input_ids: {tokenizer.decode(batch['input_ids'][index])}")
        logger.info(f"Decoded labels: {tokenizer.decode(batch['labels'][index])}")
        logger.info("\n")

    _labs[_labs == tokenizer.pad_token_id] = -100 
###################################################
    # TRAINING LOOP
######################################################
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    global_step = 0
    model = model.to(device)
    for epoch in range(_train.num_train_epochs):
        model.train()

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels[labels == tokenizer.pad_token_id] = -100

            output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = output['loss']
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            wandb.log(
                    {
                        'train_loss' : loss,
                        'learning_rate' : optimizer.param_groups[0]['lr'],
                        'epoch' : epoch,
                        },
                    step=global_step,
                    )


            if global_step % _train.eval_every_steps == 0 or global_step == _train.max_train_steps:
                 eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
                        model=model,
                        dataloader=eval_dataloader,
                        tokenizer=tokenizer,
                        device=device,
                        max_seq_length=_init.max_seq_length,
                        beam_size=_train.beam_size,
                    )

                 wandb.log(
                         {
                             'eval/bleu' : eval_results['bleu'],
                             'eval/generation_length': eval_results['generation_length'],
                             },
                         step=global_step,
                         )
                 logger.info("Generation example:")
                 random_index = random.randint(0, len(last_input_ids) - 1)
                 logger.info(f"Input sentence: {tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
                 logger.info(f"Generated sentence: {last_decoded_preds[random_index]}")
                 logger.info(f"Reference sentence: {last_decoded_labels[random_index][0]}")

                 logger.info("Saving model checkpoint to %s", _init.output_dir)
                 model.save_pretrained(_init.output_dir)
                # YOUR CODE ENDS HERE logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(_init.output_dir)

    logger.info("Uploading tokenizer, model and config to wandb")
    wandb.save(os.path.join(_init.output_dir, "*"))

    logger.info(f"Script finished succesfully, model saved in {_init.output_dir}")


if __name__ == "__main__":
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")

    main()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



