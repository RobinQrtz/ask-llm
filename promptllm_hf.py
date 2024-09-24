import logging
import os
import json
import re
from argparse import ArgumentParser
from functools import partial
from time import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ask_llm.askllm_prompts import (
    PROMPT_ASKLLM_DA,
    PROMPT_ASKLLM_EN,
    PROMPT_ASKLLM_NO,
    PROMPT_ASKLLM_SV,
)
from ask_llm.fineweb_prompts import (
    PROMPT_FINEWEB_DA,
    PROMPT_FINEWEB_EN,
    PROMPT_FINEWEB_NO,
    PROMPT_FINEWEB_SV,
)
from ask_llm.promptllm_dataset import PromptLLMDataset, pad_collate_fn

DEVICE = "cuda"  # the device to load the model onto
YES_IDS = [7566, 9642, 9891, 10035]  # llama yes ids
JA_IDS = [5967, 53545, 23720, 12203]
YES_IDS.extend(JA_IDS)
NO_IDS = [2360, 2822, 912, 2201]


def get_args():
    args = ArgumentParser()
    args.add_argument(
        "--model_name", type=str, default="models/Meta-Llama-3.1-70B-Instruct"
    )
    args.add_argument(
        "--language", type=str, choices=["Swedish", "Norwegian", "Danish"]
    )
    args.add_argument("--num_samples", type=int, default=None)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--max_length", type=int, default=512)
    args.add_argument("--max_new_tokens", type=int, default=512)
    args.add_argument("--cache_dir", type=str, default="cache_dir/")
    args.add_argument("--output_dir", type=str, default="output/")
    args.add_argument("--input_file", type=str)
    args.add_argument("--log_level", type=str, default="info")
    args.add_argument("--log_file", type=str, default=None)
    args.add_argument(
        "--prompt", type=str, choices=["askllm", "askllm-en", "fineweb", "fineweb-en"]
    )

    args = args.parse_args()

    if args.prompt == "askllm":
        match args.language:
            case "Swedish":
                args.prompt = PROMPT_ASKLLM_SV
            case "Norwegian":
                args.prompt = PROMPT_ASKLLM_NO
            case "Danish":
                args.prompt = PROMPT_ASKLLM_DA
    elif args.prompt == "askllm-en":
        args.prompt = PROMPT_ASKLLM_EN
    elif args.prompt == "fineweb-en":
        args.prompt = PROMPT_FINEWEB_EN
    elif args.prompt == "fineweb":
        match args.language:
            case "Swedish":
                args.prompt = PROMPT_FINEWEB_SV
            case "Norwegian":
                args.prompt = PROMPT_FINEWEB_NO
            case "Danish":
                args.prompt = PROMPT_FINEWEB_DA
    return args


def setup_logger(args):
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    logging.basicConfig(level=numeric_level)
    logger = logging.getLogger(__name__, filename=args.log_file)
    return logger


def setup_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # model.config.pad_token_id = model.config.eos_token_id

    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    # logger.info(model.generation_config)
    # tokenizer.padding_side = "left"
    return tokenizer, model


def setup_data(args, tokenizer):
    # Load the dataset in streaming mode
    # dataset = load_dataset("wikipedia", language="sv", date="20240520", streaming=True, cache_dir="wiki-sv",)
    #
    file_format = args.input_file.split(".")[-1]
    if file_format == "jsonl":
        file_format = "json"

    assert file_format == "parquet" or file_format == "json"

    ds = load_dataset(
        file_format,
        data_files={"train": args.input_file},
        split="train",
        cache_dir=args.cache_dir,
    )

    if args.num_samples is None or args.num_samples <= 0:
        args.num_samples = len(ds)
    else:
        ds = ds.take(min(args.num_samples, len(ds)))

    pd = PromptLLMDataset(ds, tokenizer, args.prompt, args.language, args.max_length)

    my_collate_fn = partial(pad_collate_fn, tokenizer=tokenizer)

    dataloader = torch.utils.data.DataLoader(
        pd,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=my_collate_fn,
    )
    return dataloader, ds


def main():
    args = get_args()
    tokenizer, model = setup_model(args)
    dataloader, ds = setup_data(args, tokenizer)
    logger = setup_logger(args)

    logger.info(args)
    logger.info(f"Length dataloader {len(dataloader)}")
    # Generate with dataloader
    start = time()
    scores = []
    answers = []
    for batch in tqdm(dataloader):
        batch["input_ids"] = batch["input_ids"].to(DEVICE)
        batch["attention_mask"] = batch["attention_mask"].to(DEVICE)
        outputs = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            output_logits=True,
            return_dict_in_generate=True,
        )

        if args.prompt.startswith("fineweb"):
            generated_answers = tokenizer.batch_decode(outputs["sequences"])
            answers.extend(generated_answers)
        elif args.prompt.startswith("askllm"):
            #
            # 1) compute softmax
            # 2) get probs for yes-ids
            # 3) get values for output token 3 & 4 as llama is not consistent
            # 4) sum those
            # 5) stack and sum yes-id-values for each item in the batch
            scores_yes = torch.sum(
                torch.stack(
                    [
                        torch.sum(
                            torch.nn.functional.softmax(outputs.logits[i], dim=-1)[
                                :, YES_IDS
                            ],
                            dim=-1,
                        )
                        for i in [3, 4]
                    ],
                ),
                dim=0,
            )
            logger.debug(f"scores_yes: {scores_yes}")
            scores.append(scores_yes)
    logger.info(
        f"The loop took {time() - start:.4f} seconds; {args.num_samples / (time() - start):.4f} samples/second"
    )

    if args.prompt.startswith("askllm"):
        scores = torch.cat(scores)
        scores = scores.tolist()
        ds = ds.add_column("yes_score", scores)
    elif args.prompt.startswith("fineweb"):
        ds = ds.add_column("promptllm_answer", answers)
    ds.to_parquet(
        os.path.join(args.output_dir, os.path.basename(args.input_file) + ".parquet")
    )


if __name__ == "__main__":
    main()
