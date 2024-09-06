from argparse import ArgumentParser
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
from functools import partial
from tqdm import tqdm
from ask_llm.prompt import apply_askllm_prompt
from ask_llm.askllm_dataset import AskLLMDataset
import logging

args = ArgumentParser()
args.add_argument(
    "--model_name", type=str, default="models/Meta-Llama-3.1-70B-Instruct"
)
args.add_argument("--language", type=str, default="Swedish")
args.add_argument("--data_shard", type=int, default=0)
args.add_argument("--num_samples", type=int, default=40)
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--max_tokens", type=int, default=512)
args.add_argument("--cache_dir", type=str, default="models/cache/")
args.add_argument("--output_dir", type=str, default="output/")

args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(args)

device = "cuda"  # the device to load the model onto
yes_ids = [7566, 9642, 9891, 10035]  # llama yes ids

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    cache_dir=args.cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load the dataset in streaming mode
# dataset = load_dataset("wikipedia", language="sv", date="20240520", streaming=True, cache_dir="wiki-sv",)
dataset = load_dataset(
    "json",
    data_files={"train": args.input_file},
    split="train",
    cache_dir="data/cache_dir",
)

logging.info(f"Shuffling dataset")
# should shuffle before take
dataset = dataset.shuffle(seed=666)
ds = dataset.take(args.num_samples)

# # compute ppl for pkv
# prompt_prefix_length = fineweb_prompt_prefix_length(
#     PROMPT, tokenizer, language="svenska", prompt_type=prompt_type
# )

# Apply prompt template and save the result
# logging.info(f"Applying prompts")
# ds = ds.map(
#    apply_askllm_prompt,
#    batched=True,
#    num_proc=16,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "language": args.language,
#        "max_tokens": args.max_tokens,
#    },
# )


ds = AskLLMDataset(ds, tokenizer, args.language, args.max_tokens, device)

dataloader = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, num_workers=4, pin_memory=True
)

# Generate with dataloader
for batch in tqdm(dataloader):
    outputs = model.generate(
        **batch,
        max_new_tokens=1,
        do_sample=False,
        num_beams=1,
        output_logits=True,
        return_dict_in_generate=True,
    )
    logits = outputs.logits
    logger.debug(f"logits.shape: {logits.shape}")  # (batch_size, vocab_size)
    # convert logits to probabilities
    # see https://github.com/huggingface/transformers/blob/56baa03380fc17d85705240ebbc57c075a9b3f23/tests/generation/test_utils.py#L3507  # noqa: E501
    probs = torch.nn.functional.softmax(logits, dim=-1)
    logger.debug(f"probs.shape: {probs.shape}")  # (batch_size, vocab_size)
    yes_probs = probs[:, yes_ids]
    logger.debug(f"yes_probs.shape: {yes_probs.shape}")  # (batch_size, num_yes_tokens)
    scores = torch.sum(yes_probs, dim=-1)
    logger.debug(f"scores.shape: {scores.shape}")  # (batch_size,)
