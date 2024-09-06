import torch
from torch.utils.data import Dataset
from ask_llm.prompt import askllm_prompt


class AskLLMDataset(Dataset):
    def __init__(self, ds, tokenizer, language, max_tokens, device):
        self.ds = ds.with_format("torch")
        self.tokenizer = tokenizer
        self.language = language
        self.max_tokens = max_tokens
        self.device = device

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return askllm_prompt(
            self.ds[idx], self.tokenizer, self.language, self.max_tokens
        )["text_prompt"].to(self.device)
