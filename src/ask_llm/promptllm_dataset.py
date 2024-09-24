from torch.utils.data import Dataset

from ask_llm.prompt import apply_prompt


class PromptLLMDataset(Dataset):
    def __init__(self, ds, tokenizer, prompt, language, max_length):
        self.ds = ds.with_format("torch")
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.language = language
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return apply_prompt(
            self.ds[idx],
            self.prompt,
            self.tokenizer,
            self.language,
            self.max_length,
        ).squeeze()


def pad_collate_fn(batch, tokenizer):
    batch = tokenizer.pad({"input_ids": batch})
    return batch
