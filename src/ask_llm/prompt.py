def apply_prompt(example, prompt, tokenizer, language, max_length):
    extract = tokenizer.decode(
        tokenizer.encode(
            example["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
    ).replace("###", "")
    prompted_text = prompt.format(extract=extract, language=language)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can evaluate the educational value of a web page.",
        },
        {"role": "user", "content": prompted_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_special_tokens=True, return_tensors="pt"
    )
