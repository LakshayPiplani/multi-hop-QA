from huggingface_hub import login
login(token="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")

import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

from data_module import load_examples
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example


def main():
    # Resolve project root and data directories
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_subdir = "hotpot_train_v1.1"
    dev_subdir = "hotpot_dev_distractor_v1"

    # Load examples
    print("Loading training examples...")
    train_examples = load_examples(str(processed_dir), train_subdir)
    print(f"Loaded {len(train_examples)} training examples")
    print("Loading evaluation examples...")
    dev_examples = load_examples(str(processed_dir), dev_subdir)
    print(f"Loaded {len(dev_examples)} evaluation examples")

    # Prepare Dataset objects
    def make_dataset(examples):
        data = {"text": [], "labels": []}
        for ex in examples:
            graph = build_graph(ex)
            prompt = serialize_example(ex, graph, num_divergent=2)
            data["text"].append(prompt)
            data["labels"].append(prompt)
        return Dataset.from_dict(data)

    train_ds = make_dataset(train_examples)
    dev_ds = make_dataset(dev_examples)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Use EOS token for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Resize embeddings for pad token
    model.resize_token_embeddings(len(tokenizer))

    # Tokenization function using text_target
    def tokenize_fn(batch):
        tokenized = tokenizer(
            text=batch["text"],
            text_target=batch["labels"],
            truncation=True,
            max_length=1024,
            padding="longest"
        )
        return tokenized

    print("Tokenizing training data...")
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "labels"]
    )
    print("Tokenizing evaluation data...")
    dev_ds = dev_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "labels"]
    )

    # Apply LoRA adapters
    # print("Applying LoRA adapters...")
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["c_attn", "c_proj"],
    #     lora_dropout=0.05,
    #     task_type="CAUSAL_LM"
    # )
    # model = get_peft_model(model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "sft"),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to=["none"]
    )

    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train and save
    trainer.train()
    trainer.save_model(str(PROJECT_ROOT / "models" / "sft"))

if __name__ == "__main__":
    main()
