from huggingface_hub import login
login(token="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")

import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

from data_module import load_examples
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    # Resolve project root and data directories
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    #train_subdir = "hotpot_train_v1.1"
    train_subdir = "hotpot_dev_distractor_v1"
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
            # For SFT, labels = full prompt including gold candidate
            data["text"].append(prompt)
            data["labels"].append(prompt)
        return Dataset.from_dict(data)

    train_ds = make_dataset(train_examples)
    dev_ds = make_dataset(dev_examples)

    # Load tokenizer and tokenize datasets
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            text_target=batch["labels"],  # Use text_target for labels
            truncation=True,
            max_length=2048,
            padding="max_length"  # Changed to max_length padding
        )
    
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=2048,
            padding=True
        )
        # Tokenize labels separately
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["labels"],
                truncation=True,
                max_length=2048,
                padding=True
            )["input_ids"]
        tokenized["labels"] = labels

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

    # Initialize model without bitsandbytes
    print("Loading base model...")
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        load_in_8bit=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device}
    )

    # Apply LoRA adapters
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    fp16_flag = True if torch.cuda.is_available() else False
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "sft"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        fp16=fp16_flag,
        report_to=["none"],
        label_names=["labels"]
    )

    #data_collator = DataCollatorWithPadding(tokenizer, padding= True)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8,
        padding=True
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
    trainer.save_model(str(PROJECT_ROOT / "models" / "sft1"))

if __name__ == "__main__":
    main()