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
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
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

    # finding list of subdirectories in processed_dir
    sub_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    train_subdir, dev_subdir = sub_dirs if 'train' in sub_dirs[0] else sub_dirs[::-1]
    print(f"Subdirectories in {processed_dir}: {sub_dirs}")
    print(f"Using {train_subdir} for training and {dev_subdir} for evaluation")
    #train_subdir = "hotpot_dev_distractor_v1"
    #dev_subdir = "hotpot_dev_distractor_v1"

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
            prompt, gold_only = serialize_example(ex, graph, num_divergent=2)
            # For SFT, labels = full prompt including gold candidate
            data["text"].append(prompt)
            data["labels"].append(gold_only)
        return Dataset.from_dict(data)

    train_ds = make_dataset(train_examples)
    dev_ds = make_dataset(dev_examples)

    # Load tokenizer and tokenize datasets
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    def tokenize_fn(batch, MAX_LEN = 2048):
        """
        HF will automatically:
        • return input_ids / attention_mask for `text`
        • create labels from `text_target`, aligning & padding
        • mask all user-side tokens in labels with -100
        """
        return tokenizer(
            text=batch["text"],            # full prompt (question + candidates)
            text_target=batch["labels"],   # gold chain only
            truncation=True,
            max_length=MAX_LEN,
            padding="longest"              # dynamic pad to longest in batch
        )


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
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float32,
            device_map="auto"
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
    torch.cuda.empty_cache()
    fp16_flag = True if torch.cuda.is_available() else False
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "sft2"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        num_train_epochs=2,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        fp16=fp16_flag,
        report_to=["none"]
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
    trainer.save_model(str(PROJECT_ROOT / "models" / "sft2"))

if __name__ == "__main__":
    main()