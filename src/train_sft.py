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
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model

from data_module import load_examples
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example

MODEL_ID = "meta-llama/Llama-3.2-1B"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Resolve project root and data directories
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_subdir = "train" ## only load training data

    # Load examples
    print("Loading training examples...")
    train_examples = load_examples(str(processed_dir), train_subdir)
    print(f"Loaded {len(train_examples)} training examples. Will split 20% into eval dataset")

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

    full_ds = make_dataset(train_examples)
    split = full_ds.train_test_split(test_size=0.20, seed=42)
    train_ds = split["train"]   # 80 %
    eval_ds  = split["test"]    # 20 %

    # Load tokenizer and tokenize datasets
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    def tokenize_fn(batch, MAX_LEN=2048):
        """
        1. Tokenise the *full prompt*   → `inputs`
        2. Tokenise the gold-only text → `labels`
        3. Left-pad `labels` with -100 so it matches the length of `inputs`
        """

        # 1️⃣ Full prompt (question + paragraphs + all candidates)
        enc_full = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="longest",
        )

        enc_gold = tokenizer(
            batch["labels"],
            truncation=True,
            max_length=MAX_LEN,
            # we pad left so the gold sequence aligns to the right end of the prompt
            padding="longest",
        )

        # 3️⃣ Build label tensor, mask non-assistant tokens with -100
        labels = []
        for inp_ids, gold_ids in zip(enc_full["input_ids"], enc_gold["input_ids"]):
            pad_len  = len(inp_ids) - len(gold_ids)
            # pad left with -100 to match length
            labels.append([-100] * pad_len + gold_ids)

        enc_full["labels"] = labels
        return enc_full

    print("Tokenizing training data...")
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "labels"]
    )

    print("Tokenizing eval data...")
    eval_ds = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "labels"]
    )

    # Initialize model without bitsandbytes
    print("Loading base model...")
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(load_in_4bit=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
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
    fp16_flag = torch.cuda.is_available()
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "sft3"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        num_train_epochs=5,
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
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train and save
    trainer.train()
    trainer.save_model(str(PROJECT_ROOT / "models" / "sft3"))

if __name__ == "__main__":
    main()