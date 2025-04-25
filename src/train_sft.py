from huggingface_hub import login
login(token="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")

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from pathlib import Path

from data_module import load_examples
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example

# --- Constants ---
MODEL_ID = "meta-llama/Llama-3.2-1B" 
MAX_LENGTH = 2048  # Reduce if OOM errors occur

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    train_examples = load_examples(str(PROJECT_ROOT / "data" / "processed"), "train")
    print(f"Loaded {len(train_examples)} examples.")

    # Add this check before training
    sample_prompt, sample_label = serialize_example(train_examples[0], build_graph(train_examples[0]))
    print("--- Prompt ---\n", sample_prompt)
    print("--- Label ---\n", sample_label)

    # --- Dataset Preparation ---
    def serialize_data(examples):
        texts, labels = [], []
        for ex in examples:
            graph = build_graph(ex)
            prompt, gold_text = serialize_example(ex, graph, num_divergent=2)
            texts.append(prompt)
            labels.append(gold_text)
        return {"text": texts, "labels": labels}

    full_ds = Dataset.from_dict(serialize_data(train_examples))
    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds, eval_ds = split_ds["train"], split_ds["test"]

    # --- Tokenization ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Critical for causal LM generation!

    ## test tokenizer
    sample = tokenizer(["Hello world"], return_tensors="pt", padding="longest")
    print("sample tokenization\n", sample["input_ids"])  # Should be left-padded (e.g., `[0, 0, 123, 456]`)

    def tokenize_fn(batch):
        tokenized = tokenizer(text=batch["text"], text_target=batch["labels"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
        return tokenized
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # Tokenize both text and labels together to ensure alignment
        tokenized = tokenizer(
            batch["text"],
            text_target=batch["labels"],  # Handles alignment automatically
            truncation=False,
            max_length=MAX_LENGTH,
            padding=True,
        )
        return tokenized

        # Tokenize inputs (full prompt)
        inputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # Let collator handle padding
        )
        
        # Tokenize labels (gold response only)
        labels = tokenizer(
            batch["labels"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],  # Collator will align these
        }

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "labels"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text", "labels"])
    print("shape of train_ds:", train_ds.shape)
    # Should be identical except for the last dimension
    # --- Model Initialization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # --- LoRA Configuration ---
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    fp16_flag = torch.cuda.is_available()
    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "llama3-sft-final"),
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

    # --- Data Collator ---
    collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding="longest",
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    return_tensors="pt",
    )
    sample_batch = collator([train_ds[0], train_ds[1]])  # Check first 2 examples
    print("Input IDs shape:", sample_batch["input_ids"].shape)
    print("Labels shape:", sample_batch["labels"].shape)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # --- Training ---
    trainer.train()
    trainer.save_model(str(PROJECT_ROOT / "models" / "llama3-sft-final"))

    # --- Test Generation ---
    test_prompt = "QUESTION: What is the capital of France? PARAGRAPHS:..."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    print("Test Generation:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()