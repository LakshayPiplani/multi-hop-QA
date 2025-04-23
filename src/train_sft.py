import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    LlamaTokenizer, LlamaForCausalLM,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# Local imports: adjust according to your module names
from data_module import Example, load_examples
from graph_builder import build_graph
from token_utilizer_Llama import serialize_example_llama2


def create_training_dataset(
    examples: list[Example],
    max_seq_len: int,
    tokenizer: LlamaTokenizer
) -> Dataset:
    """
    Build a HuggingFace Dataset with prompts and responses, tokenized and labeled
    for causal LM fine-tuning.
    """
    prompts = []
    responses = []
    for ex in examples:
        graph = build_graph(ex)
        prompt = serialize_example_llama2(ex, graph)
        # The model should generate the gold candidate again as response
        # Here we assume CANDIDATE 1 is gold, so we repeat that block as the target
        # Adjust serialize_example_llama2 to tag the gold block accordingly
        # For simplicity, we extract the gold candidate text from prompt
        # Assuming "/SYS>>" end and "/INST]" split
        # TODO: customize extraction logic
        # Placeholder: response = "CANDIDATE 1:\n" + ...
        response = prompt  # full echo (warning: this is simplistic)
        prompts.append(prompt)
        responses.append(response)

    data = {'prompt': prompts, 'response': responses}
    ds = Dataset.from_dict(data)

    def preprocess_fn(batch):
        # Encode prompt + response together
        inputs = tokenizer(
            batch['prompt'], batch['response'],
            padding='max_length', truncation=True,
            max_length=max_seq_len, return_tensors='pt'
        )
        # For causal LM, labels = input_ids; optionally mask prompt tokens with -100
        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        # Mask prompt portion
        for i, prompt in enumerate(batch['prompt']):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            labels[i, :len(prompt_ids)] = -100
        inputs['labels'] = labels
        return inputs

    tokenized = ds.map(
        preprocess_fn, batched=True, remove_columns=ds.column_names
    )
    tokenized.set_format(type='torch')
    return tokenized


def main():
    # Resolve directories
    PROJECT_ROOT = Path(__file__).parent.parent
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    # Load Examples
    train_exs = load_examples(str(processed_dir), 'hotpot_train_v1.1')
    val_exs   = load_examples(str(processed_dir), 'hotpot_dev_distractor_v1')

    # Load tokenizer and model
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map='auto'
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=['q_proj','v_proj'],
        lora_dropout=0.05, bias='none'
    )
    model = get_peft_model(model, peft_config)

    # Prepare datasets
    max_seq_len = 2048
    train_ds = create_training_dataset(train_exs, max_seq_len, tokenizer)
    val_ds   = create_training_dataset(val_exs,   max_seq_len, tokenizer)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Training arguments
    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / 'models' / 'sft'),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=1000,
        fp16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()
    # Save final model
    trainer.save_model(str(PROJECT_ROOT / 'models' / 'sft'))

if __name__ == '__main__':
    main()
