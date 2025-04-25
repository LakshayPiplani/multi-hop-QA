"""
Adaptive GoT inference for Llama-3.2-1B fine-tuned on HotpotQA with proper Llama-3 chat template
"""

import os, re, math, argparse, torch, json
from collections import deque
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from evaluate import load as load_metric
from data_module import load_examples
from graph_builder import build_graph

# Config
TAU_BITS = 1.3
MAX_STEPS = 4
MAX_NEWTOK = 32
MAX_NODES = 800
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = ROOT / "models" / "sft2"
PROC_DIR = ROOT / "data" / "processed"
DEV_SPLIT = "test"
MODEL_ID = "meta-llama/Llama-3.2-1B" #  model path
TOKENIZER_ID = "meta-llama/Llama-3.2-3B"  # Official tokenizer

def entropy_bits(logits):
    p = logits.softmax(-1)
    return -(p * p.log2()).sum().item()

def build_llama3_prompt(question, graph, gold_path=None, answer=None):
    """
    Builds prompt in exact Llama-3 chat format:
    
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a careful multi-hop reasoner. Think step-by-step.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    QUESTION: {question}
    PARAGRAPHS:
      pid: title || sentences
    CANDIDATE 1:
      [STEP 1] pid
      FINAL: answer<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    [STEP 1] pid
    FINAL: answer
    """
    # System message
    prompt = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>",
        "You are a careful multi-hop reasoner. Think step-by-step.",
        "<|eot_id|>"
    ]
    
    # User message with question and paragraphs
    prompt.extend([
        "<|start_header_id|>user<|end_header_id|>",
        f"QUESTION: {question}",
        "PARAGRAPHS:"
    ])
    
    # Add all paragraphs
    for pid, data in graph.nodes(data=True):
        sent = " ".join(data["sentences"])
        prompt.append(f"  {pid}: {data['title']} || {sent}")
    
    # Add gold candidate if provided (for training)
    if gold_path and answer:
        prompt.append("\nCANDIDATE 1:")
        for i, pid in enumerate(gold_path, 1):
            prompt.append(f"  [STEP {i}] {pid}")
        prompt.append(f"  FINAL: {answer}")
    
    prompt.append("<|eot_id|>")  # End user turn
    
    # Assistant header (for model to complete)
    prompt.append("<|start_header_id|>assistant<|end_header_id|>\n")
    
    return "\n".join(prompt)
"""
def agot_inference(model, tokenizer, ex, tau_bits=TAU_BITS):
    graph = build_graph(ex)
    prompt = build_llama3_prompt(ex.question, graph)
    #print(f"Prompt: {prompt}")
    Node = tuple[str, list[int], float]  # (prompt, chain, logprob)
    #print(f"Node type:", Node)
    queue = deque([(prompt, [], 0.0)])
    
    completed = []
    
    while queue and len(completed) < 4 and len(queue) < MAX_NODES:
        #print("queue2:", queue)
        current_prompt, chain, lp = queue.popleft()
        #print(lp)
        if len(chain) >= MAX_STEPS:
            #print("Max steps reached, skipping...")
            continue
        
        # Tokenize and generate
        inputs = tokenizer(current_prompt, return_tensors="pt").to(DEVICE)
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                #temperature=0.7,
                #top_p=0.9,
                max_new_tokens=1000
            )
        #print("o1", tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        #here
        # Decode new tokens only
        new_tokens = outputs[0][input_length:]
        new_text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        print(f"New text: {new_text}")
        #print(f"Chain: {chain}")
        
        # Check for final answer
        if "FINAL:" in new_text:
            answer = new_text.split("FINAL:")[-1].strip()
            completed.append((chain, answer, lp))
            continue
        
        # Extract next step
        step_match = re.search(r"\[STEP\s+\d+\]\s*(\d+)", new_text)
        if not step_match:
            continue
            
        next_pid = int(step_match.group(1))
        
        # Update logprob
        step_logits = outputs.scores[0][0]
        token_id = tokenizer.encode(str(next_pid), add_special_tokens=False)[0]
        new_lp = lp + step_logits.log_softmax(-1)[token_id].item()
        
        # Get paragraph content
        para = " ".join(graph.nodes[next_pid]["sentences"])
        
        # Build next prompt
        next_prompt = (
            current_prompt + new_text +
            f"\nPARA {next_pid}: {para}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
        
        queue.append((next_prompt, chain + [next_pid], new_lp))
        
        # Branch if uncertain
        if entropy_bits(step_logits) >= tau_bits:
            queue.append((next_prompt, chain + [next_pid], new_lp - 1.0))
    
    return max(completed, key=lambda x: x[2])[1] if completed else ""
    """

def agot_inference(model, tokenizer, ex, tau_bits=TAU_BITS):
    graph = build_graph(ex)
    # Build prompt WITHOUT the assistant header at the end
    prompt = build_llama3_prompt(ex.question, graph).replace(
        "<|start_header_id|>assistant<|end_header_id|>\n", ""
    )
    
    print("\n=== INITIAL PROMPT ===")
    print(prompt)
    print("=====================")

    queue = deque([(prompt, [], 0.0)])
    completed = []
    
    while queue and len(completed) < 4 and len(queue) < MAX_NODES:
        current_prompt, chain, lp = queue.popleft()
        
        if len(chain) >= MAX_STEPS:
            continue

        # Force generation to continue by removing EOS during generation
        inputs = tokenizer(current_prompt, return_tensors="pt").to(DEVICE)
        
        print("\n=== GENERATION INPUT ===")
        print(f"Input length: {len(inputs.input_ids[0])} tokens")
        print("Last 20 tokens:", tokenizer.decode(inputs.input_ids[0][-20:]))
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEWTOK,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=None,  # Disable early stopping
                    early_stopping=False
                )
            
            new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            print("\n=== MODEL OUTPUT ===")
            print("Raw tokens:", new_tokens)
            print("Decoded text:", repr(new_text))
            
            # Clean the output
            new_text = new_text.split('<|eot_id|>')[0].strip()
            if not new_text:
                print("Empty generation after cleaning")
                continue
                
            print("Cleaned text:", repr(new_text))

            # Check for final answer
            if "FINAL:" in new_text:
                answer = new_text.split("FINAL:")[-1].strip()
                completed.append((chain, answer, lp))
                continue
                
            # Extract next step
            step_match = re.search(r"\[STEP\s+\d+\]\s*(\d+)", new_text)
            if not step_match:
                print("No step pattern found")
                continue
                
            next_pid = step_match.group(1)
            print(f"Found next PID: {next_pid}")

            # [Rest of your processing logic...]

        except Exception as e:
            print(f"Generation error: {str(e)}")
            continue

    return completed[0][1] if completed else ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--tau", type=float, default=TAU_BITS)
    args = parser.parse_args()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Model loading
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        ).to(DEVICE)
    
    # Load LoRA adapter if exists
    if (MODEL_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, MODEL_DIR)
    
        print("\n=== BASIC GENERATION TEST ===")
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        #print(f"Input IDs: {inputs['input_ids']}")
        print(f"Tokenized: {tokenizer.batch_decode(inputs['input_ids'])}")


    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            #temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # Load data
    examples = load_examples(str(PROC_DIR), DEV_SPLIT)[:args.num]
    print(f"Loaded {len(examples)} examples")
    metric = load_metric("squad_v2")
    
    # Evaluation
    predictions = []
    references = []
    
    for ex in tqdm(examples, desc="Evaluating"):
        answer = agot_inference(model, tokenizer, ex, args.tau)
        
        predictions.append({
            "id": ex.qid,
            "prediction_text": answer,
            "no_answer_probability": 0.0
        })
        
        references.append({
            "id": ex.qid,
            "answers": {
                "text": [ex.answer],
                "answer_start": [0]
            }
        })
    #print(predictions)
    # Compute metrics
    results = metric.compute(
        predictions=predictions,
        references=references
    )
    
    print(f"\nResults:")
    print(f"Exact Match: {results['exact']:.2f}%")
    print(f"F1: {results['f1']:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()