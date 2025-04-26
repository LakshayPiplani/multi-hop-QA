"""
Adaptive GoT inference for Llama-3.2-1B fine-tuned on HotpotQA with proper Llama-3 chat template
"""
from huggingface_hub import login
login(token="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")
import warnings
warnings.filterwarnings("ignore")
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
MODEL_DIR = ROOT / "models" / "sft2" / "checkpoint-1390"
PROC_DIR = ROOT / "data" / "processed"
DEV_SPLIT = "test"
MODEL_ID = "meta-llama/Llama-3.2-1B" #  model path
TOKENIZER_ID = "meta-llama/Llama-3.2-1B"  # Official tokenizer

def entropy_bits(logits):
    p = logits.softmax(-1)
    return -(p * p.log2()).sum().item()

def build_llama3_prompt(question, graph, gold_path=None, answer=None):
    # System message
    prompt = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>",
        "You are a careful multi-hop reasoner. Think step-by-step.",
        "Your task is to:"
        "1. First identify the most relevant paragraph(s) to the question",
        "2. Extract key information from those paragraphs",
        "3. Combine the information to form a final answer",
        "4. Present your reasoning clearly with [STEP X] tags before each step",
        "5. End with FINAL: ",
        """
        EXAMPLE QUESTION: Which company is older, EOG Resources or General Mills?
        PARAGRAPHS:
        0: EOG Resources || Founded in 1999...
        1: General Mills || Founded in 1866...

        [STEP 1] Check founding dates in paragraph 0 and 1
        [STEP 2] EOG founded in 1999 (paragraph 0)
        [STEP 3] General Mills founded in 1866 (paragraph 1)
        FINAL: General Mills is older
        """
        "<|eot_id|>"
    ]
    
    # User message with question and paragraphs
    prompt.extend([
        "<|start_header_id|>user<|end_header_id|>",
        f"Here is the QUESTION you are supposed to answer: {question}",
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
    prompt.append(f"FINAL")
    return "\n".join(prompt)

def agot_inference(model, tokenizer, ex, tau_bits=TAU_BITS):
    #print("\n=== AGOT INFERENCE START ===")
    graph = build_graph(ex)
    #print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    prompt = build_llama3_prompt(ex.question, graph)
    #print("\n=== PROMPT BUILT ===")
    #print(f"Prompt preview:\n{prompt[:12000]}...\n")

    queue = deque([(prompt, [], 0.0)])
    completed = []

    while queue and len(completed) < 4 and len(queue) < MAX_NODES:
        current_prompt, chain, lp = queue.popleft()
        if len(chain) >= MAX_STEPS:
            print("Max steps reached.")
            continue

        inputs = tokenizer(current_prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[-1]

        #print("Tokenization complete.")

        try:
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEWTOK,
                    do_sample=False,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,  # Try with EOS set
                    early_stopping=False
                )

            new_tokens = outputs[0][input_len:]
            #print(f"Generated token IDs: {new_tokens}")
            #print(f"New tokens count: {len(new_tokens)}")

            if len(new_tokens) == 0:
                print("No tokens were generated.")
                continue

            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            #print("\n=== RAW GENERATED TEXT ===")
            #print(repr(new_text))

            cleaned_text = re.sub(r"^[0-9]+[^a-zA-Z]+|[^a-zA-Z]+$", "", new_text).strip()
            #print("\n=== CLEANED TEXT with REGEX: THE MODEL'S ANSWER===")
            #print(cleaned_text)
            return cleaned_text

            if not cleaned_text:
                print("Generated text is empty after cleaning.")
                continue

            if "FINAL:" in cleaned_text:
                answer = cleaned_text.split("FINAL:")[-1].strip()
                print(f"Found FINAL answer: {answer}")
                completed.append((chain, answer, lp))
                continue

            step_match = re.search(r"\[STEP\s+\d+\]\s*(\d+)", cleaned_text)
            if not step_match:
                print("No STEP pattern found in the output.")
                continue

            next_pid = int(step_match.group(1))
            print(f"Extracted next paragraph ID: {next_pid}")

            para = " ".join(graph.nodes[next_pid]["sentences"])
            next_prompt = (
                current_prompt + cleaned_text +
                f"\nPARA {next_pid}: {para}"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

            queue.append((next_prompt, chain + [next_pid], lp + 1.0))
            print(f"Enqueued next prompt with chain: {chain + [next_pid]}")

        except Exception as e:
            print(f"Generation error: {e}")
            continue
    #print("\n=== AGOT INFERENCE END ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--tau", type=float, default=TAU_BITS)
    args = parser.parse_args()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
    
        test_prompt = "What is the capital of France?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                #temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        clean_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model output: {clean_output}")

    # Load data
    examples = load_examples(str(PROC_DIR), DEV_SPLIT)
    # add length
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
        #print(f"GROUND TRUTH: {ex.answer}")
        references.append({
            "id": ex.qid,
            "answers": {
                "text": [ex.answer],
                "answer_start": [0]
            }
        })
    results = metric.compute(
        predictions=predictions,
        references=references
    )
    
    print(f"\nResults:", results)
    print(f"Exact Match: {results['exact']:.2f}%")
    print(f"F1: {results['f1']:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()