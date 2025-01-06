 
import os
import torch
from pathlib import Path
from peft import PeftModel
import h5py
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "/Users/agokrani/Documents/experiments/aideml/wsdm"  # Replace modal volume with local directory
HF_TOKEN = ""  # Add your Hugging Face token here or use environment variable

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.padding_side = "right"
tokenizer.pad_token_id = 128001

def generate_logits_for_batch(model, sequences):
    # Tokenize the batch of sequences
    inputs = tokenizer(sequences["text"], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print(len(inputs["input_ids"][0]))
   
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(outputs.logits)
    
    return outputs.logits

def tokenize(tokenizer, *args, **kwargs):
    def tokenize_func(examples):   
        texts = []
        for i in range(len(examples['prompt'])):
            texts.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": """Let's think step by step to judge which response is better for the given prompt. Please keep your thoughts clear and concise and at max around 300 words. The ouput should be in the following format:\n```## Rationale: <Your reasoning>\n## Winner: <model_a or model_b>```\n\n""",
                        },
                        {
                            "role": "user",
                            "content": f"Prompt: {examples['prompt'][i]}\n\nResponse A: ```{examples['response_a'][i]}```\n\nResponse B: ```{examples['response_b'][i]}```\n\n",
                        },
                        {
                            "role": "assistant",
                            "content": f"## Rationale: {examples['rationale'][i]}\n## Winner: {examples['winner'][i]}",
                        },
                    ],
                    tokenize=False,
                )
            )
        return {"text": texts}
    return tokenize_func

def gen_logits(dataset):
    # Setup model configuration
    # torch_dtype = torch.bfloat16
    # quant_storage_dtype = torch.bfloat16

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch_dtype,
    #     bnb_4bit_quant_storage=quant_storage_dtype,
    # )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        token=HF_TOKEN,
    )
        
    # Load PEFT model
    #peft_model_path = "lora_model/final-checkpoint-200"
    #model = PeftModel.from_pretrained(model, peft_model_path, device_map="auto")

    batch_size = 1
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = os.path.join(OUTPUT_DIR, "distillation_logits_test.h5")

    max_seq_len = 4096
    with h5py.File(file_name, "w") as hf:
        dset_logits = hf.create_dataset(
            "logits",
            shape=(len(dataset), max_seq_len, model.config.vocab_size),
            dtype=np.float16,
            chunks=(1, max_seq_len, model.config.vocab_size),
            compression="gzip",
            compression_opts=9
        )

        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating logits"):
            batch = dataset[i:i+batch_size]
            logits = generate_logits_for_batch(model, batch)

            print(logits)
            seq_len = logits.shape[1] 
            if seq_len > max_seq_len:
                dset_logits[i:i+batch_size, :max_seq_len, :] = np.zeros((batch_size, max_seq_len, model.config.vocab_size)).astype(np.float16)
            else: 
                dset_logits[i:i+batch_size, :seq_len, :] = logits.half().cpu().numpy()

def main():
    # Load dataset
    dataset = load_from_disk("/Users/agokrani/Documents/experiments/aideml/wsdm/wsdm2024-cot-dataset/shard_0")
   
    # Tokenize the dataset
    dataset = dataset.map(
        tokenize(tokenizer),
        batched=True,
    )
    
    # Select indices (optional, remove if you want to process the full dataset)
    select_indices = list(range(2))
    gen_logits(dataset.select(select_indices))

if __name__ == "__main__":
    main()
