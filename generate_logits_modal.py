import os
import modal
import modal.config
import torch

from pathlib import Path
import modal.gpu
from peft import PeftModel

import tensorflow as tf
import torch
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


VOL_MOUNT_PATH = Path("/vol")


BASE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
cache_dir = "base_model"

output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate",
        "transformers",
        "torch",
        "datasets",
        "tensorboard",
        "trl==0.12.2",
        "peft",
        "bitsandbytes",
        "wheel",
        "tensorflow",
        "h5py"
    ).run_commands(  # add flash-attn
        "pip install flash-attn --no-build-isolation"
    )
)
app = modal.App(name="llama3.1-70b28b-cot-distillation-wsdm", image=image)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=modal.Secret.from_name("huggingface-secret"))
tokenizer.padding_side = "right"
tokenizer.pad_token_id = 128001



def generate_logits_for_batch(model, sequences, max_seq_len):
    # Tokenize the batch of sequences
    inputs = tokenizer(
        sequences["text"], 
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        padding=False,
        max_length=max_seq_len,
        return_overflowing_tokens=False,
        return_length=False,
    )
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print(len(inputs["input_ids"][0]))
    if len(inputs["input_ids"][0]) > max_seq_len:
        logits = torch.zeros((len(inputs["input_ids"]), max_seq_len, model.config.vocab_size), dtype=torch.float16)
    else: 
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
    print(logits)
    
    return logits


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

@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def gen_logits(dataset):
    dataset = Dataset.from_list(dataset)

    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    # quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch_dtype,
    #         bnb_4bit_quant_storage=quant_storage_dtype,
    # )
    cache_dir2 = str(Path(VOL_MOUNT_PATH) / cache_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL if not os.path.exists(cache_dir2) else cache_dir2,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        #quantization_config=quantization_config,
        torch_dtype=quant_storage_dtype, 
        attn_implementation = "flash_attention_2"
    )
    
    if not os.path.exists(cache_dir2): 
        model.save_pretrained(cache_dir2)
    
        output_vol.commit()
    
    peft_model_path = os.path.join(VOL_MOUNT_PATH, "lora_model/final-checkpoint-200")
    model = PeftModel.from_pretrained(model, peft_model_path, device_map="auto")

    #max_new_tokens = max([len(tokens) for tokens in dataset["tokens"]])
    # Process the dataset in batches
    batch_size = 1 # Adjust as needed


    file_name = os.path.join(VOL_MOUNT_PATH, "lora_model/distillation_logits-1.tfrecord")
    max_seq_len = 4096
    
    try:
        with tf.io.TFRecordWriter(file_name) as writer:
            for i in tqdm(range(0, len(dataset), batch_size), desc="Generating logits"):
                batch = dataset[i:i+batch_size]
                logits = generate_logits_for_batch(model, batch, max_seq_len)
                
                # Convert to float16 and move to CPU
                logits = logits.half().cpu().numpy()
                
                # Only store the actual sequence length, no padding
                actual_seq_len = logits.shape[1]
                if actual_seq_len > max_seq_len:
                    logits = logits[:, :max_seq_len, :]
                
                # Create TFRecord Example with minimal features
                feature = {
                    'logits': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[logits.tobytes()]
                    )),
                    'seq_len': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[logits.shape[1]]
                    ))
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    finally:
        # Ensure the volume is committed even if an error occurs
        output_vol.commit()

def main():
    # Load dataset
    dataset = load_from_disk("/Users/agokrani/Documents/experiments/aideml/wsdm/wsdm2024-cot-dataset/shard_0")
   
    # Tokenize the dataset
    dataset = dataset.map(
        tokenize(tokenizer),
        batched=True,
    )
    select_indicies = list(range(100, 200))
    with app.run():    
        gen_logits.remote(dataset.select(select_indicies).to_list())

if __name__ == "__main__":
    main()
