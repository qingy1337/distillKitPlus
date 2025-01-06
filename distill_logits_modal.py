import os
import h5py
import modal
import torch
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig


VOL_MOUNT_PATH = Path("/vol")
CACHE_DIR = "base_model"

# Set up Modal image and volumes
cuda_version = "12.4.0"
flavor = "devel"
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
        "tf-keras",
        "h5py"
    ).run_commands(
        "pip install flash-attn --no-build-isolation"
    )
)

app = modal.App(name="distillation-training", image=image)
output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)


# Configuration
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "/vol/wsdm2024-cot-dataset/shard_0",
        "split": None, # None for dataset from the disk or a split name
        "logits_file": "/vol/lora_model/distillation_logits-1.tfrecord", # Path to the precomputed logits file for distillation in h5 format. Make sure to use teacher model with same vocab. 
        "num_samples": 100, # You can pass a number here to limit the number of samples to use.
        "select_range": (100, 200), # Tuple to select specific sample range of the dataset. Make sure the num_samples matches the total elements in the range.
    },
    "models": {
        "teacher": "meta-llama/Llama-3.1-70B-Instruct", # only needed if logits_file is not provided
        "student": "meta-llama/Llama-3.1-8B-Instruct", # The student model to train
        "student_adapter": "/vol/lora_model/checkpoint-6016",  # Path to the student LoRA adapter
        "teacher_adapter": None, # Path to teacher's LoRA adapter
        "teacher_vocab_size": None
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": None,
        "pad_token_id": 128001
    },
    "training": {
        "output_dir": "/vol/distilled_model",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 100,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": True,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": False,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.1
    },
    "model_config": {
        "use_flash_attention": False
    },
    "lora": {
        "enable_training": True,  # Set to True to enable LoRA training for student model
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
    }
}

def comparison_format(tokenizer):
    """Format function for comparison-style datasets"""
    def format_func(examples):   
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
                    tokenize=False
                )
            )
        return {"text": texts}
    return format_func

class TFRecordDataLoader:
    def __init__(self, file_path, vocab_size, valid_indices=None, num_samples=None):
        """
        Initializes the DataLoader for TFRecord files, filtering based on valid indices if provided.
        
        Args:
            file_path (str): Path to the TFRecord file.
            vocab_size (int): Vocabulary size for reshaping logits.
            valid_indices (list[int], optional): List of valid indices to filter the dataset.
        """
        self.file_path = file_path
        self.feature_description = {
            'logits': tf.io.FixedLenFeature([], tf.string),
            'seq_len': tf.io.FixedLenFeature([], tf.int64)
        }
        self.vocab_size = vocab_size
        self.valid_indices = valid_indices
        self.num_samples = num_samples

        # Index records while pinning dataset operations to the CPU
        with tf.device('/cpu:0'):
            self.dataset = tf.data.TFRecordDataset(self.file_path)
            self.record_offsets = self._index_tfrecord()

    def _parse_function(self, record):
        """Parses a single TFRecord example."""
        parsed_features = tf.io.parse_single_example(record, self.feature_description)
        seq_len = parsed_features['seq_len'].numpy()
        logits_raw = parsed_features['logits'].numpy()

        # Decode the logits
        logits = np.frombuffer(logits_raw, dtype=np.float16)
        logits = logits.reshape((seq_len, self.vocab_size))

        return {'logits': logits, 'seq_len': seq_len}
    
    def _index_tfrecord(self):
        """
        Indexes the TFRecord file to calculate offsets for each record,
        filtering based on valid indices if provided.
        
        Returns:
            list[int]: List of indices for valid records in the dataset.
        """
        offsets = []
        with tf.device('/cpu:0'):
            if self.num_samples is not None:
                target_samples = self.num_samples
            else:
                target_samples = sum(1 for _ in self.dataset)
            
            for index, record in enumerate(self.dataset):
                if len(offsets) >= target_samples:
                    break
                if self.valid_indices is None or index in self.valid_indices:
                    offsets.append(index)
        
        return offsets

    def __len__(self):
        """Returns the total number of valid records in the dataset."""
        return len(self.record_offsets)

    def __getitem__(self, idx):
        """
        Returns the record(s) at the given valid index.
        
        Args:
            idx (int): Index of the record to retrieve.
        
        Returns:
            dict: Parsed record (logits and sequence length).
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.record_offsets):
                raise IndexError("Index out of range")
            offset = self.record_offsets[idx]
            with tf.device('/cpu:0'):
                for i, record in enumerate(self.dataset):
                    if i == offset:
                        return self._parse_function(record)
            raise IndexError("Index out of range")

        else:
            raise TypeError("Index must be an int")

class DistillationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length, teacher_vocab_size, format_func=None, split=None, num_samples=None, logits_file=None, select_range=None, ignore_index=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.logits_file = logits_file
        self.ignore_index = ignore_index
        self.split = split
        self.format_func = format_func or self._default_format
        self.teacher_vocab_size = teacher_vocab_size
        self.select_range = select_range

        if self.split is None: 
            dataset = load_from_disk(file_path)
        else: 
            dataset = load_dataset(file_path, split=self.split)
        
        # Apply formatting function
        self.dataset = dataset.map(
            self.format_func,
            batched=True,
        )
        
        # Apply tokenization
        self.dataset = self.dataset.map(
            self._tokenize,
            batched=True,
        )

        if num_samples:
            if self.select_range:
                samples_to_select = list(range(self.select_range[0], self.select_range[1]))
                assert num_samples == len(samples_to_select)
                self.dataset = self.dataset.select(samples_to_select)
            else: 
                self.dataset = self.dataset.select(range(num_samples))
        
        self._compute_valid_indices()
        
        if self.logits_file:
            self.logits = TFRecordDataLoader(
                self.logits_file,
                self.teacher_vocab_size, 
                valid_indices=self.valid_indices,
                num_samples=num_samples
            )
        self.dataset = self.dataset.select(self.valid_indices)
        
        print(f"dataset length: {len(self.dataset)}")
        print(f"logits length: {len(self.logits)}")
        assert len(self.dataset) == len(self.logits), "Number of samples in dataset and logits file do not match."
    
    def _compute_valid_indices(self):
        self.valid_indices = []
        for idx, example in enumerate(self.dataset):
            seq_length = len(example["input_ids"])
            if seq_length <= self.max_seq_length:  # Add constraints as needed
                self.valid_indices.append(idx)


    def _tokenize(self, element): 
        outputs = self.tokenizer(
            element["text"],
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=self.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"], 
            "attention_mask": outputs["attention_mask"]
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.logits_file:
            return {
                "input_ids": self.dataset[index]["input_ids"],
                "attention_mask": self.dataset[index]["attention_mask"],
                "logits": self.logits[index]["logits"]
            }
        
        return {
            "input_ids": self.dataset[index]["input_ids"],
            "attention_mask": self.dataset[index]["attention_mask"]
        }

    def _default_format(self, examples):
        """Default formatting function that just passes through the text"""
        return {"text": examples["text"] if "text" in examples else examples}

from components.trainer import LogitsTrainer

@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train():
    # Set up device and data types
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    # Initialize tokenizers with proper checks
    if config["models"]["teacher"] is None:
        assert config["dataset"]["logits_file"] is not None, "Logits file is required if teacher model is not provided."
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"],
        token=os.environ["HF_TOKEN"]
    )
    student_tokenizer.padding_side = "right"
    student_tokenizer.pad_token_id = config["tokenizer"]["pad_token_id"] if config["tokenizer"]["pad_token_id"] else student_tokenizer.eos_token_id

    # Apply chat template if specified
    if config["tokenizer"].get("chat_template"):
        student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Set up quantization config for models
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
    )
    
    
    # Load models with configurable flash attention
    model_kwargs = {
        "device_map": "auto",
        "token": os.environ["HF_TOKEN"],
        "quantization_config": quantization_config,
        "torch_dtype": quant_storage_dtype,
        "attn_implementation": "flash_attention_2" if config["model_config"]["use_flash_attention"] else "sdpa"
    }

    # Load teacher model if needed
    teacher_model = None
    if config["models"]["teacher"] is not None and config["dataset"]["logits_file"] is None:
        teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
        if config["models"]["teacher_adapter"] is not None:
            teacher_model = PeftModel.from_pretrained(
                teacher_model, 
                config["models"]["teacher_adapter"]
            )

    # Load student model
    model_path = config["models"]["student"]
    if os.path.exists(str(Path(VOL_MOUNT_PATH) / CACHE_DIR / config["models"]["student"])):
        model_path = str(Path(VOL_MOUNT_PATH) / CACHE_DIR / config["models"]["student"])
    
    student_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    student_model = prepare_model_for_kbit_training(student_model)
    if not os.path.exists(str(Path(VOL_MOUNT_PATH) / CACHE_DIR / config["models"]["student"])): 
        student_model.save_pretrained(str(Path(VOL_MOUNT_PATH) / CACHE_DIR / config["models"]["student"]))
    
        output_vol.commit()

    # Handle student model configuration (LoRA or adapter)
    if config["models"]["student_adapter"] is not None:
        print("WARNING: Student adapter provided. Using adapter settings from the loaded adapter.")
        print("WARNING: Ignoring LoRA config and spectrum settings.")
        student_model = PeftModel.from_pretrained(
            student_model, 
            config["models"]["student_adapter"]
        )

        # Ensure all LoRA parameters are trainable
        for name, param in student_model.named_parameters():
            if "lora" in name.lower():
                if not param.requires_grad:
                    param.requires_grad = True
                    print(f"Set {name} to trainable")

    elif config["lora"]["enable_training"]:
        if "spectrum" in config:
            print("WARNING: LoRA training is enabled. Ignoring spectrum configuration.")
        
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["dropout"],
            bias=config["lora"]["bias"],
            task_type=config["lora"]["task_type"]
        )
        student_model = get_peft_model(student_model, lora_config)
        
        
    
    # Handle spectrum-based layer freezing if neither LoRA nor adapter is used
    elif not config["lora"]["enable_training"] and config["models"]["student_adapter"] is None:
        if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
            with open(config["spectrum"]["layers_to_unfreeze"], 'r') as file:
                unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
            
            for name, param in student_model.named_parameters():
                if not any(layer in name for layer in unfrozen_layers):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("Applied spectrum-based layer freezing configuration.")
        else:
            print("No spectrum configuration found. All layers of the student model will be trainable.")

    # Ideally teacher vocab size should be provided in the config file. If not we will infer it from teacher model or student model depending on 
    # the configuration. Note: If teacher model is not loaded due to logits file. It will be inferred from student model.
    if config["models"]["teacher_vocab_size"] is not None: 
        teacher_vocab_size = config["models"]["teacher_vocab_size"]    
    elif teacher_model is not None:
        teacher_vocab_size = teacher_model.config.vocab_size
    else: 
        teacher_vocab_size = student_model.config.vocab_size 

    # Initialize dataset
    dataset = DistillationDataset(
        file_path=config["dataset"]["name"],
        tokenizer=student_tokenizer,
        max_seq_length=config["tokenizer"]["max_length"],
        teacher_vocab_size=teacher_vocab_size,
        format_func=comparison_format(student_tokenizer),
        split=config["dataset"]["split"],
        num_samples=config["dataset"]["num_samples"],
        select_range=config["dataset"]["select_range"],
        logits_file=config["dataset"]["logits_file"]
    )

    # # Training arguments
    training_arguments = SFTConfig(**config["training"])
    training_arguments.max_seq_length=config["tokenizer"]["max_length"]
    training_arguments.remove_unused_columns=False

    # Initialize trainer
    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=dataset,
        tokenizer=student_tokenizer,
        args=training_arguments,
        data_collator=DataCollatorForCompletionOnlyLM("<|start_header_id|>assistant<|end_header_id|>\n\n", tokenizer=student_tokenizer),
    )

    # Add teacher model to trainer if needed
    if teacher_model is not None:
        trainer.teacher_model = teacher_model

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    save_path = Path(config["training"]["output_dir"]) / "final-distilled-checkpoint"
    trainer.save_model(save_path)
    
    # Commit changes to volume
    output_vol.commit()

def main():
    with app.run():
        train.remote()

if __name__ == "__main__":
    main()
