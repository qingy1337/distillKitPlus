import os
import yaml
import h5py
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Configuration
config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "/Users/agokrani/Documents/experiments/aideml/wsdm/wsdm2024-cot-dataset/shard_0",
        "split": None, # None for dataset from the disk or a split name
        "logits_file": "/Users/agokrani/Documents/experiments/aideml/wsdm/distillation_logits_test.h5", # Path to the precomputed logits file for distillation in h5 format. Make sure to use teacher model with same vocab. 
        "num_samples": 2, # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "meta-llama/Llama-3.1-70B-Instruct", # only needed if logits_file is not provided
        "student": "Qwen/Qwen2.5-0.5B", # The student model to train
        "student_adapter": None,  # Path to the student LoRA adapter
        "teacher_adapter": None  # Path to teacher's LoRA adapter
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "pad_token_id": None
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": False
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
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
        "target_modules": "all-linear"
    }
}


from components.formatters import comparison_format, sharegpt_format

class DistillationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length, format_func=None, split=None, num_samples=None, logits_file=None, ignore_index=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.logits_file = logits_file
        self.ignore_index = ignore_index
        self.split = split
        self.format_func = format_func or self._default_format

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
        if self.logits_file:
            self._load_h5()
        
        if num_samples:
            self.dataset = self.dataset.select(range(num_samples))
            self.logits = self.logits[:num_samples]
        assert len(self.dataset) == len(self.logits), "Number of samples in dataset and logits file do not match."
    
    def _load_h5(self):
        # Open the file if it's not already opened
        h5_file = h5py.File(self.logits_file, 'r')
        self.logits = h5_file['logits']
        

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
                "logits": self.logits[index]
            }
        
        return {
            "input_ids": self.dataset[index]["input_ids"],
            "attention_mask": self.dataset[index]["attention_mask"]
        }

    def _default_format(self, examples):
        """Default formatting function that just passes through the text"""
        return {"text": examples["text"] if "text" in examples else examples}


# # Set up environment
accelerator = Accelerator()
device = accelerator.device

# Load tokenizers
if config["models"]["teacher"] is None:
    assert config["dataset"]["logits_file"] is not None, "Logits file is required if teacher model is not provided."

student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

student_tokenizer.padding_side = "right"
student_tokenizer.pad_token_id = config["tokenizer"]["pad_token_id"] if config["tokenizer"]["pad_token_id"] else student_tokenizer.eos_token_id

# Apply chat template to student tokenizer
if config["tokenizer"]["chat_template"]:
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

# # Load and preprocess dataset
# Initialize dataset with comparison format by default
dataset = DistillationDataset(
    file_path=config["dataset"]["name"],
    tokenizer=student_tokenizer,
    max_seq_length=config["tokenizer"]["max_length"],
    format_func=comparison_format(student_tokenizer),
    split=config["dataset"]["split"],
    num_samples=config["dataset"]["num_samples"],
    logits_file=config["dataset"]["logits_file"]
)

# # Preprocess and tokenize the dataset
# print("Preprocessing and tokenizing dataset...")
# original_columns = dataset.column_names
# dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

# def tokenize_function(examples):
#     return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

# tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
# tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"    

# For teacher model loading
if config["models"]["teacher"] is not None and config["dataset"]["logits_file"] is None:
    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
    if config["models"]["teacher_adapter"] is not None:
        teacher_model = PeftModel.from_pretrained(
            teacher_model, 
            config["models"]["teacher_adapter"]
        )

# For student model loading
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

if config["models"]["student_adapter"] is not None:
    # If student adapter is provided, use it and warn about ignoring other settings
    print("WARNING: Student adapter provided. Using adapter settings from the loaded adapter.")
    print("WARNING: Ignoring LoRA config and spectrum settings.")
    student_model = PeftModel.from_pretrained(
        student_model, 
        config["models"]["student_adapter"]
    )
elif config["lora"]["enable_training"]:
    # If LoRA training is enabled, it takes precedence over spectrum
    if "spectrum" in config:
        print("WARNING: LoRA training is enabled. Ignoring spectrum configuration.")
    
    # Configure new LoRA for training
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )
    student_model = get_peft_model(student_model, lora_config)
    
    # Ensure all LoRA parameters are trainable
    for name, param in student_model.named_parameters():
        if "lora" in name.lower():
            if not param.requires_grad:
                param.requires_grad = True
                print(f"Set {name} to trainable")
    
    print("Initialized new LoRA configuration for training.")
    print("Verified all LoRA parameters are trainable.")

# Handle spectrum-based layer freezing if LoRA is not being used
if not config["lora"]["enable_training"] and config["models"]["student_adapter"] is None:
    if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
        def freeze_student_spectrum(model, unfrozen_layers_file):
            with open(unfrozen_layers_file, 'r') as file:
                unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
            
            for name, param in model.named_parameters():
                if not any(layer in name for layer in unfrozen_layers):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # Apply freezing to student model
        freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
        print("Applied spectrum-based layer freezing configuration.")
    else:
        print("No spectrum configuration found. All layers of the student model will be trainable.")


from components.trainer import LogitsTrainer

    # Set up training arguments
    training_arguments = SFTConfig(**config["training"])
    training_arguments.max_seq_length = config["tokenizer"]["max_length"]
    training_arguments.remove_unused_columns = False
    training_arguments.temperature = config["distillation"]["temperature"]
    training_arguments.alpha = config["distillation"]["alpha"]

    # Initialize trainer
    trainer = LogitsTrainer(  # type: ignore
        model=student_model,
        train_dataset=dataset,
        tokenizer=student_tokenizer,
        data_collator=DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=student_tokenizer),
        args=training_arguments,
    )

    # Add teacher model if needed
    if teacher_model is not None:
        trainer.teacher_model = teacher_model

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    trainer.save_model(config["training"]["output_dir"])

def main():
    # Set up training arguments
    training_arguments = SFTConfig(**config["training"])
    training_arguments.max_seq_length = config["tokenizer"]["max_length"]
    training_arguments.remove_unused_columns = False
    training_arguments.temperature = config["distillation"]["temperature"]
    training_arguments.alpha = config["distillation"]["alpha"]

    # Initialize trainer
    trainer = LogitsTrainer(  # type: ignore
        model=student_model,
        train_dataset=dataset,
        tokenizer=student_tokenizer,
        data_collator=DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=student_tokenizer),
        args=training_arguments,
    )

    # Add teacher model if needed
    if teacher_model is not None:
        trainer.teacher_model = teacher_model

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    trainer.save_model(config["training"]["output_dir"])

if __name__ == "__main__":
    main()
