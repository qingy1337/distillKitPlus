DEFAULT_CONFIG = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "/vol/wsdm2024-cot-dataset/shard_0",
        "split": None,
        "logits_file": "/vol/lora_model/distillation_logits-debug.tfrecord",
        "num_samples": 1,
        "select_range": (100,101),
    },
    "models": {
        "teacher": "meta-llama/Llama-3.1-70B-Instruct", # only needed if logits_file is not provided
        "student": "meta-llama/Llama-3.1-8B-Instruct", # The student model to train
        "student_adapter": "/vol/lora_model/checkpoint-6016",  # Path to the student LoRA adapter
        "teacher_adapter": "/vol/lora_model/final-checkpoint-200",
        "teacher_vocab_size": None
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": None,
        "pad_token_id": 128001
    },
    "training": {
        "output_dir": "/vol/distilled_model/debug",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 100,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.1
    },
    "model_config": {
        "use_flash_attention": True
    },
    "lora": {
        "enable_training": True,
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
    },
    "quantization": {
        "enabled": True,
    }
}
