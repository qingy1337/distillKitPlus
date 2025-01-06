DEFAULT_CONFIG = {
    "project_name": "distil-logits",
    "dataset": {
        "name": None,
        "split": None,
        "logits_file": None,
        "num_samples": None,
        "select_range": None,
    },
    "models": {
        "teacher": None,
        "student": None,
        "student_adapter": None,
        "teacher_adapter": None,
        "teacher_vocab_size": None
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": None,
        "pad_token_id": 128001
    },
    "training": {
        "output_dir": None,
        "num_train_epochs": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 100,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": True,
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
        "enable_training": True,
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
    }
}
