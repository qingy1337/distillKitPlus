import os
import modal
from pathlib import Path
from components.config import DEFAULT_CONFIG
from components.models import load_models
from components.dataset import DistillationDataset
from components.trainer import LogitsTrainer
from components.formatters import comparison_format

from transformers import DataCollatorForCompletionOnlyLM, SFTConfig

VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate", "transformers", "torch", "datasets",
        "tensorboard", "trl==0.12.2", "peft", "bitsandbytes",
        "wheel", "tensorflow", "h5py"
    ).run_commands("pip install flash-attn --no-build-isolation")
)

app = modal.App(name="distill-logits", image=image)

@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train(config=None):
    if config is None:
        config = DEFAULT_CONFIG
    
    # Load models and tokenizer
    models = load_models(config)
    student_model = models["student_model"]
    student_tokenizer = models["student_tokenizer"]
    teacher_model = models.get("teacher_model")
    
    # Initialize dataset
    dataset = DistillationDataset(
        file_path=config["dataset"]["name"],
        tokenizer=student_tokenizer,
        max_seq_length=config["tokenizer"]["max_length"],
        teacher_vocab_size=config["models"]["teacher_vocab_size"],
        format_func=comparison_format(student_tokenizer),
        split=config["dataset"]["split"],
        num_samples=config["dataset"]["num_samples"],
        select_range=config["dataset"].get("select_range"),
        logits_file=config["dataset"]["logits_file"]
    )

    # Training arguments
    training_args = SFTConfig(
        **config["training"],
        max_seq_length=config["tokenizer"]["max_length"],
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=dataset,
        tokenizer=student_tokenizer,
        args=training_args,
        data_collator=DataCollatorForCompletionOnlyLM(
            "<|im_start|>assistant\n",
            tokenizer=student_tokenizer
        ),
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"]
    )

    if teacher_model is not None:
        trainer.teacher_model = teacher_model

    # Train and save
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    trainer.save_model(os.path.join(VOL_MOUNT_PATH, "final_model"))
    output_vol.commit()

def main():
    config = DEFAULT_CONFIG
    config.update({
        "models": {
            "teacher": None,  # Not needed if using logits file
            "student": "meta-llama/Llama-3.1-8B-Instruct",
            "teacher_vocab_size": 32000
        },
        "dataset": {
            "name": "/vol/dataset",
            "logits_file": "/vol/logits/distillation_logits.tfrecord",
            "num_samples": 100
        }
    })
    
    with app.run():
        train.remote(config)

if __name__ == "__main__":
    main()
