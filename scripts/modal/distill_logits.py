import os
import modal
from pathlib import Path
from components.config import load_config
from scripts.local.distill_logits import train


VOL_MOUNT_PATH = Path("/vol")
output_vol = modal.Volume.from_name("finetune-volume", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate", "transformers", "torch", "datasets",
        "tensorboard", "trl==0.12.2", "peft", "bitsandbytes",
        "wheel", "tensorflow", "h5py", "tf-keras"
    ).run_commands("pip install flash-attn --no-build-isolation")
)

app = modal.App(name="distill-logits", image=image)

@app.function(
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=86400,
    volumes={VOL_MOUNT_PATH: output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_modal(config):    
    train(config)    
    output_vol.commit()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/default_config.json", help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)  # Will load default if args.config is None
    
    with app.run():
        train_modal.remote(config)

if __name__ == "__main__":
    main()
