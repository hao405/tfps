#!/usr/bin/env python3
"""
AMD Multi-GPU Training Script for PatchTST MoE Cluster
This script properly configures AMD GPUs for multi-GPU training using ROCm.
"""

import os
import sys
import torch
import subprocess

def check_amd_gpu_setup():
    """Check and validate AMD GPU setup"""
    print("=== AMD GPU Setup Check ===")

    # Check PyTorch ROCm support
    if torch.version.hip is not None:
        print(f"✓ PyTorch ROCm support detected: {torch.version.hip}")
    else:
        print("✗ PyTorch ROCm support not detected")
        return False

    # Check available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    if gpu_count == 0:
        print("✗ No GPUs detected")
        return False

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")

    return True

def set_amd_environment():
    """Set AMD ROCm environment variables"""
    print("\n=== Setting AMD Environment ===")

    # Essential AMD ROCm environment variables
    env_vars = {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",  # Use all available GPUs
        "MIOPEN_DISABLE_CACHE": "1",
        "MIOPEN_SYSTEM_DB_PATH": "",
        "HSA_USERPTR_FOR_PAGED_MEM": "0",
        "ROCM_PATH": "/opt/rocm"
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")

def run_training():
    """Run the training with AMD multi-GPU setup"""
    print("\n=== Starting AMD Multi-GPU Training ===")

    # Training configuration for solar dataset
    training_config = {
        "random_seed": 2023,
        "is_training": 1,
        "root_path": "./dataset/Solar/",
        "data_path": "solar_AL.txt",
        "model_id": "solar_96_96",
        "model": "PatchTST_MoE_cluster",
        "data": "solar",
        "features": "M",
        "target": "0",
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 137,
        "c_out": 137,
        "e_layers": 3,
        "n_heads": 4,
        "d_model": 16,
        "d_ff": 64,
        "dropout": 0.3,
        "fc_dropout": 0.3,
        "head_dropout": 0,
        "patch_len": 16,
        "stride": 8,
        "T_num_expert": 16,
        "T_top_k": 1,
        "F_num_expert": 16,
        "F_top_k": 1,
        "beta": 0.1,
        "des": "Exp",
        "train_epochs": 100,
        "devices": "0,1,2,3,4,5,6,7",
        "use_multi_gpu": True,
        "use_gpu": True,  # Set to True for AMD GPUs
        "itr": 1,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "num_workers": 10,
        "patience": 15,
        "lradj": "type3",
        "pct_start": 0.3,
        "use_amp": False,
        "revin": 1,
        "affine": 0,
        "subtract_last": 0,
        "decomposition": 0,
        "kernel_size": 25,
        "individual": 1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "do_predict": False,
        "loss": "mse",
        "checkpoints": "./checkpoints/",
        "test_flop": False
    }

    # Build command
    cmd = ["python", "run_longExp.py"]

    for key, value in training_config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    print(f"Running command: {' '.join(cmd)}")

    # Execute training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Error running training: {e}")

def main():
    """Main function"""
    print("AMD Multi-GPU Training for PatchTST MoE Cluster")
    print("=" * 50)

    # Check GPU setup
    if not check_amd_gpu_setup():
        print("AMD GPU setup check failed. Exiting.")
        sys.exit(1)

    # Set environment
    set_amd_environment()

    # Verify GPU detection after environment setup
    print("\n=== Verifying GPU Detection ===")
    gpu_count = torch.cuda.device_count()
    print(f"PyTorch detected {gpu_count} GPUs")

    if gpu_count == 0:
        print("✗ No GPUs detected after environment setup")
        sys.exit(1)

    # Run training
    run_training()

    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()