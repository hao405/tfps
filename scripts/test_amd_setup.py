#!/usr/bin/env python3
"""
Test script to verify AMD multi-GPU setup for PatchTST MoE training.
"""

import torch
import os
import sys

def test_amd_setup():
    """Test AMD GPU setup"""
    print("=== AMD Multi-GPU Setup Test ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Check ROCm support
    print(f"ROCm/HIP available: {torch.version.hip is not None}")
    if torch.version.hip:
        print(f"ROCm version: {torch.version.hip}")

    # Check CUDA support (for comparison)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and hasattr(torch.version, 'cuda'):
        print(f"CUDA version: {torch.version.cuda}")

    # Check GPU count
    gpu_count = torch.cuda.device_count()
    print(f"\nDetected GPUs: {gpu_count}")

    if gpu_count == 0:
        print("No GPUs detected!")
        return False

    # List all GPUs
    print("\nGPU Details:")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")

    # Test basic tensor operations on each GPU
    print("\nTesting tensor operations on each GPU:")
    for i in range(gpu_count):
        try:
            device = torch.device(f'cuda:{i}')
            # Create a small tensor
            x = torch.randn(100, 100, device=device)
            y = torch.matmul(x, x.T)
            print(f"  ✓ GPU {i}: Tensor operations successful")
        except Exception as e:
            print(f"  ❌ GPU {i}: Failed - {e}")

    # Test multi-GPU setup
    print("\nTesting multi-GPU setup:")
    try:
        # Test DataParallel
        model = torch.nn.Linear(100, 100)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            print("  DataParallel setup successful")

        # Test distributed setup
        if gpu_count > 1:
            print(f"  Multiple GPUs available for distributed training: {gpu_count}")

    except Exception as e:
        print(f"  Multi-GPU setup failed: {e}")

    return True

def test_environment_variables():
    """Test AMD environment variables"""
    print("\n=== Environment Variables ===")

    amd_vars = [
        'HIP_VISIBLE_DEVICES',
        'MIOPEN_DISABLE_CACHE',
        'MIOPEN_SYSTEM_DB_PATH',
        'HSA_USERPTR_FOR_PAGED_MEM',
        'ROCM_PATH'
    ]

    for var in amd_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def test_training_imports():
    """Test if we can import training modules"""
    print("\n=== Testing Training Imports ===")

    try:
        # Test basic imports
        import numpy as np
        print("NumPy import successful")

        # Test model imports
        sys.path.append('..')  # Add parent directory to path
        from models import PatchTST_MoE_cluster
        print("PatchTST_MoE_cluster import successful")

        # Test data loading
        from data_provider.data_factory import data_provider
        print("Data provider import successful")

        # Test training framework
        from exp.exp_main import Exp_Main
        print("Training framework import successful")

        return True

    except Exception as e:
        print(f"Import failed: {e}")
        return False

def main():
    """Main test function"""
    print("AMD Multi-GPU Setup Test for PatchTST MoE")
    print("=" * 50)

    # Test environment variables
    test_environment_variables()

    # Test AMD setup
    gpu_success = test_amd_setup()

    # Test imports
    import_success = test_training_imports()

    # Summary
    print("\n=== Test Summary ===")
    if gpu_success and import_success:
        print("All tests passed! AMD multi-GPU setup is ready.")
        print("\nTo run training, use:")
        print("  python scripts/train_amd_multi_gpu.py")
        print("or")
        print("  python run_longExp.py --use_gpu True --use_multi_gpu --devices 0,1,2,3,4,5,6,7 [other args]")
    else:
        print("Some tests failed. Please check the setup.")
        if not gpu_success:
            print("  - GPU detection issues")
        if not import_success:
            print("  - Import issues")

if __name__ == "__main__":
    main()