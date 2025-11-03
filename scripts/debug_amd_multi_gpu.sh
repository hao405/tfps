#!/bin/bash

# AMD Multi-GPU Debugging Script
# This script helps debug multi-GPU training issues

echo "=== AMD Multi-GPU Debugging Script ==="
echo "Debugging multi-GPU training configuration"

# Function to check GPU status
check_gpu_status() {
    echo "=== GPU Status Check ==="
    if command -v rocm-smi > /dev/null 2>&1; then
        echo "Current GPU Status:"
        rocm-smi --showid --showmeminfo vram --showtemp --showpower
    else
        echo "rocm-smi not available"
    fi
}

# Function to test PyTorch GPU access
test_pytorch_gpu() {
    echo "=== PyTorch GPU Test ==="
    python -c "
import torch
import sys

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f'Number of GPUs: {device_count}')

    for i in range(device_count):
        try:
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f'GPU {i}: {name} ({memory_gb:.2f} GB)')

            # Test memory allocation
            try:
                test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                print(f'  ✓ GPU {i} memory allocation test passed')
                del test_tensor
            except Exception as e:
                print(f'  ✗ GPU {i} memory allocation failed: {e}')

        except Exception as e:
            print(f'  ✗ GPU {i} access failed: {e}')
else:
    print('No CUDA devices available')
    print('Available devices:', torch.cuda.device_count() if hasattr(torch.cuda, 'device_count') else 'Unknown')
"
}

# Function to test DataParallel
test_data_parallel() {
    echo "=== DataParallel Test ==="
    python -c "
import torch
import torch.nn as nn

try:
    # Simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    device_count = torch.cuda.device_count()
    print(f'Available GPUs: {device_count}')

    if device_count > 1:
        model = SimpleModel()

        # Test DataParallel
        try:
            model_parallel = nn.DataParallel(model, device_ids=list(range(device_count)))
            model_parallel = model_parallel.cuda()

            # Test forward pass
            test_input = torch.randn(32, 10).cuda()
            output = model_parallel(test_input)

            print(f'✓ DataParallel test passed with {device_count} GPUs')
            print(f'Input shape: {test_input.shape}')
            print(f'Output shape: {output.shape}')

        except Exception as e:
            print(f'✗ DataParallel test failed: {e}')
    else:
        print('Need at least 2 GPUs for DataParallel test')

except Exception as e:
    print(f'✗ DataParallel test error: {e}')
"
}

# Function to test distributed training
test_distributed() {
    echo "=== Distributed Training Test ==="
    python -c "
import torch
import torch.distributed as dist

try:
    print('Testing distributed training setup...')
    print(f'Distributed available: {dist.is_available()}')
    print(f'NCCL available: {dist.is_nccl_available()}')
    print(f'Gloo available: {dist.is_gloo_available()}')
    print(f'MPI available: {dist.is_mpi_available()}')

    if dist.is_available():
        print('✓ Distributed training is available')

        # Test backend initialization
        if dist.is_nccl_available():
            print('✓ NCCL backend available')
        if dist.is_gloo_available():
            print('✓ Gloo backend available')
    else:
        print('✗ Distributed training not available')

except Exception as e:
    print(f'✗ Distributed test error: {e}')
"
}

# Function to test memory usage
test_memory_usage() {
    echo "=== Memory Usage Test ==="
    python -c "
import torch
import gc

try:
    device_count = torch.cuda.device_count()
    print(f'Testing memory usage on {device_count} GPUs...')

    for i in range(device_count):
        try:
            torch.cuda.set_device(i)

            # Get memory info
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3

            print(f'GPU {i}:')
            print(f'  Total memory: {total:.2f} GB')
            print(f'  Allocated: {allocated:.2f} GB')
            print(f'  Reserved: {reserved:.2f} GB')
            print(f'  Free: {total - reserved:.2f} GB')

            # Test large tensor allocation
            try:
                large_tensor = torch.randn(10000, 10000, device=f'cuda:{i}')
                print(f'  ✓ Large tensor allocation successful')
                del large_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'  ✗ Large tensor allocation failed: {e}')

        except Exception as e:
            print(f'  ✗ GPU {i} memory test failed: {e}')

except Exception as e:
    print(f'✗ Memory test error: {e}')
"
}

# Function to test training script
test_training_script() {
    echo "=== Training Script Test ==="
    echo "Testing minimal training configuration..."

    export MIOPEN_DISABLE_CACHE=1
    export MIOPEN_DEBUG_DISABLE_FIND_DB=1
    export HIP_VISIBLE_DEVICES="0"

    timeout 30 python -u ../run_longExp.py \
      --random_seed 2023 \
      --is_training 1 \
      --root_path ../dataset/Solar/ \
      --data_path solar_AL.txt \
      --model_id solar_debug_96_96 \
      --model PatchTST_MoE_cluster \
      --data solar \
      --features M \
      --target 0 \
      --seq_len 96 \
      --pred_len 96 \
      --enc_in 137 \
      --c_out 137 \
      --e_layers 1 \
      --n_heads 4 \
      --d_model 8 \
      --d_ff 16 \
      --dropout 0.1 \
      --fc_dropout 0.1 \
      --head_dropout 0 \
      --patch_len 8 \
      --stride 4 \
      --T_num_expert 2 \
      --T_top_k 1 \
      --F_num_expert 2 \
      --F_top_k 1 \
      --beta 0.1 \
      --des 'Debug' \
      --train_epochs 1 \
      --devices 0 \
      --use_gpu True \
      --gpu 0 \
      --itr 1 \
      --batch_size 8 \
      --learning_rate 0.001 \
      --patience 1 2>&1 || echo "Test completed (timeout or error expected)"
}

# Main execution
echo "Starting AMD Multi-GPU debugging..."
echo "Timestamp: $(date)"

# Run all tests
check_gpu_status
echo ""
test_pytorch_gpu
echo ""
test_data_parallel
echo ""
test_distributed
echo ""
test_memory_usage
echo ""
test_training_script

echo ""
echo "=== Debugging Summary ==="
echo "Check the output above for any errors or warnings."
echo "Common issues:"
echo "1. GPUs not detected: Check ROCm installation and GPU drivers"
echo "2. Memory allocation errors: Reduce batch size or model size"
echo "3. DataParallel errors: Ensure all GPUs are accessible"
echo "4. Training script errors: Check data paths and model parameters"
echo ""
echo "For multi-GPU training, ensure:"
echo "- All GPUs are visible to PyTorch"
echo "- Sufficient memory on each GPU"
echo "- Correct environment variables are set"
echo "- Data paths are correct"""}