#!/bin/bash

# AMD GPU Setup Test Script
# This script tests AMD GPU configuration for multi-GPU training

echo "=== AMD GPU Setup Test ==="
echo "Testing AMD GPU configuration for multi-GPU training"

# Test basic environment
echo "1. Testing basic environment..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not available')"

# Test ROCm availability
echo ""
echo "2. Testing ROCm availability..."
if command -v rocm-smi > /dev/null 2>&1; then
    echo "✓ rocm-smi found"
    echo "GPU Information:"
    rocm-smi --showid --showmeminfo vram --showtemp
else
    echo "✗ rocm-smi not found - ROCm may not be installed"
fi

# Test HIP environment
echo ""
echo "3. Testing HIP environment..."
if command -v hipconfig > /dev/null 2>&1; then
    echo "✓ hipconfig found"
    hipconfig --platform
else
    echo "✗ hipconfig not found"
fi

# Test PyTorch ROCm support
echo ""
echo "4. Testing PyTorch ROCm support..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
else:
    print('No CUDA devices found')
"

# Test environment variables
echo ""
echo "5. Testing environment variables..."
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-'not set'}"
echo "MIOPEN_DISABLE_CACHE: ${MIOPEN_DISABLE_CACHE:-'not set'}"
echo "MIOPEN_DEBUG_DISABLE_FIND_DB: ${MIOPEN_DEBUG_DISABLE_FIND_DB:-'not set'}"

# Test multi-GPU import
echo ""
echo "6. Testing multi-GPU imports..."
python -c "
try:
    import torch
    import torch.nn as nn
    print('✓ PyTorch imports successful')

    # Test DataParallel
    if hasattr(nn, 'DataParallel'):
        print('✓ DataParallel available')
    else:
        print('✗ DataParallel not available')

    # Test distributed training
    if hasattr(torch.distributed, 'is_available'):
        print('✓ Distributed training available:', torch.distributed.is_available())
    else:
        print('✗ Distributed training not available')

except ImportError as e:
    print('✗ Import error:', e)
except Exception as e:
    print('✗ Error:', e)
"

# Test data loading
echo ""
echo "7. Testing data loading..."
if [ -f "../dataset/Solar/solar_AL.txt" ]; then
    echo "✓ Solar data file found"
    echo "File size: $(ls -lh ../dataset/Solar/solar_AL.txt | awk '{print $5}')"
else
    echo "✗ Solar data file not found at ../dataset/Solar/solar_AL.txt"
fi

# Test script execution
echo ""
echo "8. Testing training script execution..."
echo "Running a quick test with minimal configuration..."

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HIP_VISIBLE_DEVICES="0"

python -u ../run_longExp.py \
  --random_seed 2023 \
  --is_training 1 \
  --root_path ../dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_test_96_96 \
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
  --des 'Test' \
  --train_epochs 1 \
  --devices 0 \
  --use_gpu True \
  --gpu 0 \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 1 2>&1 || echo "Test execution completed (errors are expected for this minimal test)"

echo ""
echo "=== Test completed ==="
echo "If the test shows GPU information and PyTorch can access AMD GPUs, the setup is working."
echo "Check the logs above for any error messages."