# AMD Multi-GPU Training Setup for PatchTST MoE

This guide explains how to set up and use AMD GPUs for multi-GPU training with the PatchTST MoE cluster model.

## Problem Description

The original codebase was designed primarily for NVIDIA GPUs and uses CUDA-specific configurations. When running on AMD GPUs, the following issues occur:

1. **Device Detection**: Uses `CUDA_VISIBLE_DEVICES` which is NVIDIA-specific
2. **GPU Detection**: Relies on `torch.cuda.is_available()` which may not work properly with AMD GPUs
3. **Multi-GPU Setup**: Missing AMD-specific initialization and environment variables

## Solution Overview

The solution involves:

1. **Updated device detection** to handle AMD ROCm/HIP GPUs
2. **Proper environment variables** for AMD GPUs
3. **Enhanced multi-GPU setup** for AMD hardware
4. **Comprehensive testing** and validation scripts

## Quick Start

### 1. Test Your AMD GPU Setup

```bash
python scripts/test_amd_setup.py
```

This will verify:
- PyTorch ROCm support
- GPU detection
- Multi-GPU capabilities
- Required imports

### 2. Run Training with AMD Multi-GPU

```bash
python scripts/train_amd_multi_gpu.py
```

Or manually with specific parameters:

```bash
python run_longExp.py \
  --use_gpu True \
  --use_multi_gpu \
  --devices 0,1,2,3,4,5,6,7 \
  --model PatchTST_MoE_cluster \
  --data solar \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0005
```

## Detailed Setup

### Environment Variables

Set these environment variables for optimal AMD GPU performance:

```bash
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # Use all available GPUs
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_SYSTEM_DB_PATH=""
export HSA_USERPTR_FOR_PAGED_MEM=0
export ROCM_PATH=/opt/rocm
```

### Code Changes

#### 1. Device Detection (`exp/exp_basic.py`)

The device acquisition now properly detects AMD GPUs:

```python
def _acquire_device(self):
    if self.args.use_gpu:
        # Check for AMD ROCm/HIP support
        if torch.version.hip is not None:
            # AMD GPU detected via ROCm
            print('AMD ROCm GPU detected')
            if self.args.use_multi_gpu:
                # For multi-GPU, set HIP_VISIBLE_DEVICES
                os.environ["HIP_VISIBLE_DEVICES"] = self.args.devices
                print(f'Use AMD Multi-GPU: {self.args.devices}')
                device = torch.device('cuda:0')  # AMD uses cuda:0 interface through ROCm
            else:
                # Single AMD GPU
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use AMD GPU: cuda:{}'.format(self.args.gpu))
        elif torch.cuda.is_available():
            # NVIDIA GPU detected
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use NVIDIA GPU: cuda:{}'.format(self.args.gpu))
        else:
            # No GPU detected, fallback to CPU
            device = torch.device('cpu')
            print('No GPU detected, using CPU')
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device
```

#### 2. Multi-GPU Model Setup (`exp/exp_main.py`)

Enhanced multi-GPU setup for AMD hardware:

```python
if self.args.use_multi_gpu and self.args.use_gpu:
    if torch.version.hip is not None:
        # AMD multi-GPU setup
        print('Setting up AMD Multi-GPU with DataParallel')
        model = nn.DataParallel(model)  # AMD uses automatic device detection
    else:
        # NVIDIA multi-GPU setup
        print('Setting up NVIDIA Multi-GPU with DataParallel')
        model = nn.DataParallel(model, device_ids=self.args.device_ids)
```

#### 3. GPU Detection (`run_longExp.py`)

Enhanced GPU detection in the main script:

```python
# Enhanced GPU detection for both AMD and NVIDIA GPUs
if args.use_gpu:
    if torch.version.hip is not None:
        # AMD ROCm GPU detected
        print("AMD ROCm GPU detected via PyTorch")
        args.use_gpu = True
    elif torch.cuda.is_available():
        # NVIDIA CUDA GPU detected
        print("NVIDIA CUDA GPU detected via PyTorch")
        args.use_gpu = True
    else:
        # No GPU detected
        print("No GPU detected, falling back to CPU")
        args.use_gpu = False
```

## Available Scripts

### 1. `scripts/test_amd_setup.py`
Comprehensive test script that verifies:
- PyTorch ROCm support
- GPU detection and count
- Tensor operations on each GPU
- Multi-GPU setup with DataParallel
- Required module imports

### 2. `scripts/train_amd_multi_gpu.py`
Complete training script with:
- Automatic AMD environment setup
- GPU detection and validation
- Proper multi-GPU configuration
- Training loop with logging

### 3. `scripts/debug_amd_gpu.sh`
Bash script for debugging AMD GPU setup:
- System information
- Environment variable checks
- ROCm tool availability
- GPU device detection
- Recommendations

### 4. `scripts/solar_amd_fixed.sh`
Fixed bash script for solar dataset training on AMD GPUs.

### 5. `scripts/solar_amd_cluster.sh`
Advanced AMD cluster configuration with auto-detection.

## Troubleshooting

### Issue 1: "Use CPU" message appears

**Problem**: GPU not detected, falling back to CPU

**Solutions**:
1. Run `python scripts/test_amd_setup.py` to diagnose
2. Check PyTorch ROCm installation: `python -c "import torch; print(torch.version.hip)"`
3. Verify AMD drivers: `rocm-smi` or `hip-smi`
4. Set environment variables manually

### Issue 2: Multi-GPU not working

**Problem**: Only one GPU is used despite multiple being available

**Solutions**:
1. Check `HIP_VISIBLE_DEVICES` environment variable
2. Verify `--use_multi_gpu` flag is set
3. Ensure `--devices` parameter lists all GPU IDs
4. Check PyTorch DataParallel setup

### Issue 3: Training is slow

**Problem**: Performance issues on AMD GPUs

**Solutions**:
1. Set `MIOPEN_DISABLE_CACHE=1` to avoid cache issues
2. Use appropriate batch size for your GPUs
3. Check memory usage with `rocm-smi`
4. Consider using mixed precision training (`--use_amp`)

### Issue 4: Import errors

**Problem**: Cannot import required modules

**Solutions**:
1. Install required packages: `pip install -r requirements.txt`
2. Check PyTorch ROCm version compatibility
3. Verify all dependencies are installed

## Performance Tips

1. **Batch Size**: Start with 32 and adjust based on your GPU memory
2. **Learning Rate**: Use 0.0005 as a starting point
3. **Multi-GPU Scaling**: Performance scales well with multiple AMD GPUs
4. **Memory Management**: Monitor GPU memory usage with `rocm-smi`
5. **Environment Variables**: Always set AMD-specific environment variables

## System Requirements

- AMD GPUs with ROCm support
- PyTorch with ROCm support
- ROCm drivers and tools
- Python 3.6+
- Linux operating system (ROCm requirement)

## Verification

After setup, verify everything works:

```bash
# Test GPU detection
python scripts/test_amd_setup.py

# Check environment
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run a quick training test
python run_longExp.py --use_gpu True --use_multi_gpu --devices 0,1 --train_epochs 1
```

## Support

For issues specific to:
- **AMD GPUs**: Check ROCm documentation and forums
- **PyTorch ROCm**: Verify PyTorch-ROCm compatibility
- **This codebase**: Review the troubleshooting section above

## Changelog

- **v1.0**: Initial AMD multi-GPU support implementation
- Enhanced device detection for AMD GPUs
- Added proper environment variable handling
- Created comprehensive test and training scripts
- Added detailed documentation and troubleshooting guide