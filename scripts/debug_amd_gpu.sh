#!/bin/bash

# AMD GPU检测和调试脚本
# 用于诊断AMD集群上的GPU检测问题

echo "========================================="
echo "AMD GPU检测和调试脚本"
echo "========================================="

# 系统信息
echo "=== 系统信息 ==="
echo "主机名: $(hostname)"
echo "操作系统: $(uname -a)"
echo "当前目录: $(pwd)"
echo "用户: $USER"
echo ""

# 环境变量检查
echo "=== 环境变量检查 ==="
echo "ROCM_PATH: ${ROCM_PATH:-'未设置'}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-'未设置'}"
echo "MIOPEN_DISABLE_CACHE: ${MIOPEN_DISABLE_CACHE:-'未设置'}"
echo "MIOPEN_SYSTEM_DB_PATH: ${MIOPEN_SYSTEM_DB_PATH:-'未设置'}"
echo "HSA_USERPTR_FOR_PAGED_MEM: ${HSA_USERPTR_FOR_PAGED_MEM:-'未设置'}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-'未设置'}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'未设置'}"
echo ""

# ROCm工具检查
echo "=== ROCm工具检查 ==="

# 检查rocm-smi
if command -v rocm-smi &> /dev/null; then
    echo "✓ rocm-smi 找到: $(which rocm-smi)"
    echo "rocm-smi版本信息:"
    rocm-smi --version 2>/dev/null || echo "无法获取版本信息"

    echo ""
    echo "AMD GPU设备列表:"
    rocm-smi --showid 2>/dev/null || echo "无法获取设备列表"

    echo ""
    echo "GPU详细信息:"
    rocm-smi --showproductname 2>/dev/null || echo "无法获取产品信息"

else
    echo "✗ rocm-smi 未找到"
fi

echo ""

# 检查hip-smi
if command -v hip-smi &> /dev/null; then
    echo "✓ hip-smi 找到: $(which hip-smi)"
    echo "hip-smi输出:"
    hip-smi 2>/dev/null || echo "hip-smi执行失败"
else
    echo "✗ hip-smi 未找到"
fi

echo ""

# 检查其他ROCm工具
for tool in rocminfo hipconfig clinfo; do
    if command -v $tool &> /dev/null; then
        echo "✓ $tool 找到: $(which $tool)"
        if [ "$tool" = "rocminfo" ]; then
            echo "$tool设备摘要:"
            rocminfo 2>/dev/null | grep -E "(Device Type|Name)" | head -10 || echo "无法获取设备信息"
        fi
    else
        echo "✗ $tool 未找到"
    fi
done

echo ""

# 设备文件检查
echo "=== 设备文件检查 ==="
if [ -d "/dev/kfd" ]; then
    echo "✓ /dev/kfd 存在 (AMD GPU核心驱动)"
else
    echo "✗ /dev/kfd 不存在"
fi

if [ -d "/dev/dri" ]; then
    echo "✓ /dev/dri 存在 (DRM设备)"
    echo "DRM设备列表:"
    ls -la /dev/dri/ 2>/dev/null | grep -E "(card|render)" || echo "无DRM设备"
else
    echo "✗ /dev/dri 不存在"
fi

echo ""

# Python环境检查
echo "=== Python环境检查 ==="
echo "Python版本: $(python --version 2>&1)"
echo "Python路径: $(which python)"

# 检查PyTorch
echo ""
echo "PyTorch检查:"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'CUDA设备数: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  设备{i}: {torch.cuda.get_device_name(i)}')
print(f'ROCm可用: {torch.version.hip is not None}')
if torch.version.hip:
    print(f'ROCm版本: {torch.version.hip}')
" 2>/dev/null || echo "PyTorch检查失败"

echo ""

# 实际GPU数量检测
echo "=== GPU数量检测 ==="
GPU_COUNT=0

# 方法1: rocm-smi
if command -v rocm-smi &> /dev/null; then
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "0")
    echo "通过rocm-smi检测到: $GPU_COUNT 个GPU"
fi

# 方法2: hip-smi
if [ $GPU_COUNT -eq 0 ] && command -v hip-smi &> /dev/null; then
    GPU_COUNT=$(hip-smi 2>/dev/null | grep -c "GPU" || echo "0")
    echo "通过hip-smi检测到: $GPU_COUNT 个GPU"
fi

# 方法3: 环境变量
if [ $GPU_COUNT -eq 0 ] && [ -n "$HIP_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)
    echo "通过HIP_VISIBLE_DEVICES检测到: $GPU_COUNT 个GPU"
fi

# 方法4: PyTorch
if [ $GPU_COUNT -eq 0 ]; then
    PYTHON_GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$PYTHON_GPU_COUNT" -gt 0 ]; then
        GPU_COUNT=$PYTHON_GPU_COUNT
        echo "通过PyTorch检测到: $GPU_COUNT 个GPU"
    fi
fi

echo "最终GPU数量: $GPU_COUNT"

echo ""

# 推荐配置
echo "=== 推荐配置 ==="
if [ $GPU_COUNT -ge 8 ]; then
    echo "✓ 检测到8个或更多GPU，建议使用全卡配置:"
    echo "  --devices 0,1,2,3,4,5,6,7 --use_multi_gpu --use_gpu False"
elif [ $GPU_COUNT -ge 4 ]; then
    echo "✓ 检测到4-7个GPU，建议使用:"
    echo "  --devices 0,1,2,3 --use_multi_gpu --use_gpu False"
elif [ $GPU_COUNT -ge 2 ]; then
    echo "✓ 检测到2-3个GPU，建议使用:"
    echo "  --devices 0,1 --use_multi_gpu --use_gpu False"
elif [ $GPU_COUNT -eq 1 ]; then
    echo "✓ 检测到1个GPU，建议使用:"
    echo "  --gpu 0 --use_gpu False"
else
    echo "⚠ 未检测到GPU，将使用CPU模式:"
    echo "  --gpu 0 --use_gpu False"
fi

echo ""
echo "=== 调试完成 ==="
echo "请根据以上信息配置您的训练脚本"

# 生成推荐的设备配置
echo ""
echo "生成的设备配置:"
if [ $GPU_COUNT -gt 0 ]; then
    GPU_LIST=""
    for ((i=0; i<GPU_COUNT; i++)); do
        if [ $i -eq 0 ]; then
            GPU_LIST="$i"
        else
            GPU_LIST="$GPU_LIST,$i"
        fi
    done
    echo "export HIP_VISIBLE_DEVICES=\"$GPU_LIST\""
    echo "export MIOPEN_DISABLE_CACHE=1"
    echo "export MIOPEN_SYSTEM_DB_PATH=\"\""
fi

echo ""
echo "========================================="
echo "调试脚本执行完成！"
echo "========================================="

# 可选：保存调试信息到文件
DEBUG_FILE="amd_gpu_debug_$(date +%Y%m%d_%H%M%S).log"
echo "保存调试信息到: $DEBUG_FILE"
exec >"$DEBUG_FILE" 2>>1
set -x

# 重新运行部分检查以保存到文件
echo "=== 保存的调试信息 ===" > "$DEBUG_FILE"
rocm-smi --showid 2>/dev/null >> "$DEBUG_FILE" || true
hip-smi 2>/dev/null >> "$DEBUG_FILE" || true
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, ROCm: {torch.version.hip is not None}')" 2>/dev/null >> "$DEBUG_FILE" || true

echo "调试信息已保存到 $DEBUG_FILE" 2>>1