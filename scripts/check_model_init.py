#!/usr/bin/env python3
"""
检查模型初始化,查找NaN的源头
"""

import torch
import torch.nn as nn
import sys
import os
import argparse
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PatchTST_MoE_cluster

def check_model_initialization(args):
    """检查模型初始化是否正常"""
    print("=== Model Initialization Check ===\n")
    
    # 创建模型
    print("1. Creating model...")
    try:
        model = PatchTST_MoE_cluster.Model(args).float()
        print(f"   Model created successfully")
        print(f"   Model type: {type(model)}")
    except Exception as e:
        print(f"   ERROR: Failed to create model: {e}")
        return False
    
    # 检查模型参数
    print("\n2. Checking model parameters...")
    nan_params = []
    inf_params = []
    zero_params = []
    large_params = []
    
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if torch.isnan(param).any():
            nan_params.append((name, torch.isnan(param).sum().item()))
        if torch.isinf(param).any():
            inf_params.append((name, torch.isinf(param).sum().item()))
        if (param.abs() < 1e-8).all():
            zero_params.append(name)
        if (param.abs() > 1e3).any():
            large_params.append((name, param.abs().max().item()))
    
    print(f"   Total parameters: {total_params:,}")
    
    if nan_params:
        print(f"   ERROR: Found {len(nan_params)} parameters with NaN values:")
        for name, count in nan_params[:5]:
            print(f"     - {name}: {count} NaN values")
    else:
        print(f"   ✓ No NaN in parameters")
    
    if inf_params:
        print(f"   ERROR: Found {len(inf_params)} parameters with Inf values:")
        for name, count in inf_params[:5]:
            print(f"     - {name}: {count} Inf values")
    else:
        print(f"   ✓ No Inf in parameters")
    
    if zero_params:
        print(f"   WARNING: Found {len(zero_params)} parameters that are all zeros:")
        for name in zero_params[:5]:
            print(f"     - {name}")
    
    if large_params:
        print(f"   WARNING: Found {len(large_params)} parameters with very large values (>1e3):")
        for name, max_val in large_params[:5]:
            print(f"     - {name}: max={max_val:.2e}")
    
    # 测试前向传播
    print("\n3. Testing forward pass...")
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    seq_len = args.seq_len
    enc_in = args.enc_in
    
    print(f"   Input shape: [{batch_size}, {seq_len}, {enc_in}]")
    
    # 使用正态分布的数据
    test_input = torch.randn(batch_size, seq_len, enc_in)
    print(f"   Test input range: [{test_input.min().item():.4f}, {test_input.max().item():.4f}]")
    
    try:
        with torch.no_grad():
            s_time, s_frequency, outputs = model(test_input)
        
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        
        if torch.isnan(outputs).any():
            print(f"   ERROR: Output contains NaN!")
            print(f"   NaN count: {torch.isnan(outputs).sum().item()}")
            
            # 检查中间输出
            print(f"   s_time range: [{s_time.min().item():.4f}, {s_time.max().item():.4f}]" if not torch.isnan(s_time).all() else "   s_time: all NaN")
            print(f"   s_frequency range: [{s_frequency.min().item():.4f}, {s_frequency.max().item():.4f}]" if not torch.isnan(s_frequency).all() else "   s_frequency: all NaN")
            return False
        
        if torch.isinf(outputs).any():
            print(f"   ERROR: Output contains Inf!")
            return False
            
        print(f"   ✓ Output is valid")
        
    except Exception as e:
        print(f"   ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试不同大小的输入
    print("\n4. Testing with different input scales...")
    test_scales = [0.1, 1.0, 10.0]
    
    for scale in test_scales:
        test_input_scaled = torch.randn(batch_size, seq_len, enc_in) * scale
        try:
            with torch.no_grad():
                _, _, outputs = model(test_input_scaled)
            
            has_nan = torch.isnan(outputs).any().item()
            has_inf = torch.isinf(outputs).any().item()
            
            status = "✓" if not (has_nan or has_inf) else "✗"
            print(f"   {status} Scale {scale}: input range [{test_input_scaled.min().item():.4f}, {test_input_scaled.max().item():.4f}]", end="")
            
            if not (has_nan or has_inf):
                print(f" -> output range [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            else:
                print(f" -> {'NaN' if has_nan else ''} {'Inf' if has_inf else ''}")
                
        except Exception as e:
            print(f"   ✗ Scale {scale}: Error - {e}")
    
    # 检查梯度流
    print("\n5. Testing gradient flow...")
    model.train()
    test_input = torch.randn(batch_size, seq_len, enc_in, requires_grad=True)
    
    try:
        s_time, s_frequency, outputs = model(test_input)
        loss = outputs.sum()
        loss.backward()
        
        print(f"   ✓ Backward pass successful")
        
        # 检查梯度
        nan_grads = []
        inf_grads = []
        zero_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                if (param.grad.abs() < 1e-10).all():
                    zero_grads.append(name)
        
        if nan_grads:
            print(f"   ERROR: {len(nan_grads)} parameters have NaN gradients:")
            for name in nan_grads[:5]:
                print(f"     - {name}")
        else:
            print(f"   ✓ No NaN gradients")
        
        if inf_grads:
            print(f"   ERROR: {len(inf_grads)} parameters have Inf gradients:")
            for name in inf_grads[:5]:
                print(f"     - {name}")
        else:
            print(f"   ✓ No Inf gradients")
            
    except Exception as e:
        print(f"   ERROR during backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Check Complete ===")
    
    return len(nan_params) == 0 and len(inf_params) == 0

if __name__ == "__main__":
    # 模拟参数
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=137)
    parser.add_argument('--c_out', type=int, default=137)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fc_dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--T_num_expert', type=int, default=16)
    parser.add_argument('--T_top_k', type=int, default=1)
    parser.add_argument('--F_num_expert', type=int, default=16)
    parser.add_argument('--F_top_k', type=int, default=1)
    parser.add_argument('--eta', type=int, default=5)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # 其他参数
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--individual', type=int, default=1)
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--padding_patch', type=str, default='end')
    
    args = parser.parse_args()
    
    print("Testing PatchTST_MoE_cluster model initialization with Solar dataset parameters\n")
    
    success = check_model_initialization(args)
    
    if success:
        print("\n✓ Model initialization looks good!")
    else:
        print("\n✗ Model has initialization problems - this will cause NaN loss")
        print("\nRecommendations:")
        print("1. Check custom initialization in model code")
        print("2. Reduce model complexity (smaller d_model, fewer experts)")
        print("3. Add proper weight initialization")
        print("4. Check for division by zero in model architecture")
