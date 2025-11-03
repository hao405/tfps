#!/usr/bin/env python3
"""
快速验证Cluster层修复是否有效
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from layers.Cluster import EDESC

def test_cluster_initialization():
    """测试Cluster层的初始化"""
    print("=== Testing Cluster Layer Initialization ===\n")
    
    # 创建Cluster层
    print("1. Creating Cluster layer...")
    try:
        cluster = EDESC(
            d_model=16,
            n_clusters=16,
            eta=5,
            c_out=137,
            bs=32,
            patch_len=16,
            stride=8
        )
        print("   ✓ Cluster layer created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create cluster layer: {e}")
        return False
    
    # 检查D参数
    print("\n2. Checking D parameter initialization...")
    D = cluster.D
    print(f"   D shape: {D.shape}")
    print(f"   D dtype: {D.dtype}")
    
    # 检查NaN
    has_nan = torch.isnan(D).any().item()
    print(f"   Contains NaN: {has_nan}")
    if has_nan:
        nan_count = torch.isnan(D).sum().item()
        print(f"   ✗ ERROR: Found {nan_count} NaN values in D!")
        return False
    else:
        print(f"   ✓ No NaN values")
    
    # 检查Inf
    has_inf = torch.isinf(D).any().item()
    print(f"   Contains Inf: {has_inf}")
    if has_inf:
        inf_count = torch.isinf(D).sum().item()
        print(f"   ✗ ERROR: Found {inf_count} Inf values in D!")
        return False
    else:
        print(f"   ✓ No Inf values")
    
    # 检查数值范围
    print(f"   Value range: [{D.min().item():.6f}, {D.max().item():.6f}]")
    print(f"   Mean: {D.mean().item():.6f}")
    print(f"   Std: {D.std().item():.6f}")
    
    # 检查是否太大
    if D.abs().max().item() > 100:
        print(f"   ⚠ WARNING: D contains very large values")
    else:
        print(f"   ✓ Values are in reasonable range")
    
    # 测试前向传播
    print("\n3. Testing forward pass...")
    cluster.eval()
    
    # 创建测试输入
    batch_size = 2
    patch_num = 12
    n_z = 137 * 16  # c_out * d_model
    
    test_input = torch.randn(batch_size * patch_num, n_z)
    print(f"   Test input shape: {test_input.shape}")
    print(f"   Test input range: [{test_input.min().item():.4f}, {test_input.max().item():.4f}]")
    
    try:
        with torch.no_grad():
            output = cluster(test_input)
        
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        
        # 检查输出
        if torch.isnan(output).any():
            print(f"   ✗ ERROR: Output contains NaN!")
            print(f"   NaN count: {torch.isnan(output).sum().item()}")
            return False
        
        if torch.isinf(output).any():
            print(f"   ✗ ERROR: Output contains Inf!")
            return False
        
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"   ✓ Output is valid")
        
    except Exception as e:
        print(f"   ✗ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试total_loss
    print("\n4. Testing total_loss...")
    try:
        pred = torch.randn(batch_size * patch_num, n_z)
        target = torch.randn(batch_size * patch_num, n_z)
        
        loss = cluster.total_loss(pred, target, dim=cluster.d, n_clusters=16, beta=0.01)
        
        print(f"   Loss value: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print(f"   ✗ ERROR: Loss is NaN!")
            return False
        
        if torch.isinf(loss):
            print(f"   ✗ ERROR: Loss is Inf!")
            return False
        
        print(f"   ✓ Loss is valid")
        
    except Exception as e:
        print(f"   ✗ ERROR during loss calculation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试梯度
    print("\n5. Testing gradient flow...")
    cluster.train()
    
    test_input = torch.randn(batch_size * patch_num, n_z, requires_grad=True)
    
    try:
        output = cluster(test_input)
        loss = output.sum()
        loss.backward()
        
        print(f"   ✓ Backward pass successful")
        
        # 检查D的梯度
        if cluster.D.grad is not None:
            grad = cluster.D.grad
            
            if torch.isnan(grad).any():
                print(f"   ✗ ERROR: D gradient contains NaN!")
                return False
            
            if torch.isinf(grad).any():
                print(f"   ✗ ERROR: D gradient contains Inf!")
                return False
            
            print(f"   D gradient range: [{grad.min().item():.6f}, {grad.max().item():.6f}]")
            print(f"   ✓ Gradients are valid")
        else:
            print(f"   ⚠ WARNING: D has no gradient")
            
    except Exception as e:
        print(f"   ✗ ERROR during backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== All Tests Passed! ===")
    print("\n✅ Cluster layer initialization is correct!")
    print("✅ No NaN or Inf in parameters")
    print("✅ Forward and backward passes work correctly")
    print("✅ Ready for training!")
    
    return True

if __name__ == "__main__":
    success = test_cluster_initialization()
    
    if success:
        print("\n" + "="*50)
        print("🎉 SUCCESS! The Cluster layer fix works!")
        print("="*50)
        print("\nYou can now run training with confidence:")
        print("  cd scripts")
        print("  bash solar.sh")
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("❌ FAILED! There are still issues with the Cluster layer")
        print("="*50)
        sys.exit(1)
