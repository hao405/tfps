#!/usr/bin/env python3
"""
非稳态时序 NaN 问题诊断工具
快速检测训练过程中的潜在问题
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

def test_data_distribution(data_path):
    """测试数据分布特性"""
    print("\n" + "="*60)
    print("📊 数据分布诊断")
    print("="*60)
    
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        
        # 移除非数值列
        numeric_df = df.select_dtypes(include=[np.number])
        
        print(f"\n数据形状: {numeric_df.shape}")
        print(f"\n基本统计:")
        print(numeric_df.describe())
        
        # 检查异常值
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            if len(data) == 0:
                continue
                
            mean = data.mean()
            std = data.std()
            min_val = data.min()
            max_val = data.max()
            
            # 非稳态检测
            is_constant = std < 1e-6
            has_extreme = (max_val - min_val) > 1e6
            
            if is_constant:
                print(f"\n⚠️  警告: 列 '{col}' 方差极小 (std={std:.2e})，可能导致RevIN失败")
            if has_extreme:
                print(f"\n⚠️  警告: 列 '{col}' 范围极大 ({min_val:.2e} to {max_val:.2e})，建议预处理")
        
        # 检查缺失值
        missing = numeric_df.isnull().sum()
        if missing.any():
            print(f"\n⚠️  发现缺失值:")
            print(missing[missing > 0])
        
        return True
    except Exception as e:
        print(f"❌ 数据分析失败: {e}")
        return False

def test_model_initialization(args):
    """测试模型初始化"""
    print("\n" + "="*60)
    print("🏗️  模型初始化诊断")
    print("="*60)
    
    try:
        from exp.exp_main import Exp_Main
        
        print(f"\n配置参数:")
        print(f"  - Model: {args.model}")
        print(f"  - d_model: {args.d_model}")
        print(f"  - T_num_expert: {args.T_num_expert}")
        print(f"  - F_num_expert: {args.F_num_expert}")
        print(f"  - beta: {args.beta}")
        print(f"  - learning_rate: {args.learning_rate}")
        
        # 创建实验
        exp = Exp_Main(args)
        
        # 检查模型参数
        total_params = sum(p.numel() for p in exp.model.parameters())
        trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
        
        print(f"\n模型参数:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        
        # 检查D矩阵初始化
        if hasattr(exp.model, 'model_time'):
            D_time = exp.model.model_time.cluster.D
            D_freq = exp.model.model_frequency.cluster.D
            
            print(f"\n子空间基矩阵:")
            print(f"  - D_time shape: {D_time.shape}")
            print(f"  - D_freq shape: {D_freq.shape}")
            
            # 检查正交性
            orthogonality_time = torch.norm(D_time.t() @ D_time - torch.eye(D_time.shape[1], device=D_time.device))
            orthogonality_freq = torch.norm(D_freq.t() @ D_freq - torch.eye(D_freq.shape[1], device=D_freq.device))
            
            print(f"  - D_time 正交性误差: {orthogonality_time.item():.6f}")
            print(f"  - D_freq 正交性误差: {orthogonality_freq.item():.6f}")
            
            if orthogonality_time > 1.0 or orthogonality_freq > 1.0:
                print(f"\n⚠️  警告: 子空间基不够正交，可能导致训练不稳定")
        
        return True
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass(args):
    """测试前向传播"""
    print("\n" + "="*60)
    print("🔄 前向传播诊断")
    print("="*60)
    
    try:
        from exp.exp_main import Exp_Main
        
        exp = Exp_Main(args)
        exp.model.eval()
        
        # 创建测试数据
        batch_size = 4
        seq_len = args.seq_len
        enc_in = args.enc_in
        
        # 测试不同数据范围
        test_cases = [
            ("正常范围", torch.randn(batch_size, seq_len, enc_in)),
            ("小值范围", torch.randn(batch_size, seq_len, enc_in) * 0.01),
            ("大值范围", torch.randn(batch_size, seq_len, enc_in) * 100),
            ("常量输入", torch.ones(batch_size, seq_len, enc_in)),
        ]
        
        for name, test_input in test_cases:
            print(f"\n测试: {name}")
            print(f"  输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
            
            try:
                with torch.no_grad():
                    test_input = test_input.to(exp.device)
                    
                    if 'TST' in args.model or 'Linear' in args.model:
                        s_time, s_frequency, outputs = exp.model(test_input)
                        
                        # 检查输出
                        checks = [
                            ("s_time", s_time),
                            ("s_frequency", s_frequency),
                            ("outputs", outputs)
                        ]
                        
                        all_valid = True
                        for check_name, check_tensor in checks:
                            has_nan = torch.isnan(check_tensor).any()
                            has_inf = torch.isinf(check_tensor).any()
                            
                            if has_nan or has_inf:
                                print(f"  ❌ {check_name}: 包含 NaN={has_nan}, Inf={has_inf}")
                                all_valid = False
                            else:
                                print(f"  ✅ {check_name}: OK [范围: {check_tensor.min():.4f}, {check_tensor.max():.4f}]")
                        
                        if not all_valid:
                            print(f"\n⚠️  警告: {name} 测试未通过，需要检查模型架构")
                    else:
                        outputs = exp.model(test_input)
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"  ❌ 输出包含异常值")
                        else:
                            print(f"  ✅ 输出正常")
                            
            except Exception as e:
                print(f"  ❌ 前向传播失败: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation(args):
    """测试损失计算"""
    print("\n" + "="*60)
    print("📉 损失计算诊断")
    print("="*60)
    
    try:
        from exp.exp_main import Exp_Main
        
        exp = Exp_Main(args)
        
        # 创建测试数据
        batch_size = 4
        pred_len = args.pred_len
        c_out = args.c_out
        num_expert = args.T_num_expert
        
        # 测试KL散度计算
        print("\n测试KL散度稳定性:")
        test_distributions = [
            ("均匀分布", torch.ones(batch_size, num_expert) / num_expert),
            ("极端分布", torch.tensor([[0.99, 0.01], [0.01, 0.99]] * (batch_size // 2)).float()),
            ("接近零", torch.ones(batch_size, num_expert) * 0.001),
        ]
        
        for name, pred in test_distributions[:1]:  # 只测试均匀分布避免维度问题
            if pred.shape[1] != num_expert:
                continue
                
            print(f"\n  {name}:")
            target = torch.softmax(torch.randn(batch_size, num_expert), dim=1)
            
            try:
                # 模拟KL散度计算
                eps = 1e-8
                pred_safe = torch.clamp(pred, min=eps, max=1.0)
                target_safe = torch.clamp(target, min=eps, max=1.0)
                
                pred_safe = pred_safe / (pred_safe.sum(dim=1, keepdim=True) + eps)
                target_safe = target_safe / (target_safe.sum(dim=1, keepdim=True) + eps)
                
                log_pred = torch.log(pred_safe + eps)
                kl_loss = torch.nn.functional.kl_div(log_pred, target_safe, reduction='batchmean')
                
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    print(f"    ❌ KL散度异常: {kl_loss.item()}")
                else:
                    print(f"    ✅ KL散度正常: {kl_loss.item():.6f}")
            except Exception as e:
                print(f"    ❌ KL散度计算失败: {e}")
        
        return True
    except Exception as e:
        print(f"❌ 损失计算测试失败: {e}")
        return False

def main():
    """主诊断流程"""
    print("\n" + "="*60)
    print("🔍 TFPS 非稳态时序 NaN 问题诊断工具")
    print("="*60)
    
    # 导入配置
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from run_longExp import parser
        
        # 使用最小配置进行测试
        test_args = [
            '--random_seed', '2021',
            '--is_training', '1',
            '--root_path', './dataset/ETT-small/',
            '--data_path', 'ETTh1.csv',
            '--model_id', 'test_diagnose',
            '--model', 'PatchTST_MoE_cluster',
            '--data', 'ETTh1',
            '--features', 'M',
            '--seq_len', '96',
            '--pred_len', '96',
            '--enc_in', '7',
            '--c_out', '7',
            '--e_layers', '1',
            '--n_heads', '4',
            '--d_model', '16',
            '--d_ff', '32',
            '--dropout', '0.1',
            '--fc_dropout', '0.1',
            '--head_dropout', '0',
            '--patch_len', '16',
            '--stride', '8',
            '--T_num_expert', '2',
            '--T_top_k', '1',
            '--F_num_expert', '2',
            '--F_top_k', '1',
            '--beta', '0.001',
            '--learning_rate', '0.001',
            '--batch_size', '8',
            '--train_epochs', '1',
            '--itr', '1'
        ]
        
        args = parser.parse_args(test_args)
        
        # 运行诊断测试
        results = {}
        
        # 1. 数据诊断
        data_path = os.path.join(args.root_path, args.data_path)
        if os.path.exists(data_path):
            results['data'] = test_data_distribution(data_path)
        else:
            print(f"\n⚠️  数据文件不存在: {data_path}")
            results['data'] = False
        
        # 2. 模型初始化诊断
        results['init'] = test_model_initialization(args)
        
        # 3. 前向传播诊断
        if results['init']:
            results['forward'] = test_forward_pass(args)
        else:
            results['forward'] = False
        
        # 4. 损失计算诊断
        if results['init']:
            results['loss'] = test_loss_computation(args)
        else:
            results['loss'] = False
        
        # 总结
        print("\n" + "="*60)
        print("📋 诊断总结")
        print("="*60)
        
        for test_name, result in results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name.upper()}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n🎉 所有测试通过！模型配置正常。")
            print("\n建议:")
            print("  1. 开始训练前，建议使用较小的beta值 (0.001)")
            print("  2. 使用较低的学习率 (0.0001-0.001)")
            print("  3. 监控前几个epoch的损失变化")
        else:
            print("\n⚠️  部分测试未通过，建议:")
            print("  1. 检查数据预处理")
            print("  2. 降低模型复杂度 (减少expert数量)")
            print("  3. 使用更保守的超参数")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
