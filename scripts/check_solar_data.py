#!/usr/bin/env python3
"""
检查Solar数据集的质量,发现可能导致NaN loss的问题
"""

import numpy as np
import pandas as pd
import os

def check_solar_data(root_path, data_path):
    """检查Solar数据集的质量"""
    print("=== Solar Dataset Quality Check ===\n")
    
    # 读取数据
    print(f"Reading data from: {os.path.join(root_path, data_path)}")
    df_raw = []
    try:
        with open(os.path.join(root_path, data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    print(f"Data shape: {df_raw.shape}")
    print(f"Number of samples: {df_raw.shape[0]}")
    print(f"Number of features: {df_raw.shape[1]}")
    
    # 检查NaN值
    nan_count = np.isnan(df_raw).sum()
    print(f"\n1. NaN values: {nan_count}")
    if nan_count > 0:
        print(f"   WARNING: Found {nan_count} NaN values!")
        nan_cols = np.where(np.isnan(df_raw).any(axis=0))[0]
        print(f"   Columns with NaN: {nan_cols}")
    
    # 检查Inf值
    inf_count = np.isinf(df_raw).sum()
    print(f"\n2. Inf values: {inf_count}")
    if inf_count > 0:
        print(f"   WARNING: Found {inf_count} Inf values!")
        inf_cols = np.where(np.isinf(df_raw).any(axis=0))[0]
        print(f"   Columns with Inf: {inf_cols}")
    
    # 统计信息
    print(f"\n3. Data Statistics:")
    print(f"   Min value: {df_raw.min():.6f}")
    print(f"   Max value: {df_raw.max():.6f}")
    print(f"   Mean value: {df_raw.mean():.6f}")
    print(f"   Std value: {df_raw.std():.6f}")
    
    # 检查异常大的值
    threshold = 1e6
    large_values = (np.abs(df_raw) > threshold).sum()
    if large_values > 0:
        print(f"\n   WARNING: Found {large_values} values with absolute value > {threshold}")
    
    # 检查零方差列
    print(f"\n4. Zero Variance Check:")
    zero_var_cols = []
    for i in range(df_raw.shape[1]):
        if df_raw[:, i].std() < 1e-8:
            zero_var_cols.append(i)
    
    if zero_var_cols:
        print(f"   WARNING: Found {len(zero_var_cols)} columns with zero or near-zero variance")
        print(f"   Columns: {zero_var_cols[:10]}...")  # 只显示前10个
    else:
        print(f"   All columns have sufficient variance")
    
    # 按列统计
    print(f"\n5. Per-column Statistics (first 5 columns):")
    for i in range(min(5, df_raw.shape[1])):
        col_data = df_raw[:, i]
        print(f"   Column {i}: min={col_data.min():.4f}, max={col_data.max():.4f}, "
              f"mean={col_data.mean():.4f}, std={col_data.std():.4f}")
    
    # 标准化测试
    print(f"\n6. Standardization Test:")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # 使用70%的数据作为训练集
    num_train = int(len(df_raw) * 0.7)
    train_data = df_raw[:num_train]
    
    try:
        scaler.fit(train_data)
        scaled_data = scaler.transform(df_raw)
        
        print(f"   Scaled data shape: {scaled_data.shape}")
        print(f"   Scaled min: {scaled_data.min():.6f}")
        print(f"   Scaled max: {scaled_data.max():.6f}")
        print(f"   Scaled mean: {scaled_data.mean():.6f}")
        print(f"   Scaled std: {scaled_data.std():.6f}")
        
        # 检查标准化后是否有NaN或Inf
        if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
            print(f"   ERROR: Standardization produced NaN or Inf values!")
            nan_after = np.isnan(scaled_data).sum()
            inf_after = np.isinf(scaled_data).sum()
            print(f"   NaN after scaling: {nan_after}")
            print(f"   Inf after scaling: {inf_after}")
        else:
            print(f"   Standardization successful!")
            
    except Exception as e:
        print(f"   ERROR during standardization: {e}")
    
    print("\n=== Check Complete ===")
    
    # 建议
    print("\n=== Recommendations ===")
    if nan_count > 0 or inf_count > 0:
        print("1. Clean the data to remove NaN and Inf values")
    if large_values > 0:
        print("2. Consider clipping or removing extreme outliers")
    if zero_var_cols:
        print("3. Remove or handle zero-variance columns")
    if nan_count == 0 and inf_count == 0 and large_values == 0:
        print("Data looks clean! If still getting NaN loss, try:")
        print("1. Reduce learning rate (e.g., 1e-6 or 5e-6)")
        print("2. Reduce batch size")
        print("3. Increase dropout")
        print("4. Check model initialization")

if __name__ == "__main__":
    import sys
    
    # 默认路径
    root_path = "../dataset/Solar/"
    data_path = "solar_AL.txt"
    
    # 允许命令行参数
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    
    check_solar_data(root_path, data_path)
