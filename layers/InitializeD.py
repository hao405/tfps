from collections import defaultdict
import numpy as np
import torch

'''
def seperate(Z, y_pred, n_clusters):
    # 根据预测标签 y_pred，将矩阵 Z 中的行按聚类标签分离，并存储到一个字典 Z_seperate 中，其中字典的键是聚类标签，值是对应标签的行列表。
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    
    # 该代码的目的是根据聚类标签将数据矩阵 Z 分离，然后对每个聚类的数据进行 SVD 分解，最终将所有聚类的分解结果整合到一个矩阵 D 中，用于初始化某种计算或算法。
    # num_expert * d = n_z

    Z_seperate = seperate(Z, y_pred, n_clusters)
    Z_full = None
    U = np.zeros([n_clusters * d, n_clusters * d])
    print("Initialize D")
    # 对每个聚类中的数据进行奇异值分解 (SVD)，并将结果中的左奇异矩阵 u 的前 d 列赋值给 U 矩阵中的相应位置。
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
        U[:,i*d:(i+1)*d] = u[:,0:d]
    D = U
    print("Shape of D: ", D.transpose().shape)
    print("Initialization of D Finished")
    return D
'''


def seperate(Z, y_pred, n_clusters):
    # 根据预测标签 y_pred，将矩阵 Z 中的行按聚类标签分离，并存储到一个字典 Z_seperate 中，其中字典的键是聚类标签，值是对应标签的行列表。
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = {}
    for i in range(n_clusters):
        Z_seperate[i] = []
    for j in range(len(y_pred)):
        cluster_label = y_pred[j]
        Z_seperate[cluster_label].append(Z[j].cpu().detach().numpy())  # 将Tensor转换为NumPy数组并添加到列表中
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    '''
    改进的D矩阵初始化，专门针对非稳态时序数据
    使用正交化和数值稳定的SVD分解
    '''
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([n_clusters * d, n_clusters * d])
    print("Initialize D for non-stationary time series")
    
    for i in range(n_clusters):
        # 对每个聚类中的数据进行 SVD 分解
        data = np.array(Z_seperate[i])  # 转换为NumPy数组
        
        # 处理空聚类或数据不足的情况
        if len(data) == 0 or data.shape[0] < d:
            print(f"Warning: Cluster {i} has insufficient data ({len(data)} samples), using random orthogonal initialization")
            # 使用随机正交矩阵初始化
            random_matrix = np.random.randn(n_clusters * d, d)
            U[:, i*d:(i+1)*d], _ = np.linalg.qr(random_matrix)
            continue
        
        # 对非稳态数据进行中心化和归一化，提高SVD的数值稳定性
        data_mean = data.mean(axis=0, keepdims=True)
        data_centered = data - data_mean
        
        # 计算数据的尺度并归一化
        data_std = np.std(data_centered, axis=0, keepdims=True) + 1e-8
        data_normalized = data_centered / data_std
        
        # 使用经济型SVD，更加数值稳定
        try:
            u, ss, v = np.linalg.svd(data_normalized.T, full_matrices=False)
            
            # 只取奇异值较大的主成分，过滤噪声
            # 这对非稳态数据特别重要
            threshold = ss[0] * 0.01  # 保留至少是最大奇异值1%的成分
            valid_dims = np.sum(ss > threshold)
            valid_dims = min(valid_dims, d)
            
            if valid_dims < d:
                print(f"Cluster {i}: Only {valid_dims}/{d} dimensions have significant singular values")
                # 补充剩余维度为正交向量
                U[:, i*d:i*d+valid_dims] = u[:, :valid_dims]
                # 使用QR分解生成剩余的正交向量
                if valid_dims < d:
                    remaining = d - valid_dims
                    random_complement = np.random.randn(u.shape[0], remaining)
                    ortho_complement, _ = np.linalg.qr(random_complement)
                    U[:, i*d+valid_dims:(i+1)*d] = ortho_complement
            else:
                U[:, i*d:(i+1)*d] = u[:, :d]
                
        except np.linalg.LinAlgError as e:
            print(f"SVD failed for cluster {i}: {e}, using random orthogonal initialization")
            random_matrix = np.random.randn(n_clusters * d, d)
            U[:, i*d:(i+1)*d], _ = np.linalg.qr(random_matrix)

    D = U
    
    # 最终的正交性检查
    orthogonality = np.linalg.norm(D.T @ D - np.eye(D.shape[1]))
    print(f"Shape of D: {D.T.shape}, Orthogonality error: {orthogonality:.6f}")
    print("Initialization of D Finished")
    return D

