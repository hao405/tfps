from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchTST_MoE_cluster
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile

from layers.Cluster import EDESC
from layers.InitializeD import Initialization_D
from layers.RevIN import RevIN

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTST_MoE_cluster': PatchTST_MoE_cluster,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # NOTE: do NOT wrap the model with DataParallel here. Wrapping must happen
        # after the model is moved to the target device to ensure parameters/buffers
        # reside on the expected device (cuda:0). The actual DataParallel wrapping
        # is performed in Exp_Basic.__init__ after .to(self.device).
        if self.args.use_multi_gpu and self.args.use_gpu:
            if torch.version.hip is not None:
                print('AMD ROCm detected; multi-GPU requested. DataParallel will be applied after moving model to device.')
            else:
                print('NVIDIA CUDA detected; multi-GPU requested. DataParallel will be applied after moving model to device.')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        为非稳态时序优化器配置
        使用AdamW with weight decay来提高鲁棒性
        """
        # 对不同的参数组使用不同的学习率
        # MoE的路由参数需要更小心的更新
        param_groups = []
        
        # 聚类/路由相关参数（D矩阵等）使用更小的学习率
        routing_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'cluster' in name.lower() or 'D' in name or 'routing' in name.lower():
                    routing_params.append(param)
                else:
                    other_params.append(param)
        
        # 为路由参数设置更保守的学习率（1/10）
        if len(routing_params) > 0:
            param_groups.append({
                'params': routing_params,
                'lr': self.args.learning_rate * 0.1,
                'weight_decay': 0.01
            })
            print(f"Routing parameters: {len(routing_params)} params with lr={self.args.learning_rate * 0.1}")
        
        if len(other_params) > 0:
            param_groups.append({
                'params': other_params,
                'lr': self.args.learning_rate,
                'weight_decay': 0.01
            })
            print(f"Other parameters: {len(other_params)} params with lr={self.args.learning_rate}")
        
        # 使用AdamW而非Adam，weight decay有助于防止非稳态导致的过拟合
        model_optim = optim.AdamW(param_groups if param_groups else self.model.parameters(), 
                                  lr=self.args.learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-8,
                                  weight_decay=0.01)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _check_nan_inf(self, tensor, name="tensor"):
        """检查tensor中是否有NaN或Inf值"""
        if torch.isnan(tensor).any():
            print(f"Warning: {name} contains NaN values!")
            return True
        if torch.isinf(tensor).any():
            print(f"Warning: {name} contains Inf values!")
            return True
        return False
    
    def _clip_gradients(self, model, max_norm=1.0):
        """梯度裁剪"""
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm
    
    def _diagnose_model_state(self, model, batch_data, iteration):
        """
        诊断模型状态 - 针对非稳态时序的专门监控
        """
        diagnostics = {}
        
        # 检查模型参数
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                if grad_norm > 100 or param_norm > 1000:
                    diagnostics[name] = {
                        'grad_norm': grad_norm,
                        'param_norm': param_norm,
                        'status': 'CRITICAL'
                    }
        
        # 检查输入数据的统计特性（非稳态检测）
        if batch_data is not None:
            data_mean = batch_data.mean().item()
            data_std = batch_data.std().item()
            data_max = batch_data.max().item()
            data_min = batch_data.min().item()
            
            diagnostics['input_data'] = {
                'mean': data_mean,
                'std': data_std,
                'max': data_max,
                'min': data_min,
                'range': data_max - data_min
            }
            
            # 非稳态检测：如果数据范围或标准差异常
            if data_std < 1e-6 or data_std > 1e6:
                diagnostics['input_data']['status'] = 'NON_STATIONARY_WARNING'
        
        if diagnostics:
            print(f"\n{'='*50}")
            print(f"Model Diagnostics at Iteration {iteration}")
            print(f"{'='*50}")
            for key, value in diagnostics.items():
                print(f"{key}: {value}")
            print(f"{'='*50}\n")
        
        return diagnostics

    def _get_profile(self, model):
        _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
        macs, params = profile(model, inputs=(_input,))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def _refined_subspace_affinity(self, s):
        """
        计算refined subspace affinity
        针对非稳态时序的自适应软分配策略
        """
        eps = 1e-8
        
        # 检查输入的有效性
        if torch.isnan(s).any() or torch.isinf(s).any():
            print("Warning: Invalid values in subspace affinity input, using uniform distribution")
            return torch.ones_like(s) / s.shape[1]
        
        # 使用温和的非线性变换，避免极端值
        # 对于非稳态数据，使用平方根而非平方可以减小方差
        s_clamped = torch.clamp(s, min=eps, max=1e6)
        
        # 使用log-sum-exp技巧进行数值稳定的softmax风格归一化
        # 这对处理非稳态分布的极端值非常重要
        s_log = torch.log(s_clamped + eps)
        
        # 沿着聚类维度进行归一化
        s_max = torch.max(s_log, dim=1, keepdim=True)[0]
        s_exp = torch.exp(s_log - s_max)
        
        # 平方操作但限制幅度
        weight = torch.clamp(s_exp ** 2, max=1e6)
        
        # 两次归一化以获得refined affinity
        weight_sum = weight.sum(0, keepdim=True) + eps
        weight_norm = weight / weight_sum
        
        weight_norm_sum = weight_norm.sum(1, keepdim=True) + eps
        result = weight_norm / weight_norm_sum
        
        # 最终的安全检查
        result = torch.where(torch.isnan(result) | torch.isinf(result), 
                           torch.ones_like(result) / result.shape[1], 
                           result)
        
        return result

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Update refined subspace affinity
                tmp_s_time = s_time.data
                s_tilde_time = self._refined_subspace_affinity(s=tmp_s_time)
                tmp_s_frequency = s_frequency.data
                s_tilde_frequency = self._refined_subspace_affinity(s=tmp_s_frequency)

                # Total loss function
                n_z = self.args.c_out * self.args.d_model
                T_dim = int(n_z / self.args.T_num_expert)
                F_dim = int(n_z / self.args.F_num_expert)
                loss_cluster_time = self.model.model_time.cluster.total_loss(pred=s_time, target=s_tilde_time,
                                                                        dim=T_dim, n_clusters=self.args.T_num_expert,
                                                                        beta=self.args.beta)
                loss_cluster_frequency = self.model.model_frequency.cluster.total_loss(pred=s_frequency, target=s_tilde_frequency,
                                                                             dim=F_dim, n_clusters=self.args.F_num_expert,
                                                                             beta=self.args.beta)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true) + self.args.alpha * loss_cluster_time + self.args.gama * loss_cluster_frequency

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self._get_profile(self.model)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    
                    # 检查模型输出是否包含NaN或Inf
                    if self._check_nan_inf(outputs, "model outputs"):
                        print(f"Skipping batch {i} due to NaN/Inf in outputs")
                        continue
                    if self._check_nan_inf(s_time, "s_time"):
                        print(f"Skipping batch {i} due to NaN/Inf in s_time")
                        continue
                    if self._check_nan_inf(s_frequency, "s_frequency"):
                        print(f"Skipping batch {i} due to NaN/Inf in s_frequency")
                        continue
                    
                    # Update refined subspace affinity
                    tmp_s_time = s_time.data
                    s_tilde_time = self._refined_subspace_affinity(s=tmp_s_time)
                    tmp_s_frequency = s_frequency.data
                    s_tilde_frequency = self._refined_subspace_affinity(s=tmp_s_frequency)

                    # 检查refined subspace affinity输出
                    if self._check_nan_inf(s_tilde_time, "s_tilde_time"):
                        print(f"Skipping batch {i} due to NaN/Inf in s_tilde_time")
                        continue
                    if self._check_nan_inf(s_tilde_frequency, "s_tilde_frequency"):
                        print(f"Skipping batch {i} due to NaN/Inf in s_tilde_frequency")
                        continue

                    # Total loss function
                    n_z = self.args.c_out * self.args.d_model
                    T_dim = int(n_z / self.args.T_num_expert)
                    F_dim = int(n_z / self.args.F_num_expert)
                    
                    # 计算聚类损失，添加异常检查
                    try:
                        loss_cluster_time = self.model.model_time.cluster.total_loss(pred=s_time, target=s_tilde_time,
                                                                           dim=T_dim, n_clusters=self.args.T_num_expert,
                                                                           beta=self.args.beta)
                        loss_cluster_frequency = self.model.model_frequency.cluster.total_loss(pred=s_frequency, target=s_tilde_frequency,
                                                                           dim=F_dim, n_clusters=self.args.F_num_expert,
                                                                           beta=self.args.beta)
                        
                        # 检查聚类损失
                        if self._check_nan_inf(loss_cluster_time, "loss_cluster_time"):
                            print(f"Skipping batch {i} due to NaN/Inf in cluster time loss")
                            continue
                        if self._check_nan_inf(loss_cluster_frequency, "loss_cluster_frequency"):
                            print(f"Skipping batch {i} due to NaN/Inf in cluster frequency loss")
                            continue
                    except Exception as e:
                        print(f"Error computing cluster losses: {e}")
                        continue
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # 计算前向损失
                    loss_fore = criterion(outputs, batch_y)
                    
                    # 检查前向损失
                    if self._check_nan_inf(loss_fore, "loss_fore"):
                        print(f"Skipping batch {i} due to NaN/Inf in forecasting loss")
                        continue
                    
                    # 总损失，使用较小的系数
                    loss = loss_fore + self.args.alpha * loss_cluster_time + self.args.gama * loss_cluster_frequency
                    
                    # 最终检查总损失
                    if self._check_nan_inf(loss, "total loss"):
                        print(f"Skipping batch {i} due to NaN/Inf in total loss")
                        continue

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    # 定期诊断模型状态（非稳态监控）
                    if (i + 1) % 500 == 0:
                        self._diagnose_model_state(self.model, batch_x, i + 1)
                    
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # 梯度裁剪
                    grad_norm = self._clip_gradients(self.model, max_norm=1.0)
                    if grad_norm > 10.0:  # 梯度过大时的警告
                        print(f"Warning: Large gradient norm: {grad_norm}")
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪
                    grad_norm = self._clip_gradients(self.model, max_norm=1.0)
                    if grad_norm > 10.0:  # 梯度过大时的警告
                        print(f"Warning: Large gradient norm: {grad_norm}")
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in self.model.parameters())

        print("模型总参数数量:", total_params)

        preds = []
        trues = []
        clusters_time = []
        clusters_frequency = []
        inputx = []
        inference_time = 0  # 初始化 inference_time
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()  # 计时开始
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                inference_time += time.time() - start_time  # 计算推理时间
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                cluster_time = s_time.detach().cpu().numpy()
                cluster_frequency = s_frequency.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                cluster_time = cluster_time
                cluster_frequency = cluster_frequency

                preds.append(pred)
                trues.append(true)
                clusters_time.append(cluster_time)
                clusters_frequency.append(cluster_frequency)

                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        clusters_time = np.array(clusters_time)
        clusters_frequency = np.array(clusters_frequency)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # 将总体推理时间除以推理次数，得到平均推理时间
        total_samples = len(test_loader)
        if total_samples > 0:
            inference_time /= total_samples

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        result_file = './result_{}_{}.txt'.format(self.args.data_path.split('.')[0], self.args.seq_len)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, Average Inference Time:{}, total_params:{}'.format(mse, mae, rse, inference_time, total_params))
        f = open(result_file, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, Average Inference Time:{}, total_params:{}'.format(mse, mae, rse, inference_time, total_params))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'cluster_time_result.npy', clusters_time)
        np.save(folder_path + 'cluster_frequency_result.npy', clusters_frequency)
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
