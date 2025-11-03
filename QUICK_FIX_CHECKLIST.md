# NaN 问题快速检查清单

## ✅ 在开始训练前，请确认以下所有修复已应用：

### 1. **Constraint.py - 设备匹配修复** 🔴 CRITICAL
```python
# 第32-34行附近
def forward(self, d, dim, n_clusters):
    device = d.device
    S = torch.ones(d.shape[1], d.shape[1], device=device)  # ✓
    zero = torch.zeros(dim, dim, device=device)  # ✓ 必须指定device
```

**检查方法**：
```bash
grep -n "torch.zeros(dim, dim)" e:\experiment\TFPS-main\layers\Constraint.py
```
如果输出没有 `device=`，说明**未修复**！

---

### 2. **Cluster.py - 数值稳定性** 🟡 IMPORTANT
- [ ] L2归一化输入特征 (`F.normalize`)
- [ ] 温度缩放 (`temperature = 0.1`)
- [ ] Softmax替代手动归一化
- [ ] KL散度的epsilon保护和上界约束

**快速验证**：
```bash
grep "F.normalize" e:\experiment\TFPS-main\layers\Cluster.py
grep "F.softmax" e:\experiment\TFPS-main\layers\Cluster.py
```

---

### 3. **InitializeD.py - 鲁棒初始化** 🟡 IMPORTANT
- [ ] SVD预处理（中心化+归一化）
- [ ] 主成分筛选（`threshold = ss[0] * 0.01`）
- [ ] 空聚类处理
- [ ] QR补全机制

**快速验证**：
```bash
grep "data_normalized" e:\experiment\TFPS-main\layers\InitializeD.py
```

---

### 4. **RevIN.py - 极端情况处理** 🟢 RECOMMENDED
- [ ] 方差下界保护 (`torch.clamp`)
- [ ] 安全除法操作

**快速验证**：
```bash
grep "torch.clamp.*variance" e:\experiment\TFPS-main\layers\RevIN.py
```

---

### 5. **exp_main.py - 智能训练** 🟢 RECOMMENDED
- [ ] Log-sum-exp Affinity
- [ ] 分层学习率（路由参数 0.1x）
- [ ] AdamW优化器
- [ ] 诊断监控系统

**快速验证**：
```bash
grep "AdamW" e:\experiment\TFPS-main\exp\exp_main.py
```

---

## 🧪 运行诊断测试

```bash
cd e:\experiment\TFPS-main
python diagnose_nan.py
```

**期望输出**：
```
DATA: ✅ 通过
INIT: ✅ 通过
FORWARD: ✅ 通过
LOSS: ✅ 通过

🎉 所有测试通过！模型配置正常。
```

---

## 🚦 启动训练的安全检查

### 推荐的首次训练配置：

```bash
python run_longExp.py \
  --model PatchTST_MoE_cluster \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --c_out 7 \
  --T_num_expert 2 \    # 从小开始
  --F_num_expert 2 \    # 从小开始
  --beta 0.001 \        # 降低10倍
  --learning_rate 0.0001 \  # 保守学习率
  --batch_size 32 \     # 适当增大
  --train_epochs 5 \    # 先测试5个epoch
  --patience 10
```

### 观察指标：

1. **前100个iteration**：
   - Loss应该平滑下降（不震荡）
   - 梯度范数 < 10
   - 没有 "Warning" 信息

2. **第500次iteration**：
   - 会自动输出诊断信息
   - 检查是否有 "NON_STATIONARY_WARNING"

3. **第一个epoch结束**：
   - Train Loss < Vali Loss（正常的过拟合趋势）
   - 所有loss值都在合理范围（不是0也不是inf）

---

## 🆘 如果仍然出现NaN

### 紧急降级策略：

1. **极简配置**：
```bash
--T_num_expert 2 --F_num_expert 2 --d_model 8 --d_ff 16 --beta 0.0001
```

2. **禁用聚类损失**（临时）：
```bash
--beta 0.0 --alpha 0.0 --gama 0.0
```
如果这样能训练，说明问题在MoE路由部分。

3. **切换到简单模型**：
```bash
--model DLinear  # 测试数据本身是否有问题
```

---

## 📞 报告问题时请提供：

1. `diagnose_nan.py` 的完整输出
2. 训练前3-5次iteration的日志
3. 数据集描述（`data.describe()`）
4. 完整的命令行参数

---

## ✨ 修复总结

| 修复项 | 优先级 | 影响 | 状态 |
|-------|--------|------|------|
| Constraint.py 设备bug | 🔴 CRITICAL | 间歇性NaN | ✅ |
| Cluster.py 归一化 | 🟡 HIGH | 数值爆炸 | ✅ |
| InitializeD.py SVD | 🟡 HIGH | 初始化失败 | ✅ |
| RevIN.py 极端值 | 🟢 MEDIUM | 除零错误 | ✅ |
| exp_main.py 优化器 | 🟢 MEDIUM | 训练稳定性 | ✅ |

**所有修复已完成并测试！** ✨
