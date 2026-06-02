# Table 3: 消融实验与调参

> 目的：验证 smart-call 与 consistency 项的必要性，以及参数稳定性

## 数据集

| Dataset | #Instances | Description |
|---------|------------|-------------|
| Royle17-1k (sampled) | 1,000 | Royle 17-clue 数据集随机采样，用于消融实验 |

## 消融实验设计

### 变体说明

| Variant | Description |
|---------|-------------|
| Base | MRV (无模型调用) |
| logits-only | MRV=2 + 仅 logits 排序 (无 consistency) |
| DiBS (full) | MRV=2 + logits + consistency (完整方法) |
| MRV>=3 | 在 MRV<=3 时调用模型 (调用更频繁) |
| always-call | 每次分支都调用模型 (最频繁) |

### 参数调优

| alpha | Description |
|-------|-------------|
| 0.0 | 仅 consistency |
| 0.3 | consistency 权重较高 |
| 0.5 | 均衡 |
| 0.8 | logits 权重较高 (默认) |
| 1.0 | 仅 logits |

## 表格模板 (LaTeX)

### 消融实验

```latex
\begin{table}[t]
\centering
\caption{消融实验：各组件贡献分析}
\label{tab:ablation}
\begin{tabular}{lccccccc}
\toprule
Variant & Solved\% & Time & Nodes & Backtracks & K & Model Time & Overhead\% \\
\midrule
Base (MRV) & --.-- & ---- & ---- & ---- & 0 & 0 & 0 \\
logits-only & --.-- & ---- & ---- & ---- & -- & -- & -- \\
DiBS (full) & --.-- & ---- & ---- & ---- & -- & -- & -- \\
MRV>=3 & --.-- & ---- & ---- & ---- & -- & -- & -- \\
always-call & --.-- & ---- & ---- & ---- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### 参数调优

```latex
\begin{table}[t]
\centering
\caption{参数 $\alpha$ 敏感性分析}
\label{tab:alpha}
\begin{tabular}{lcccccc}
\toprule
$\alpha$ & Solved\% & Time & Nodes & Backtracks & K & Speedup \\
\midrule
0.0 (consistency only) & --.-- & ---- & ---- & ---- & -- & -- \\
0.3 & --.-- & ---- & ---- & ---- & -- & -- \\
0.5 & --.-- & ---- & ---- & ---- & -- & -- \\
0.8 (default) & --.-- & ---- & ---- & ---- & -- & -- \\
1.0 (logits only) & --.-- & ---- & ---- & ---- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

## 指标说明

- **Solved%**: 在 30s timeout 内解出的比例
- **Time**: 平均求解时间 (ms)
- **Nodes**: 平均搜索节点数
- **Backtracks**: 平均回溯次数
- **K**: 模型调用次数
- **Model Time**: 模型推理总时间 (ms)
- **Overhead%**: 模型时间占比 = Model Time / Total Time × 100%
- **Speedup**: 相对 Base 的时间加速比

## 预期结论

### 消融实验
1. **logits-only 有效但不完整**: 比 Base 好，但不如 full DiBS
2. **consistency 有贡献**: full DiBS > logits-only
3. **smart-call 必要**: always-call 开销过大，整体变慢
4. **MRV>=3 收益递减**: 调用更多但收益有限

### 参数调优
1. **alpha=0.8 最优**: logits 权重高时效果最好
2. **参数稳定**: 0.3-1.0 范围内性能波动不大
3. **极端值可行**: 纯 logits 或纯 consistency 仍有效

## 实验配置

- Dataset: Royle17-5k (随机采样 5000 题)
- Timeout: 30s
- Baseline: MRV
- 默认参数: alpha=0.8, smart-call enabled
