# Table 2: 多数据集泛化对比

> 目的：展示 DiBS 在不同来源、不同难度分布的数据集上的泛化能力

## 数据集选择

| Dataset | #Instances | Source | Description |
|---------|------------|--------|-------------|
| Hardest-1106 | 375 | Royle | 论坛公认最难的 1106 题 |
| Top1465 | 1,465 | Royle | Magictour top1465 经典难题 |
| Serg-10k | 10,000 | Royle | Serg benchmark 标准测试集 |
| SATNet-10k | 10,000 | SATNet | SATNet 论文数据集 |
| Kaggle-10k | 10,000 | Kaggle | Kaggle 大规模数据集采样 |
| Hardest-11plus | 10,000 | Royle | 论坛最难的 11+ 题目采样 |
| Hardest-1905 | 10,000 | Royle | 论坛最难的 1905 题目采样 |
| Royle-Kaggle | 10,000 | Royle | Royle Kaggle 数据集采样 |
| Unbiased-10k | 10,000 | Royle | Royle 无偏数据集采样 |

> 注：大数据集采样前 10,000 题

## 表格模板 (LaTeX)

```latex
\begin{table*}[t]
\centering
\caption{DiBS 在多数据集上的泛化性能对比}
\label{tab:generalization}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{llccccccc}
\toprule
Dataset & Solver & Solved\% & Time Mean & Time P95 & Nodes & Backtracks & Speedup & Red\% \\
\midrule
\multirow{2}{*}{Hardest-1106}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Top1465}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Serg-10k}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{SATNet-10k}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Kaggle-10k}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Hardest-11plus}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Hardest-1905}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Royle-Kaggle}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\midrule
\multirow{2}{*}{Unbiased-10k}
    & MRV+FC & --.-- & ---- & ---- & ---- & ---- & -- & -- \\
    & DiBS & --.-- & ---- & ---- & ---- & ---- & --.--× & --.-- \\
\bottomrule
\end{tabular}
\end{table*}
```

## 指标说明

- **Solved%**: 在 30s timeout 内解出的比例
- **Time Mean**: 平均求解时间 (ms)
- **Time P95**: 95 分位时间 (ms)
- **Nodes**: 平均搜索节点数
- **Backtracks**: 平均回溯次数
- **Speedup**: 时间加速比 = Time(MRV+FC) / Time(DiBS)
- **Red%**: 节点减少比例 = (Nodes(MRV+FC) - Nodes(DiBS)) / Nodes(MRV+FC) × 100%

## 预期结论

1. **难题上收益大**: Hardest-1106, Top1465 上节点和时间大幅减少
2. **简单题不拖慢**: Kaggle, SATNet 等简单数据集上开销可控
3. **泛化良好**: 不同来源的数据集都能获得正向收益

## 实验配置

- Timeout: 30s
- Baseline: MRV+FC (最优 CP baseline)
- DiBS 配置: alpha=0.8, smart-call enabled
- 采样策略: 大数据集采样前 10,000 题

## 运行命令

```bash
cd DiBS

# 运行所有数据集
python3 DiBS/run_table2_experiments.py --gpus "0,1,2,3" --solvers "MRV,DiBS"

# 只运行特定数据集
python3 DiBS/run_table2_experiments.py --datasets "Hardest-1106,Top1465" --solvers "MRV,DiBS"

# 测试模式（每个数据集只跑 100 题）
python3 DiBS/run_table2_experiments.py --max-puzzles 100 --solvers "MRV,DiBS"
```
