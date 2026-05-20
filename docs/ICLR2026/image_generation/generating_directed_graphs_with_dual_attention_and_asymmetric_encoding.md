---
title: >-
  [论文解读] Generating Directed Graphs with Dual Attention and Asymmetric Encoding
description: >-
  [ICLR 2026][图像生成][有向图生成] 提出 Directo，首个基于离散流匹配（Discrete Flow Matching）的有向图生成模型，通过方向感知的双注意力机制和非对称位置编码捕获有向边的方向依赖，同时建立了有向图生成的标准化评测体系。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "有向图生成"
  - "离散流匹配"
  - "双注意力机制"
  - "非对称位置编码"
  - "图生成基准"
---

# Generating Directed Graphs with Dual Attention and Asymmetric Encoding

**会议**: ICLR 2026  
**arXiv**: [2506.16404](https://arxiv.org/abs/2506.16404)  
**代码**: [GitHub](https://github.com/acarballocastro/DIRECTO)  
**领域**: 图生成  
**关键词**: 有向图生成, 离散流匹配, 双注意力机制, 非对称位置编码, 图生成基准  

## 一句话总结

提出 Directo，首个基于离散流匹配（Discrete Flow Matching）的有向图生成模型，通过方向感知的双注意力机制和非对称位置编码捕获有向边的方向依赖，同时建立了有向图生成的标准化评测体系。

## 研究背景与动机

- **领域现状**：图生成模型在药物发现、社交网络建模等领域取得显著进展，但绝大多数方法专注于无向图生成，有向图生成研究严重不足
- **现有痛点**：有向图（digraph）存在两大瓶颈——(1) 建模层面，边方向性使得可学习空间急剧增长（$n=4$ 时有向图 218 种 vs 无向图仅 11 种），简单扩展无向图架构无法处理这种组合爆炸；(2) 评估层面，缺乏标准化的有向图生成基准和评测指标
- **核心矛盾**：已有少量 DAG 生成方法（D-VAE、LayerDAG）需要拓扑排序预处理，且仅限于无环图；而一般有向图的生成方法几乎空白
- **本文目标**：构建一个通用的有向图生成框架，同时覆盖 DAG 和一般有向图
- **切入角度**：从架构（方向感知注意力）、生成框架（离散流匹配）、输入特征（非对称位置编码）三个维度协同增强有向图建模能力
- **核心 idea**：通过双注意力区分入边/出边的信息流，结合有向位置编码和离散流匹配框架，实现对有向图结构的精确生成

## 方法详解

### 整体框架

Directo 构建在离散流匹配（DFM）框架之上。生成过程从噪声分布 $\bm{p}_0$ 出发，通过连续时间马尔可夫链（CTMC）逐步去噪恢复原始图分布 $\bm{p}_1$。去噪网络预测干净图的节点/边分布，驱动速率矩阵 $\bm{R}_t^\theta$ 控制转移。

### 关键设计

#### 1. 有向图的离散流匹配

- **功能**：将 DFM 扩展到有向图设置
- **核心思路**：与无向图不同，有向图中边 $(i,j)$ 和 $(j,i)$ 是独立的分类变量，可属于不同类别。噪声过程在每个节点 $x^{(n)}$ 和每条有向边 $e^{(i,j)}$ 上独立进行线性插值：$p_{t|1}^X(x_t^{(n)} | x_1^{(n)}) = t \cdot \delta(x_t^{(n)}, x_1^{(n)}) + (1-t) \cdot p_{\text{noise}}^X(x_t^{(n)})$
- **训练损失**：交叉熵损失 $\mathcal{L} = \mathbb{E}[-\sum_n \log p_{1|t}^{\theta,(n)} - \lambda \sum_{i \neq j} \log p_{1|t}^{\theta,(i,j)}]$
- **设计动机**：DFM 的训练-采样解耦特性允许后训练优化（如自适应时间调度），这对处理有向图的额外复杂度尤为重要

#### 2. 非对称位置编码

- **功能**：为有向图设计方向感知的位置编码，注入超越局部邻域的结构信息
- **核心思路**：实验了三种方案——(1) 磁拉普拉斯（MagLap）：通过复值相位偏移保留边方向；(2) Multi-$q$ MagLap：堆叠多个不同势函数的 MagLap 提供更丰富表示；(3) 有向 RRWP：组合正向/反向转移概率捕获入出方向的非对称流
- **设计动机**：标准拉普拉斯编码基于对称邻接矩阵，无法区分非对称结构角色

#### 3. 双注意力（Dual Attention）机制

- **功能**：设计能区分源→目标和目标→源信息流的注意力模块
- **核心思路**：构建两组方向特定的注意力映射：
    - 源→目标方向：$\bm{Y}_{\text{ST}}[i,j] = \frac{\bm{Q}_S[i] \cdot \bm{K}_T[j]}{\sqrt{d_q}}$
    - 目标→源方向：$\bm{Y}_{\text{TS}}[i,j] = \frac{\bm{Q}_T[i] \cdot \bm{K}_S[j]}{\sqrt{d_q}}$
  
  通过 FiLM 层用边特征调制后，将两个方向的注意力拼接并统一 softmax：$\bm{A}_{\text{aggr}} = \text{softmax}(\text{concat}(\bm{Y}'_{\text{ST}}, \bm{Y}'_{\text{TS}}))$，聚合后更新节点特征 $\bm{X}' = \bm{A}_{\text{aggr}} \bm{V}_{\text{aggr}}$
- **设计动机**：标准注意力对称处理边关系，无法捕获有向图中源节点和目标节点的不同语义

#### 4. 多尺度信息调制

- **FiLM 层**：$\text{FiLM}(\bm{E}, \bm{E}_{\text{attn}}) = \bm{E} \bm{W}_1 + (\bm{E} \bm{W}_2) \odot \bm{E}_{\text{attn}} + \bm{E}_{\text{attn}}$，实现节点-边-图三级特征融合
- **PNA 层**：通过 max/min/mean/std 四种池化聚合局部信息，更新全局图特征

### 损失函数

标准交叉熵损失，独立应用于节点和边的预测，超参数 $\lambda$ 控制边重建损失的权重。

## 实验关键数据

### 主实验

| 模型 | ER-DAG Ratio↓ | ER-DAG V.U.N.↑ | SBM Ratio↓ | SBM V.U.N.↑ | TPU V.U.N.↑ | VG RBF↓ |
|---|---|---|---|---|---|---|
| MLE | 15.1 | 0.0 | 11.6 | 0.0 | 24.7 | 0.618 |
| LayerDAG | 4.2 | 21.5 | - | - | 98.5 | - |
| DeFoG | 1.6 | 75.0 | 4.3 | 37.0 | 72.0 | 0.085 |
| Directo RRWP | **1.7** | **94.0** | 1.8 | **99.5** | 77.0 | **0.038** |
| Directo MagLap | **1.3** | 92.0 | **2.0** | 96.5 | **80.5** | 0.051 |

### 消融实验

| 配置 | ER-DAG V.U.N.↑ | SBM V.U.N.↑ |
|---|---|---|
| RRWP + Double depth（加参数不加双注意力）| 72.0 | 0.0 |
| RRWP + Dual attention | **94.0** | **99.5** |
| MagLap + Double depth | 80.0 | 8.0 |
| MagLap + Dual attention | **91.0** | **77.0** |

### 关键发现

1. **双注意力是核心**：即使不加位置编码（No PE），双注意力仍能实现非零 V.U.N.，而单纯增加网络深度完全无法替代
2. **方向感知 PE 优于通用 PE**：有向 RRWP 和 MagLap 显著优于对称拉普拉斯编码
3. **隐式学习结构约束**：Directo 在 DAG 数据上无需显式无环约束就能生成高质量 DAG
4. LayerDAG 虽然强制 acyclicity，但在分布匹配指标（Ratio）上远不如 Directo

## 亮点与洞察

1. **系统性方案**：同时解决了有向图生成的建模难题和评测空白，是该方向的奠基性工作
2. **架构设计精巧**：双注意力通过拼接两个方向后统一 softmax，允许模型自适应分配入/出方向的注意力权重，比独立处理更有效
3. **可扩展性好**：通过 classifier-free guidance 可直接扩展到条件生成
4. **消融实验说服力强**：Table 2 清楚展示双注意力 vs 简单加深网络的巨大差距

## 局限与展望

1. **可扩展性有限**：目前仅测试到 ~200 节点的中等规模图，大规模图需要稀疏注意力等加速策略
2. **无显式结构约束**：对 acyclicity 等性质仅隐式学习，强约束场景可能需要结合 PRODIGY 等方法
3. **位置编码计算成本**：多 $q$ MagLap 在大图上计算开销显著
4. **仅验证生成任务**：双注意力架构可推广到判别任务（链接预测、节点分类），但尚未验证

## 相关工作与启发

- **无向图生成**：DiGress（Vignac et al., 2023）和 DeFoG（Qin et al., 2025）是最强基线，而朴素扩展移除对称化操作后效果大幅下降
- **DAG 生成**：D-VAE、LayerDAG 需拓扑排序，限制了通用性
- **有向 GNN**：MagNet（Zhang et al., 2021）和 Dir-GNN 的位置编码思路被本文借鉴到生成任务
- **启发**：流匹配在离散空间的成功应用值得关注，其训练-采样解耦特性在复杂结构生成中优势明显

## 评分

⭐⭐⭐⭐（4/5）

- **创新性**：⭐⭐⭐⭐ 首个基于 flow matching 的有向图生成模型，问题定义清晰
- **实验**：⭐⭐⭐⭐⭐ 合成+真实数据集全面评测，消融充分
- **写作**：⭐⭐⭐⭐ 结构清晰，benchmark 设计合理
- **实用性**：⭐⭐⭐⭐ 提供了完整评测框架和代码

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)
- [\[ICCV 2025\] Video Motion Graphs](../../ICCV2025/image_generation/video_motion_graphs.md)
- [\[ICML 2025\] Directed Graph Grammars for Sequence-based Learning](../../ICML2025/image_generation/directed_graph_grammars_for_sequence-based_learning.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](../../CVPR2026/image_generation/trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](../../CVPR2026/image_generation/cognitioncapturerpro_towards_highfidelity_visual_d.md)

</div>

<!-- RELATED:END -->
