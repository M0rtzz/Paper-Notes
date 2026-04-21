---
title: >-
  [论文解读] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration
description: >-
  [CVPR 2026][自动驾驶][轨迹预测] FoSS 提出一种频域-时域双分支框架，通过渐进螺旋重排序（HelixSort）将傅里叶频谱有序化后输入选择性状态空间模型（SSM），结合时域动态 SSM 和交叉注意力融合，在 Argoverse 1/2 上取得 SOTA 轨迹预测精度，同时参数量减少 40%+、推理延迟降低 22%。
tags:
  - CVPR 2026
  - 自动驾驶
  - 轨迹预测
  - 傅里叶变换
  - 状态空间模型
  - 双分支架构
  - 多模态预测
---

# FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration

**会议**: CVPR 2026  
**arXiv**: [2603.01284](https://arxiv.org/abs/2603.01284)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 轨迹预测, 傅里叶变换, 状态空间模型, 双分支架构, 多模态预测

## 一句话总结
FoSS 提出一种频域-时域双分支框架，通过渐进螺旋重排序（HelixSort）将傅里叶频谱有序化后输入选择性状态空间模型（SSM），结合时域动态 SSM 和交叉注意力融合，在 Argoverse 1/2 上取得 SOTA 轨迹预测精度，同时参数量减少 40%+、推理延迟降低 22%。

## 研究背景与动机
自动驾驶中精确的轨迹预测对安全至关重要，尤其在密集多智能体环境中。现有方法面临三重需求的固有权衡：建模跨智能体的长程依赖、表示多模态未来以捕获不确定性、以及满足严格的实时约束。

Transformer 架构通过自注意力实现高精度，但计算复杂度为 $\mathcal{O}(N^2)$，限制了在资源受限系统上的部署。循环模型虽然效率高，但难以捕获长程依赖和细粒度局部动态。仅在时域建模的方法往往混淆全局运动模式与局部动态，而标准傅里叶表示缺乏有序的频率语义，使序列模型难以有效处理频谱信息。

核心观察：轨迹信号在频谱域和时域呈现互补结构——**振幅谱编码全局运动趋势，相位谱捕获细粒度时间变化**。但 DFT 输出不保持从低频到高频的连续排列（因 $\omega$ 和 $T-\omega$ 对应同一物理频率），直接给 SSM 处理会导致模型在全局和局部推理之间不断切换，破坏状态演化。

本文的核心 idea：设计渐进螺旋重排序模块（HelixSort）将 DFT 系数按频谱半径排序，使 SSM 能以粗到细方式处理频谱信息，结合时域 SSM 的长程依赖建模，实现线性复杂度下的高精度多模态轨迹预测。

## 方法详解

### 整体框架
FoSS 采用双分支架构：(1) **频域分支（FD-Mamba）**——对历史轨迹进行 DFT 分解为振幅和相位，经 HelixSort 重排后输入两个并行的 SSM 子模块（Coarse2Fine-SSM 处理空间交互，SpecEvolve-SSM 处理通道演化）；(2) **时域分支（TD-Mamba）**——通过输入依赖的动态 SSM 直接在时序序列上建模长程依赖。两分支通过交叉注意力层融合，再由可学习查询向量解码出 $K$ 条候选轨迹，经加权融合输出最终预测。

### 关键设计
1. **渐进螺旋重排序（HelixSort）**:

    - 功能：将 DFT 输出的无序频谱系数重排为按频率单调递增的序列
    - 核心思路：(a) 将 1D DFT 系数 $F^{(k)} \in \mathbb{C}^T$ reshape 为 2D 网格 $\mathcal{F}^{(k)} \in \mathbb{C}^{\sqrt{T} \times \sqrt{T}}$；(b) 从频谱中心 $(u_0, v_0)$ 开始，按螺旋方向向外遍历，按频谱半径 $r = \sqrt{(u-u_0)^2 + (v-v_0)^2}$ 升序排列；(c) 生成重排索引 $\pi^{(k)}$，得到满足 $\forall i < j, r_i \leq r_j$ 的有序序列 $\widehat{F}^{(k)}$
    - 设计动机：标准 DFT 的低/高频系数交错排列，迫使 SSM 在全局和局部推理间反复切换，破坏状态演化。HelixSort 将低频集中在序列起始、高频移至末尾，使 SSM 先积累全局趋势再精调局部细节，符合粗到细推理策略。开销极小：仅 0.08% FLOPs 增加、<0.25MB 内存

2. **频域双子模块（Coarse2Fine-SSM + SpecEvolve-SSM）**:

    - 功能：分别从空间和通道维度深度建模频域特征
    - 核心思路：
        - **Coarse2Fine-SSM**：对输入特征做 FFT → HelixSort 重排振幅和相位 → 深度可分离卷积 + SiLU + 选择性 SSM + LayerNorm → iFFT 恢复时域 → 与原始特征逐元素相乘：$F_f = \text{iFFT}(A'(F_l), P'(F_l)) \odot \text{SiLU}(F_l)$
        - **SpecEvolve-SSM**：全局平均池化提取通道特征 $F_g \in \mathbb{R}^{1 \times 1 \times C}$ → 通道维度 FFT → 按频谱幅度升序排列 → iFFT → 逐元素门控融合：$F_a = \text{iFFT}(A(F_g)', P(F_g)') \odot \text{SiLU}(F_g)$，增强表示 $F_{\text{enhance}} = F_a \odot F_{in}$
        - 两子模块输出沿通道拼接后线性投影，得到最终频域表示 $F_{\text{freq}}$
    - 设计动机：空间维度的粗到细建模捕获运动轨迹的空间交互，通道维度的频谱演化捕获不同特征维度间的相关性，二者互补

3. **时域动态选择性 SSM（TD-Mamba）**:

    - 功能：在时域以线性复杂度模拟自注意力行为，捕获长程时序依赖
    - 核心思路：状态转移矩阵 $A_t, B_t, C_t, D_t$ 均由当前输入 $X(t)$ 及其局部卷积特征 $\tilde{X}(t) = \text{Conv1D}(X(t))$ 动态生成：$A_t = f_A(X(t), \tilde{X}(t))$ 等，其中 $f_A$ 等为轻量 MLP。状态更新：$h(t+1) = A_t h(t) + B_t X(t)$，输出 $Y_{\text{time}}(t) = C_t h(t) + D_t X(t)$。隐状态经 SiLU + LayerNorm 保证数值稳定
    - 设计动机：输入依赖的参数化使状态更新能根据不同时间步自动放大关键运动模式、抑制噪声，Conv1D 预处理增强对局部动态变化的敏感度

### 损失函数 / 训练策略
- 联合约束时域和频域：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \lambda \mathcal{L}_{\text{freq}}$
- 时域损失：$\mathcal{L}_{\text{time}} = \|\hat{Y}_{\text{final}} - Y\|_1$（L1 距离）
- 频域损失：$\mathcal{L}_{\text{freq}} = \|F(\hat{Y}_{\text{final}}) - F(Y)\|_1$（傅里叶变换后的 L1 距离）
- 候选轨迹通过可学习查询 $Q \in \mathbb{R}^{K \times d}$ 与融合特征 $Z$ 的交叉注意力生成，每条候选经 MLP 映射
- 训练：Adam 优化器，学习率 0.001，batch 128，50 epochs，验证集无改善 5 轮则学习率降至 10%

## 实验关键数据

### 主实验

| 数据集 | 指标 | FoSS | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| Argoverse 2 | b-minFDE6↓ | **1.69** | 1.74 (Wayformer) | 2.9% |
| Argoverse 2 | minADE6↓ | **0.61** | 0.63 (DeMo) | 3.2% |
| Argoverse 2 | minFDE6↓ | **1.07** | 1.17 (HiVT/DeMo) | 8.5% |
| Argoverse 2 | MR6↓ | **0.11** | 0.12 (Wayformer) | 8.3% |
| Argoverse 2 | 参数量 | **4.18M** | 5.92M (DeMo) | -29.4% |
| Argoverse 1 | minADE1↓ | **1.67** | 1.65 (DeMo) | 约持平 |
| Argoverse 1 | minFDE1↓ | 2.05 | **2.02** (Wayformer) | 约持平 |
| Argoverse 1 | MR1↓ | **0.11** | 0.11 (多个) | 持平 |

### 消融实验（Argoverse 2）

| 配置 | minADE6↓ | minFDE6↓ | MR6↓ | 说明 |
|------|---------|---------|------|------|
| w/o 频域分支 | 0.71 | 1.36 | 0.17 | 频域线索对全局趋势建模至关重要 |
| w/o HelixSort | 0.69 | 1.32 | 0.16 | 有序频谱遍历提升结构连贯性 |
| w/o Fourier SSM | 0.70 | 1.35 | 0.17 | 选择性 SSM 对频谱时序融合不可或缺 |
| Concat+MLP 替换交叉注意力 | 0.69 | 1.33 | 0.16 | token 级交叉交互优于简单拼接 |
| **Full model** | **0.65** | **1.29** | **0.15** | 所有组件互补最优 |

### 关键发现
- 频域分支可即插即用到其他时序骨干：FD-Mamba+Transformer（b-minFDE 1.77）、FD-Mamba+LSTM（1.91），验证通用性
- 推理延迟仅 64ms（比 QCNet 71ms 快 10%，比 HiVT 82ms 快 22%），FLOPs 22.1G（比 QCNet 51%）
- 参数量 4.18M 为所有对比方法中最小，充分证明效率优势
- 在频繁换道等高频运动场景下仍有轻微抖动，因低频分解可能低估快速横向机动

## 亮点与洞察
- HelixSort 是一个简洁而有效的模块设计：将 JPEG zigzag 编码的思想迁移到频谱重排序，使 SSM 获得有序的频率输入，几乎零开销（0.08% FLOPs）
- 将 DFT 系数的频谱半径解释为"人工时间轴"输入 SSM，巧妙地将频域分析转化为序列建模问题
- 双分支架构的互补性设计合理：频域分支捕获全局模式和局部变化的分离表征，时域分支保留原始时序上下文
- 频域损失的引入保证了预测轨迹不仅在位置上精确，在频率结构上也与真实轨迹一致

## 局限与展望
- 在频繁换道等涉及突发高频运动的场景表现略有下降，因频域分解天然偏向低频成分
- Argoverse 1 上相比 DeMo/Wayformer 的优势不如 Argoverse 2 显著（短程预测的频域优势有限）
- 未考虑地图编码（仅用轨迹数据），与使用高清地图的方法对比有失公平
- HelixSort 需要将序列长度 padding 到完美平方数，对非标准长度输入的处理是否最优未充分讨论

## 相关工作与启发
- 与 Spectral TGN（ICRA 2021）等频域轨迹预测方法比较，FoSS 的创新在于 HelixSort + SSM 的结合，解决了频谱无序性问题
- Mamba/S4 等选择性 SSM 在视频、语音中的成功实践为将其引入轨迹预测提供了基础
- 交叉注意力融合通过归一化和残差连接解决了时域/频域特征尺度不匹配问题，是多域融合的通用策略
- 频域分支的即插即用特性提示可推广到其他时序预测任务（如行为预测、交通流预测）

## 评分
- 新颖性: ⭐⭐⭐⭐ HelixSort + SSM 的频域建模组合有原创性，但双分支融合的整体框架相对常规
- 实验充分度: ⭐⭐⭐⭐ 在 Argoverse 1/2 上全面对比+消融+效率分析+即插即用验证，但缺少 nuScenes 等其他数据集
- 写作质量: ⭐⭐⭐⭐ 公式推导完整，HelixSort 的直觉解释清晰，Figure 1 的频率分解可视化有助理解
- 价值: ⭐⭐⭐⭐ 参数量和延迟的大幅降低对实际部署有直接价值，频域+SSM 的范式对轨迹预测社区有参考意义

<!-- RELATED:START -->

## 相关论文

- [U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences](u4d_uncertainty-aware_4d_world_modeling_from_lidar_sequences.md)
- [OccMamba: Semantic Occupancy Prediction with State Space Models](../../CVPR2025/autonomous_driving/occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions](../../AAAI2026/autonomous_driving/walking_further_semantic-aware_multimodal_gait_recognition_under_long-range_cond.md)
- [MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)
- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)

<!-- RELATED:END -->
