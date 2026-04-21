---
title: >-
  [论文解读] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions
description: >-
  [CVPR 2026][图像分割][无监督域适应] 本文将无监督域适应中的类别课程学习重新定义为强化学习的序贯决策问题，提出 HeuSCM 框架，通过高维语义状态感知和类别公平策略梯度实现自主学习课程规划，在 ACDC、Dark Zurich 和 Nighttime Driving 上达到 SOTA（72.9 mIoU）。
tags:
  - CVPR 2026
  - 图像分割
  - 无监督域适应
  - 语义分割
  - 课程学习
  - 强化学习
  - 恶劣天气
---

# Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions

**会议**: CVPR 2026  
**arXiv**: [2603.24322](https://arxiv.org/abs/2603.24322)  
**代码**: 无  
**领域**: 语义分割 / 域适应  
**关键词**: 无监督域适应, 语义分割, 课程学习, 强化学习, 恶劣天气

## 一句话总结

本文将无监督域适应中的类别课程学习重新定义为强化学习的序贯决策问题，提出 HeuSCM 框架，通过高维语义状态感知和类别公平策略梯度实现自主学习课程规划，在 ACDC、Dark Zurich 和 Nighttime Driving 上达到 SOTA（72.9 mIoU）。

## 研究背景与动机

**领域现状**：无监督域适应语义分割（UDA-SS）是自动驾驶环境感知的核心技术，用于将清晰天气下训练的模型迁移到恶劣天气（雾、雨、夜间）场景。主流方法采用风格迁移缓解域差异，同时通过课程学习（CL）或困难类挖掘（HCM）应对类别不均衡。

**现有痛点**：现有的CL和HCM方法存在根本性范式缺陷：(1) 难度评估依赖**固定人工设计指标**（如预测不确定性、置信度），用一维标量来描述模型高维动态的认知状态；(2) 学习路径由**人工设计规则**驱动（如"从易到难"或"全部聚焦困难类"），无法适应模型不断变化的内部状态。这种"规定式范式"在面对恶劣天气这样的复合问题时会导致类别偏差——某些类别被过度关注而其他类别学习不足。

**核心矛盾**：模型训练过程是一个高维、动态、非单调的演化过程。试图用固定、一维、人工定义的标量来静态规划这个学习路径是本质上不合理的。CL的"从易到难"会忽略当前模型真正需要的类别，HCM的"全部困难类"则缺乏类间均衡。

**本文目标**：从"设计课程"转向"学习课程"——让Agent基于模型当前状态自主发现最优学习轨迹，而非依赖人类先验假设。

**切入角度**：受强化学习启发，将课程学习建模为序贯决策问题——Agent在每个训练步骤观察模型状态，自主决定哪些类别最具信息量，以这些类别为中心进行跨域混合采样。

**核心 idea**：提出自主课程调度器，包含高维状态编码器感知模型学习状态和类别公平策略梯度确保均衡改进，实现真正的自适应类别课程学习。

## 方法详解

### 整体框架

HeuSCM 以强化学习范式运行：(1) 从分割模型中提取高维语义状态 $\mathbf{s}$；(2) GM-VAE编码器将状态压缩到低维潜空间得到 $z_t^s$；(3) SKFEN从 $z_t^s$ 中蒸馏关键特征反映学习进度；(4) ClassGen（策略网络）根据关键特征输出排序的类别列表（按信息量降序）；(5) 排序结果指导跨域混合采样生成混合图像和标签；(6) 分割模型通过 SegLoss 在混合数据上更新；(7) 类别针对性的reward信号驱动策略优化。整个系统在训练过程中持续循环。

### 关键设计

1. **高维语义状态提取（High-dimensional Semantic State Extraction, HSSE）**:

    - 功能：全面刻画分割模型在域适应过程中的学习状态
    - 核心思路：分为两阶段——(a) 低维状态表征学习：使用高斯混合VAE（GM-VAE）将高维状态 $\mathbf{s}$ 编码到紧凑潜空间，建模多模态学习状态。离散分量 $q_\psi(c|\mathbf{s})$ 捕获学习模式，连续潜变量 $q_\psi(\mathbf{z}|\mathbf{s},c)$ 编码具体状态。训练目标为最大化变分下界：$\mathcal{L}_{GM-VAE} = \mathbb{E}_{q_\psi(c|\mathbf{s})}[\mathbb{E}_{q_\psi(\mathbf{z}|\mathbf{s},c)}[\log p_\theta(\mathbf{s}|\mathbf{z})] - \text{KL}(q_\psi(\mathbf{z}|\mathbf{s},c) \| p(\mathbf{z}|c))] - \text{KL}(q_\psi(c|\mathbf{s}) \| p(c))$。(b) SKFEN：通过分组特征聚合从潜空间中蒸馏关键特征，减少冗余信息
    - 设计动机：传统方法用预测不确定性一个标量来描述学习状态，无法捕获类间耦合关系、特征空间变化等高维信息。GM-VAE能捕获多模态结构，SKFEN进一步提炼与学习进度真正相关的信号

2. **语义关键特征提取网络（SKFEN）**:

    - 功能：从潜空间中蒸馏关键特征，减少特征冗余
    - 核心思路：包含初始变换和分组处理两阶段。首先通过1×1-5×5深度可分离-1×1卷积进行特征融合和交互建模，通道扩展后分为 $G$ 组。每组分别提取max pooling和average pooling特征，拼接后通过3×3卷积融合，加上残差连接得到精炼特征 $z_{out}$
    - 设计动机：潜空间虽然已降维，但仍存在大量对策略决策无用的冗余信息。SKFEN通过分组聚合从不同角度（峰值和均值统计）捕获真正反映学习状态的关键维度

3. **类别 $\alpha$-公平策略梯度（Categorical $\alpha$-Fairness for Policy Gradients, C$\alpha$PG）**:

    - 功能：确保策略优化在所有语义类别之间实现公平的reward分配
    - 核心思路：定义每个类别的价值函数 $V_c^\pi(s) = \mathbb{E}_\pi[\sum_{k=0}^{\infty}\gamma^k r_c(t+k)|S_t=s]$，其中reward $r_c(t)$ 综合考虑可迁移性（源域/目标域特征余弦相似度）和可区分性（目标域不同类之间的分离度）。关键创新是不优化标准的求和目标 $J_{sum}$，而是优化 $\alpha$-公平目标：$J_F(\pi) = \sum_{c=1}^{C} \frac{1}{1-\alpha}(V_c^\pi(s))^{1-\alpha}$。对应的策略梯度使用公平加权优势函数 $\tilde{A}_\alpha$，权重 $w_c(s_t) = V_c^\pi(s_t)^{-\alpha}$ 与类别当前价值成反比——低价值类别获得更高权重
    - 设计动机：标准RL优化总reward会导致策略偏向已表现良好的类别（"马太效应"），加剧类别不均衡。$\alpha$-公平机制迫使策略关注落后类别，实现均衡改进

### 损失函数 / 训练策略

- 分割损失：$\mathcal{L}_{seg} = \lambda_1 \mathcal{L}_{CE}(g_\theta(x_s), y_s) + \lambda_2 \mathcal{L}_{CE}(g_\theta(\mathcal{X}_{mix}^T), \mathcal{Y}_{mix}^R)$，其中 $\lambda_1 = \lambda_2 = 1.0$
- GM-VAE重建损失：$\mathcal{L}_{recon} = \mathbb{E}_{\mathbf{s}_t}[\|\mathbf{s}_t - p_\theta(\text{Enc}_\psi(\mathbf{s}_t))\|^2]$，用于微调时保持潜空间结构
- 策略优化：最大化公平目标 $J_F(\pi)$，使用 AdamW 优化器
- 在 Cityscapes → ACDC 上训练 60k 迭代，1024×1024 crop，NVIDIA A800 GPU

## 实验关键数据

### 主实验（Cityscapes → ACDC test）

| 方法 | Backbone | mIoU |
|------|----------|------|
| DeepLab-v2 (source only) | DeepLab-v2 | 38.0 |
| Refign | DeepLab-v2 | 48.0 |
| VBLC | DeepLab-v2 | 47.8 |
| **HeuSCM (Ours)** | DeepLab-v2 | **58.7** |
| HRDA (source only) | HRDA | 68.0 |
| CoDA | HRDA | 72.6 |
| ACSegFormer | HRDA | 72.7 |
| **HeuSCM (Ours)** | HRDA | **72.9** |

### 消融实验（ACDC val, HRDA backbone）

| 配置 | mIoU | 提升 | 说明 |
|------|------|------|------|
| Baseline (Refign) | 71.1 | +0.0 | 无HSCM |
| + LSRL only | 72.2 | +1.1 | 低维状态表征贡献最大 |
| + SKFEN only | 71.7 | +0.6 | 关键特征蒸馏有效 |
| + C$\alpha$PG only | 71.6 | +0.5 | 公平策略梯度有效 |
| + LSRL + SKFEN | 72.3 | +1.2 | 状态感知协同效果好 |
| + LSRL + C$\alpha$PG | 72.0 | +0.9 | 状态+公平组合有效 |
| + 全部三个（完整模型） | **72.7** | **+1.6** | 三个模块互补 |

### 关键发现

- 在 DeepLab-v2 backbone 上提升最为显著（+10.7 mIoU），说明HeuSCM对弱backbone的提升更大
- 在 HRDA 这种强backbone上仍然取得 72.9 mIoU 的SOTA，超越 CoDA（72.6）和 ACSegFormer（72.7）
- Dark Zurich 上达到 52.8 mIoU，Nighttime Driving 上达到 59.3 mIoU，证明在不同夜间场景下的泛化性
- HCSP（课程采样策略）可以作为即插即用模块替换现有硬类挖掘方法的采样策略，在 GTA5→Cityscapes 上也有效

## 亮点与洞察

- **范式转变的洞察非常锐利**：从"设计课程"到"学习课程"的理念shift，揭示了现有CL/HCM方法的根本局限——人类无法有效地为高维动态过程设计最优路径
- **$\alpha$-公平策略梯度设计精巧**：将多Agent公平性概念迁移到单Agent多类场景，通过反比权重机制自然地解决类别偏差问题
- **GM-VAE对学习状态的建模**：用混合高斯捕获多模态学习状态的思路值得借鉴——模型在适应不同天气条件时确实可能处于不同的"学习模式"

## 局限与展望

- 三阶段训练（GM-VAE预训练→联合微调→分割训练）增加了实现复杂度和训练开销
- SKFEN的具体架构设计（分组数G、扩展维度n等）可能需要针对不同场景调整
- 当前的reward设计假设源域和目标域特征可以在同一空间中比较，对于极端域差距场景可能不够鲁棒
- 未来可以扩展到其他域适应任务（如目标检测、实例分割）以验证框架的通用性

## 相关工作与启发

- **vs CoDA**: CoDA也做从易到难的域适应，但用的是人工设计的课程（先简单域再困难域），而HeuSCM完全自主学习课程顺序
- **vs ACSegFormer**: 达到相似性能但方法正交——ACSegFormer改进分割架构，HeuSCM改进训练策略，理论上可以组合使用
- **vs CoPT**: CoPT用prompt tuning做条件适应，HeuSCM用RL做课程学习，两者关注不同层面的适应机制
- 将RL范式引入训练策略优化（而非模型本身）的思路，可迁移到active learning、数据选择等相关问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "学习课程"的范式转变、GM-VAE状态编码和$\alpha$-公平策略梯度都是创新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 三个backbone、三个基准、详细消融、泛化验证，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机阐述非常清晰，方法部分公式较密集需要反复阅读
- 价值: ⭐⭐⭐⭐ 对域适应中的课程学习提出了优雅的新范式，但具体实现复杂度较高

<!-- RELATED:START -->

## 相关论文

- [Masked Representation Modeling for Domain-Adaptive Segmentation](mrm_masked_representation_modeling_domain_adaptive.md)
- [FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](../../ECCV2024/segmentation/frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)

<!-- RELATED:END -->
