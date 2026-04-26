---
title: >-
  [论文解读] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning
description: >-
  [NeurIPS 2025][模型压缩][增量学习] 提出学习有益的"混合噪声"来抑制预训练模型在增量学习中的参数漂移，通过在任务间进行动态权重混合噪声实现 SOTA 性能，特别在 50 步增量设置下表现突出。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 增量学习
  - 预训练模型
  - 参数漂移
  - 正向噪声
  - 灾难遗忘抑制
---

# Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.16738](https://arxiv.org/abs/2509.16738)  
**代码**: https://github.com/ASCIIJK/MiN-NeurIPS2025  
**领域**: 模型压缩 / 增量学习  
**关键词**: 增量学习, 预训练模型, 参数漂移, 正向噪声, 灾难遗忘抑制

## 一句话总结

提出学习有益的"混合噪声"来抑制预训练模型在增量学习中的参数漂移，通过在任务间进行动态权重混合噪声实现 SOTA 性能，特别在 50 步增量设置下表现突出。

## 研究背景与动机

**领域现状**：预训练模型（PTM）在下游任务微调时虽然表现强大，但持续微调会产生**参数漂移**，导致之前任务的判别特征被破坏、新任务的特征会干扰现有类别的决策边界。

**现有痛点**：传统增量学习方法关注提高特征利用效率（如提示学习、原型网络），但忽视了任务间的特征干扰问题。参数漂移被视为纯粹的负面现象。

**核心洞察**：噪声不总是有害的——正向激励噪声（Pi-Noise）可以通过掩盖类间混淆、突出识别特征的方式改善分类。参数漂移即"破坏性噪声"，但可以学习"有益噪声"来对抗。

**本文目标**：主动学习有益噪声来抑制任务间的混淆模式，而非追求高效特征利用。

**切入角度**：从信息论出发，将噪声建模为隐变量，通过重参数化技巧和动态混合实现高效推理。

**核心 idea**：通过噪声扩展（每任务学习专用噪声）+ 噪声混合（动态权重融合），将参数漂移从"灾难性遗忘"转化为"可控的正向信号"。

## 方法详解

### 整体框架

MiN（Mixture of Noise）方法通过两个核心策略实现：
- **噪声扩展（Noise Expansion）**：为每个新任务学习专用的噪声生成模块
- **噪声混合（Noise Mixture）**：动态学习权重混合来自不同任务的噪声，实现单次推理

### 关键设计

1. **噪声扩展策略（Section 4.1）**

    - 功能：在预训练骨干网络的中间层插入 π-噪声层
    - 核心思路：给定中间特征 $r_l$，噪声生成过程为 $\varepsilon_t = \varepsilon \cdot \phi_t^{\sigma}(r_l W_{down}) + \phi_t^{\mu}(r_l W_{down})$，其中 $\varepsilon$ 从标准正态分布采样，$\phi_t^{\sigma}$、$\phi_t^{\mu}$ 是两层 MLP，分别生成方差和均值向量，$W_{down}$ 将高维特征投影到低维 $d_2 \ll d_1$
    - 设计动机：训练参数极少（仅 $d_2 \times d_2$ 两个矩阵），保持模型轻量

2. **噪声混合策略（Section 4.2）**

    - 功能：对于任务 t，累积了多个噪声集合 $\{\varepsilon_1, \ldots, \varepsilon_t\}$，学习动态权重进行最优混合
    - 核心思路：权重初始化基于任务原型相似度 $s_{t,i} = \frac{\mu_t \cdot \mu_i}{\|\mu_t\| \|\mu_i\|}$，归一化后加权混合 $\varphi(\{\varepsilon_1, \ldots, \varepsilon_t\}) = \sum_{i=1}^{t}\varepsilon_i \omega_i$
    - 设计动机：若单独使用每个噪声，推理复杂度随任务线性增长；混合后只需一次前向推理

3. **三步训练管道（Algorithm 1）**

    - 功能：三步迭代更新，解析学习 → 噪声学习 → 分类器更新
    - 核心思路：第一步用解析学习更新分类器 $W_t$（无梯度），第二步用辅助分类器学习任务相关噪声，第三步再次更新分类器
    - 设计动机：辅助分类器 $W_{aux}$ 初始化为零拟合残差，避免主分类器被噪声训练干扰

### 损失函数 / 训练策略

$$\mathcal{L}_{cls} = \ell(z_L W_{aux}, y - z_L W_t)$$

辅助分类器学习残差，噪声最终作用于特征空间的扰动。

## 实验关键数据

### 主实验

| 方法 | CIFAR-100 10步 | CIFAR-100 20步 | CIFAR-100 50步 | CUB-200 10步 | CUB-200 20步 |
|------|-------------|-------------|-------------|-----------|-----------|
| L2P | 85.92 | 81.90 | 74.29 | 84.29 | 81.75 |
| DualPrompt | 89.65 | 85.57 | 73.66 | 84.39 | 83.79 |
| CODA-Prompt | 91.05 | 87.51 | 69.54 | 84.15 | 83.89 |
| SLCA | 92.67 | 93.32 | 90.76 | 86.83 | 83.38 |
| FeCAM | 93.23 | 91.86 | 90.92 | 92.73 | 92.89 |
| **MiN (本文)** | **94.12** | **93.89** | **92.34** | **93.45** | **93.67** |

### 消融实验

| 配置 | CIFAR-100 20步 | 说明 |
|------|-------------|------|
| 仅 Noise Expansion | 90.56 | 基础噪声学习 |
| + 初始权重 | 91.23 | 加入相似度初始化 |
| + 权重学习 | 93.12 | 动态权重优化 |
| + 三步流程 | **93.89** | 完整管道 |

### 关键发现

- 在 50 步增量设置（最极端场景）上相比先前 SOTA 提升显著，从 90.92 → 92.34，且标准差最小
- 即使在 50 步设置下，训练参数总量极少（每任务最少）
- Grad-CAM 可视化显示 MiN 显著提高了对关键区域的关注，抑制了无关区域的响应

## 亮点与洞察

- **全新视角**：将参数漂移重新诠释为可控的噪声信号，从负面现象中挖掘积极机制。这是一个根本性的思维转换，不是"避免漂移"而是"利用漂移"。
- **轻量级设计**：噪声生成模块的参数量极小，新增参数远少于提示学习方法，且推理时混合噪声不增加额外开销。
- **极端场景鲁棒**：50 步增量学习（每步仅 2 个新类别，共 1000+ 步）下仍保持稳定，远超其他方法在此设置下的表现。

## 局限与展望

- 实验基于 CIFAR-100（10K 样本）、CUB-200（12K 样本），缺少大规模现实数据验证
- 仅在 ViT-B/16-IN21K 上验证，未探索其他骨干（如 Swin、ConvNeXt）的效果
- 温度系数和初始化权重计算尚有调优空间
- 是否适用于非视觉域（NLP、多模态增量学习）还需验证

## 相关工作与启发

- **vs L2P/DualPrompt**：它们用提示学习适配新任务，但提示在任务间不共享知识；MiN 的噪声混合天然跨任务共享
- **vs FeCAM**：FeCAM 依赖原型和特征空间对齐，MiN 在特征空间注入正向噪声，两者互补
- **可迁移方向**：知识蒸馏、对比学习中的噪声增强应用；跨域迁移学习中的噪声特征建模

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 参数漂移-有益噪声框架是全新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个数据集，充分的消融实验
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，方法表述精确
- 价值: ⭐⭐⭐⭐⭐ 指导预训练模型的有效适应，实际部署价值高

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[NeurIPS 2025\] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)
- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](../../CVPR2025/model_compression/adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)

<!-- RELATED:END -->
