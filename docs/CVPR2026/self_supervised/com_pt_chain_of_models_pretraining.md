---
title: >-
  [论文解读] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models
description: >-
  [CVPR 2026][自监督学习][模型链] 提出 Chain-of-Models Pre-Training (CoM-PT)，将视觉基础模型按大小排列形成"模型链"，通过从小到大的逆向知识转移（权重初始化+特征蒸馏）逐步加速训练，实现性能无损的训练加速且效率随模型家族规模增长而提升。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "模型链"
  - "预训练加速"
  - "逆向知识转移"
  - "CLIP"
  - "视觉基础模型"
---

# Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models

**会议**: CVPR 2026  
**arXiv**: [2604.12391](https://arxiv.org/abs/2604.12391)  
**代码**: [https://github.com/deep-optimization/CoM-PT](https://github.com/deep-optimization/CoM-PT)  
**领域**: 自监督学习 / 训练加速  
**关键词**: 模型链, 预训练加速, 逆向知识转移, CLIP, 视觉基础模型

## 一句话总结

提出 Chain-of-Models Pre-Training (CoM-PT)，将视觉基础模型按大小排列形成"模型链"，通过从小到大的逆向知识转移（权重初始化+特征蒸馏）逐步加速训练，实现性能无损的训练加速且效率随模型家族规模增长而提升。

## 研究背景与动机

**领域现状**：视觉基础模型（VFM）的预训练代价极其高昂（如 ViT-L/14 在 LAION-2B 上需 1.2×10⁵ A100 GPU 小时），现有加速方法（混合精度、掩码建模、数据高效方法等）都是在单模型维度优化。

**现有痛点**：VFM 通常以模型家族形式预训练（不同大小满足不同部署场景），但标准的独立训练方式高度冗余——模型共享相同的优化目标、数据集和训练协议，产生的共同知识被反复学习。

**核心矛盾**：模型家族规模不断增长（更多专用模型尺寸 + 更大模型范围），独立训练的总成本线性增长，产生"承担不断升级的预训练成本"与"牺牲部署灵活性"的困境。

**本文目标**：实现随模型家族规模高效扩展的预训练加速。

**切入角度**：从微观看，大模型的训练成本是主要来源；从宏观看，独立训练的冗余是低效根源。同时解决这两个瓶颈的关键是实现家族内小到大的知识复用。

**核心 idea**：将模型家族按大小排序形成模型链，最小模型标准训练，后续模型通过逆向知识转移（小→大）加速预训练。

## 方法详解

### 整体框架

模型链 $C_M: m_1 \rightarrow m_2 \rightarrow \cdots \rightarrow m_n$ 按模型大小升序排列。$m_1$ 标准独立预训练，每个后续模型 $m_{i+1}$ 通过从 $m_i$ 的逆向知识转移加速训练。逆向知识转移包含两个组件：参数空间的权重初始化和特征空间的特征蒸馏。

### 关键设计

1. **逆向权重初始化**:

    - 功能：在参数空间复用小模型的知识来初始化大模型
    - 核心思路：(i) 宽度扩展：直接将小教师的参数嵌入大学生对应位置，剩余参数随机初始化；(ii) 深度扩展：复制每层权重作为后继层。简单直接的函数保持初始化
    - 设计动机：利用已训练小模型的知识提供更好的起点，加速大模型收敛

2. **逆向特征蒸馏**:

    - 功能：在特征空间复用小模型的动态知识
    - 核心思路：$\mathcal{L}_{IFD}(F^t, F^s) = \alpha \| F^t - \mathbf{T}(F^s) \|_2^2$，通过特征变换 $\mathbf{T}(\cdot)$ 将学生特征投影到教师特征空间。在 CLIP 中同时对视觉和文本特征蒸馏：$\hat{\mathcal{L}}_{IFD} = (\mathcal{L}_{IFD}(v^t,v^s) + \mathcal{L}_{IFD}(t^t,t^s))/2$
    - 设计动机：权重初始化是静态知识，特征蒸馏捕获跨样本的动态知识，两者协同确保有效的知识转移接力

3. **模型链设计三原则**:

    - 功能：指导构建最优模型链
    - 核心思路：(i) 最优最小模型：根据数据规模选择，足够小以最大化效率但有足够容量拟合数据分布；(ii) 中间模型变体：使用 2×-4× 的扩展比率，大因子优化成本，小因子最大化加速比；(iii) 训练 epoch 分配：沿模型链线性递减
    - 设计动机：出现反直觉现象——ViT-T→S→B→L 链比 ViT-B→L 链多训练两个模型反而总成本更低 20%

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{task} + \hat{\mathcal{L}}_{IFD}$，其中任务损失为 LaCLIP 的对比损失（含文本增强）。确保 $\mathcal{L}_{IFD} < \mathcal{L}_{task}$。

## 实验关键数据

### 主实验

| 模型链 | ImageNet Top-1 | 训练MACs | 加速比 |
|--------|---------------|---------|--------|
| ViT-L 独立训练 | 38.2% | 100% | 1.0× |
| ViT-B→L | 38.0% | 48% | 2.1× |
| ViT-S→B→L | 38.1% | 36% | 2.8× |
| ViT-T→S→B→L | **38.3%** | **28%** | **3.6×** |

### 消融实验

| 配置 | ImageNet Top-1 | 说明 |
|------|---------------|------|
| 完整 CoM-PT | 38.3% | 权重初始化+特征蒸馏 |
| 仅权重初始化 | 37.8% | 无蒸馏 |
| 仅特征蒸馏 | 37.5% | 随机初始化 |
| 独立训练 | 38.2% | 基线 |

### 关键发现

- 反直觉现象：训练更多模型反而更高效——3→4→7个模型时加速比从 4.13× 跃升到 5.68× 和 7.09×
- 模型链本身驱动主要效率增益，权重初始化和蒸馏各自贡献较小但协同效果好
- 在 45 个下游数据集上验证了性能无损（<0.5% 精度损失）

## 亮点与洞察

- "训练更多模型反而更高效"是一个极具洞察力的发现：因为扩展链中的中间模型借助前驱快速收敛，总开销甚至小于直接训练大模型
- 方法对预训练范式不可知，可推广到 LLM 预训练等更计算密集的场景
- 逆向知识转移（小→大）与传统知识蒸馏（大→小）形成对偶，思路新颖

## 局限与展望

- 主要在 CLIP 上验证，尚未在 LLM 预训练上规模化测试
- 模型链的设计仍需人工调整，缺乏自动化方法
- 宽度和深度扩展使用简单的复制/插入策略，可能有更优方案
- 跨架构的模型链（如 ViT→Swin）尚未探索

## 相关工作与启发

- **vs Net2Net**: Net2Net 首先提出函数保持变换用于模型扩展，CoM-PT 将其扩展为系统性的训练管线
- **vs FLIP/DeCLIP**: 这些方法在单模型维度加速，CoM-PT 在模型家族维度加速，正交互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 模型家族级训练加速是全新视角
- 实验充分度: ⭐⭐⭐⭐⭐ 45个下游数据集的全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 微观/宏观视角分析透彻
- 价值: ⭐⭐⭐⭐⭐ 对大规模预训练有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[ICML 2026\] NITP: Next Implicit Token Prediction for LLM Pre-training](../../ICML2026/self_supervised/nitp_next_implicit_token_prediction_for_llm_pre-training.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[ECCV 2024\] Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](../../ECCV2024/self_supervised/efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)

</div>

<!-- RELATED:END -->
