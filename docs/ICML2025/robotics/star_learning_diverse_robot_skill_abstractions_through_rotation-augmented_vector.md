---
title: >-
  [论文解读] STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization
description: >-
  [ICML 2025][机器人][技能抽象] 提出STAR框架，通过旋转增强残差技能量化（RaRSQ）解决VQ-VAE的codebook坍塌问题，并通过因果技能Transformer（CST）建模技能间依赖关系，在LIBERO基准上整体成功率达93.6%，比此前SOTA QueST提升约12%。
tags:
  - "ICML 2025"
  - "机器人"
  - "技能抽象"
  - "向量量化"
  - "codebook collapse"
  - "自回归技能合成"
  - "机器人操作"
---

# STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization

**会议**: ICML 2025 Spotlight  
**arXiv**: [2506.03863](https://arxiv.org/abs/2506.03863)  
**代码**: [有](https://STAR.github.io)  
**领域**: 机器人  
**关键词**: 技能抽象, 向量量化, codebook坍塌, 残差量化, 模仿学习

## 一句话总结

提出STAR框架，通过旋转增强残差技能量化（RaRSQ）解决VQ-VAE的codebook坍塌问题，并通过因果技能Transformer（CST）建模技能间依赖关系，在LIBERO基准上整体成功率达93.6%，比此前SOTA QueST提升约12%。

## 研究背景与动机

多任务视觉运动策略学习一直是机器人操作领域的核心挑战。单个操作任务已有多模态动作分布等难题，多任务场景下动作空间高度纠缠，不同任务的特征相互交叉重叠。

**直觉方案**：将复杂动作分解为可复用的技能抽象，形成层次化框架。近期研究使用VQ-VAE等潜变量模型将连续动作空间离散化为技能表示。

**两个关键瓶颈**：

**Codebook坍塌**：VQ-VAE训练中大部分codebook向量不被使用，只有少数被频繁选择。根因在于直通梯度估计器（STE）对同一codebook向量对应的所有编码器输出施加相同梯度，忽略了嵌入间的几何关系。

**技能组合困难**：现有方法未建模不同技能之间的依赖关系，导致长horizon任务中动作序列缺乏时序一致性。

**核心idea**：编码动作序列间的几何关系到残差量化过程中是学习多样可复用技能的关键，用旋转矩阵替代STE的"一刀切"梯度传播。

## 方法详解

### 整体框架

STAR采用**两阶段训练**：阶段1训练RaRSQ从专家演示中学习离散技能表示；阶段2固定RaRSQ，训练CST根据观测预测技能序列。

### 关键设计

#### 1. 旋转增强残差技能量化（RaRSQ）

**功能**：将连续动作序列编码为层次化离散技能，同时防止codebook坍塌。

**核心思路**：在残差量化每一层，用旋转矩阵替代STE传播梯度。编码器输出 $\mathbf{z} = \phi(\mathbf{a}_{t:t+T})$，初始残差 $\mathbf{r}_0 = \mathbf{z}$，对每层 $d$：

$$k_d = \arg\min_k \|\mathbf{r}_{d-1} - \mathbf{e}_{(d,k)}\|_2^2$$

$$\tilde{\mathbf{q}}_d = \text{sg}\left[\frac{\|\mathbf{e}_{(d,k_d)}\|}{\|\mathbf{r}_{d-1}\|}\mathbf{R}_d\right]\mathbf{r}_{d-1}, \quad \mathbf{r}_d = \mathbf{r}_{d-1} - \tilde{\mathbf{q}}_d$$

旋转矩阵 $\mathbf{R}_d = \mathbf{I} - 2\hat{\mathbf{r}}_d\hat{\mathbf{r}}_d^T + 2\hat{\mathbf{q}}_d\hat{\mathbf{r}}_{d-1}^T$ 保留了嵌入间的角度信息。反向传播时 $\frac{\partial\hat{\mathbf{z}}}{\partial\mathbf{r}_{d-1}} = \frac{\|\mathbf{e}_{(d,k_d)}\|}{\|\mathbf{r}_{d-1}\|}\mathbf{R}_d$，使不同位置嵌入根据与codebook向量的相对角度获得差异化梯度。

**设计动机**：STE对同一partition内所有点施加相同梯度导致嵌入趋同坍塌。旋转机制保留角度信息——方向相近的被推远，方向不同的被拉近，维持codebook多样性。残差结构  $K^D$ 种组合使表征容量指数级增长。

#### 2. 因果技能Transformer（CST）

**功能**：给定多模态观测，自回归预测技能码序列并生成连续动作。

**核心思路**：层次化条件概率建模技能选择：

$$P(k_1,\ldots,k_D|\mathbf{o}_{t-h:t},\boldsymbol{\tau}) = \prod_{d=1}^D P(k_d|k_{<d},\mathbf{g}_t)$$

**动作精炼**：借鉴BeT引入offset预测头弥补离散化精度损失：$\hat{\mathbf{a}}_t = \psi(\sum_d \mathbf{R}_d\mathbf{e}_{d,k_d}) + \zeta_{\text{ref}}(\mathbf{g}_t)$

**设计动机**：残差量化的粗→细层次天然形成技能依赖结构（粗粒度运动基元→精细调整），自回归机制精确契合。

#### 3. 推理过程

Nucleus sampling配合温度 $\tau$ 和阈值 $p$ 采样技能码，通过解码器+offset生成动作序列，滚动式重新规划。

### 损失函数 / 训练策略

**阶段1**：$\mathcal{L} = \|\mathbf{a} - \psi(\hat{\mathbf{z}})\|_2^2 + \beta\sum_d\|\text{sg}[\mathbf{r}_{d-1}] - \tilde{\mathbf{q}}_d\|_2^2$（重建 + commitment）

**阶段2**：$\mathcal{L} = -\sum_d\log P(k_d^*|k_{<d},\mathbf{g}_t) + \lambda\|\mathbf{a}_t - \hat{\mathbf{a}}_t\|^2$（技能预测 + 动作精炼）

## 实验关键数据

### 主实验（LIBERO基准，成功率%）

| 方法 | Object | Spatial | Goal | Long | 90 | Overall |
|------|--------|---------|------|------|----|---------|
| OpenVLA | 88.4 | 84.7 | 79.2 | 53.7 | - | 76.5 |
| VQ-BeT | 90.3 | 88.7 | 61.3 | 59.7 | 84.2 | 76.8 |
| QueST | 90.0 | 84.5 | 76.7 | 69.1 | 87.4 | 81.5 |
| **STAR** | **98.3** | **95.5** | **95.0** | **88.5** | **90.8** | **93.6** |

MetaWorld MT50: STAR达92.7%，超越所有基线2.1%-5.4%。

### 消融实验

| 配置 | Object | Long | Overall | 说明 |
|------|--------|------|---------|------|
| STAR完整 | 98.3 | 88.5 | **93.6** | - |
| w/o AR | 95.3 | 83.3 | 89.5 | 去掉自回归，-4.1% |
| w/o Rotation | 93.7 | 85.7 | 91.0 | 去掉旋转，-2.6% |
| w/o Both | 93.3 | 81.5 | 87.8 | 两者都去，-5.8%（协同效应） |

### 关键发现

1. **Codebook利用率**：RaRSQ 16/16码字全部使用（100%），VQ-VAE仅7/16（43.8%），频率均值6.25% vs 14.29%
2. **复杂任务提升最大**：LIBERO-Long +19.4%，Goal +18.3%——codebook坍塌影响最大的场景
3. **真实机器人**：抽屉操作30%完成率（VQ-BeT 10%，QueST 0%），物体放置60%（VQ-BeT 30%，QueST 40%）
4. 旋转和自回归有协同效应：同时去除(-5.8%)比单独去除之和(-6.7%和-2.6%)更大

## 亮点与洞察

- 从梯度传播的几何角度理解codebook坍塌，切入点精准——STE的"一刀切"梯度是根因
- 旋转增强是轻量但高效的技巧，仅修改梯度传播方式，不增加推理计算量
- 两阶段设计清晰地分离了"技能学习"和"技能组合"
- 残差量化+自回归的组合自然形成粗到细的技能层次，与操作任务结构契合

## 局限与展望

- codebook大小 $K$ 和量化深度 $D$ 需手动调优，缺乏自适应机制
- 作为模仿学习方法，强依赖专家演示数据的质量和覆盖度
- 真实机器人实验规模较小（仅2个任务，每个10次尝试），整体成功率仍有提升空间
- 未与Diffusion Policy等最新生成式方法在真实场景中对比
- 动作精炼机制继承自BeT，可能不是最优设计选择

## 相关工作与启发

- 旋转技巧从图像生成领域迁移到机器人技能学习，展示了跨领域方法迁移的成功
- 技能的层次化分解（粗到细）与人类动作规划的层次结构很契合
- codebook collapse的解决方案对所有VQ-VAE下游任务（语音、图像、视频压缩等）都有借鉴意义
- 推理阶段的nucleus sampling为动作生成引入可控多样性，有利于多模态动作分布建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 旋转机制在机器人技能量化中的应用新颖，但旋转技巧本身已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ LIBERO全5个子集+MetaWorld MT50+真实世界+全面消融+codebook分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，技术阐述严谨，图表直观
- 价值: ⭐⭐⭐⭐ 93.6%的LIBERO成绩是实质性进步，解决了实际瓶颈问题: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization

**会议**: ICML 2025  
**arXiv**: [2506.03863](https://arxiv.org/abs/2506.03863)  
**代码**: [https://STAR.github.io](https://STAR.github.io)  
**领域**: 机器人  
**关键词**: 技能抽象, 向量量化, codebook collapse, 自回归技能合成, 机器人操作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] CommVQ: Commutative Vector Quantization for KV Cache Compression](commvq_commutative_vector_quantization_for_kv_cache_compression.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[ICCV 2025\] Certifiably Optimal Anisotropic Rotation Averaging](../../ICCV2025/robotics/certifiably_optimal_anisotropic_rotation_averaging.md)
- [\[ICML 2025\] Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism](robot-gated_interactive_imitation_learning_with_adaptive_intervention_mechanism.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](../../NeurIPS2025/robotics/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)

</div>

<!-- RELATED:END -->
