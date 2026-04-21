---
title: >-
  [论文解读] Large-Scale Training Data Attribution for Music Generative Models via Unlearning
description: >-
  [NeurIPS 2025][图像生成][训练数据归因] 将基于机器遗忘（machine unlearning）的训练数据归因方法应用于大规模文本到音乐扩散模型（115K 音轨），通过网格搜索找到最优超参数配置，并与非反事实方法对比，验证了 unlearning-based TDA 在音乐生成领域的可行性。
tags:
  - NeurIPS 2025
  - 图像生成
  - 训练数据归因
  - 机器遗忘
  - 音乐生成
  - 扩散模型
  - 版权归因
---

# Large-Scale Training Data Attribution for Music Generative Models via Unlearning

**会议**: NeurIPS 2025  
**arXiv**: [2506.18312](https://arxiv.org/abs/2506.18312)  
**代码**: 无  
**领域**: 音乐生成 / AI 伦理  
**关键词**: 训练数据归因, 机器遗忘, 音乐生成, 扩散模型, 版权归因

## 一句话总结

将基于机器遗忘（machine unlearning）的训练数据归因方法应用于大规模文本到音乐扩散模型（115K 音轨），通过网格搜索找到最优超参数配置，并与非反事实方法对比，验证了 unlearning-based TDA 在音乐生成领域的可行性。

## 研究背景与动机

生成式 AI 在音乐领域的快速发展引发了严峻的版权和归因问题：模型可能无意中复制受版权保护的素材，而对原始创作者的贡献缺乏合理的认定和回馈。**训练数据归因**（Training Data Attribution, TDA）旨在识别哪些训练数据对模型的特定输出贡献最大，从而支持更公平的艺术贡献认定。

TDA 方法分两类：

**黑盒方法**（模型不可访问）：基于相似度的协同归因，使用外部编码器计算生成输出与训练数据的相似度（如 CLAP、CLEWS）。简单实用但依赖编码器视角，不一定反映生成模型内部行为。

**白盒方法**（可访问模型参数）：基于反事实推理，"如果移除某训练样本，模型预测会如何变化？"最直接的方法是 leave-one-out retraining（计算不可行），influence function 可近似变化但在大模型上也有局限。

**机器遗忘**作为新方向出现：通过梯度上升最大化特定训练样本的损失来"遗忘"该样本，配合 Fisher Information Matrix (FIM) 正则化防止灾难性遗忘。此前 unlearning-based TDA 仅在其他领域探索过，在音乐生成领域尚属空白。

已有音乐 TDA 工作（Deng et al.）仅在 Music Transformer + MAESTRO 数据集（~200小时钢琴）上用 influence function 方法验证，本文首次将 TDA 扩展到大规模文本到音乐 DiT 模型（115K 音轨，~4356小时多风格音乐）。

## 方法详解

### 整体框架

归因流程：给定生成样本 $\hat{\mathbf{z}}$ 和训练样本 $\mathbf{z}_i$，归因分数定义为遗忘前后损失差：

$$\tau(\hat{\mathbf{z}}, \mathbf{z}_i) = \mathcal{L}(\mathbf{z}_i, \theta_{\setminus \hat{\mathbf{z}}}) - \mathcal{L}(\mathbf{z}_i, \theta_0)$$

利用 **mirrored influence hypothesis**：不是逐个遗忘训练样本（需 N 次），而是遗忘生成样本 $\hat{\mathbf{z}}$，观察对各训练样本损失的影响。每个目标样本只需一次遗忘操作。

### 关键设计

#### 1. Unlearning 算法

直接最大化目标样本损失会导致灾难性遗忘，因此结合 FIM 正则化：

$$\mathcal{L}_{\text{unlearn}}^{\hat{\mathbf{z}}}(\theta) = -\mathcal{L}(\hat{\mathbf{z}}, \theta) + \frac{N}{2}(\theta - \theta_0)^\top \mathbf{F} (\theta - \theta_0)$$

第一项通过梯度上升遗忘目标样本，第二项用 FIM 加权的二次惩罚保持模型整体性能。FIM 量化每个参数对模型输出的影响程度，对影响大的参数施加更强约束。

推导得到更新规则：$\theta = \theta_0 + \frac{1}{N} \mathbf{F}^{-1} \nabla \mathcal{L}(\hat{\mathbf{z}}, \theta)$

#### 2. Fisher Information Matrix 计算

FIM 的对角近似用于降低计算成本：

$$({\mathbf{F}_{\text{diag}}})_{jj} \approx \frac{1}{N} \sum_{i=1}^N \frac{1}{T} \sum_{t=1}^T \left(\frac{\partial \mathcal{L}_t(\mathbf{z}_i, \theta)}{\partial \theta_j}\right)^2$$

在扩散模型中，损失依赖去噪时间步 $t$，因此对多个时间步取平均。

#### 3. 静音掩码策略（Masking Silence）

音乐生成模型处理变长音频时会用零填充短片段。提出三种掩码方案：
- **无掩码**：遗忘和归因计算都不掩码 → 短音轨被零填充干扰，排名不准
- **双掩码**（$M_U + M_L$）：都掩码 → 归因排名好但极短音轨异常高
- **混合策略**（$M_U$ only）：遗忘时掩码、计算损失时不掩码 → 最佳

**设计动机**：遗忘时掩码确保零填充区域不干扰遗忘过程；计算损失时不掩码以保持与训练设定一致，避免模型行为不可预测。

### 损失函数 / 训练策略

- 模型：Latent DiT (基于 Stable Audio)，VAE 将 44.1kHz 立体声编码到 64 维潜空间
- 扩散过程：v-objective，最大处理约 2 分钟音频（2584 latent frames）
- 条件：CLAP embedding（文本到音乐）+ timing conditions（变长生成）
- FIM 计算：每个遗忘步在 2048 个随机时间步上平均梯度
- 单步遗忘耗时约 20 分钟（NVIDIA H100），全训练集损失计算约 5 小时（8 × H100）

## 实验关键数据

### 主实验：自影响实验（Train-to-Train）

用 k-means 从 CLAP 嵌入中选取 40 个多样化训练样本进行网格搜索：

| 目标层 | $M_U$ | $M_L$ | $R(\mathbf{z}_{tar})$ | $\text{CLAP}_{topk}$ | $\text{CLAP}_{botk}$ | $\text{FD}_{openl3}$ |
|--------|-------|-------|-----|------|------|------|
| Cross-Attention to_kv | ✓ | - | 103.2 | 0.38 | 0.35 | 110.5 |
| Cross-Attention Layers | ✓ | - | 1.4 | 0.60 | 0.32 | 110.4 |
| Self-Attention Layers | ✓ | - | 1.1 | 0.63 | 0.30 | 110.5 |
| **All Transformer Layers** | **✓** | **✓** | **1.0** | **0.80** | **0.38** | **110.5** |
| All Transformer Layers | - | - | 6615.7 | 0.82 | 0.42 | 110.5 |
| **All Transformer Layers (Mixed)** | **✓** | **-** | **1.0** | **0.66** | **0.26** | **110.5** |

- 学习率 $10^{-6}$、1 步更新为最优组合
- $R(\mathbf{z}_{tar}) = 1.0$ 表示目标样本在归因排名中排第一（遗忘成功）
- $\text{FD}_{openl3}$ 未变化，说明遗忘不影响整体生成质量

### 对比实验：与非反事实方法对比（Test-to-Train）

生成 16 个两分钟音轨，对比五种归因方法：

| 方法 | 类型 | 与 Unlearning 的 Pearson 相关系数 |
|------|------|------|
| LPIPS | 白盒（模型内部激活相似度） | **0.56** |
| CLAP | 黑盒（音频嵌入相似度） | 0.46 |
| CLEWS | 黑盒（音乐身份嵌入） | 0.32 |
| RPS (Representer Point) | 白盒（梯度信息） | 0.11 |

### 关键发现

1. **Unlearning 归因集中度更高**：归因分数分布呈尖锐集中模式，影响力集中在极少数训练样本上
2. **方法间排序一致性**：与 LPIPS（同为白盒方法且利用模型内部信息）相关性最高，验证了内部表征的一致性
3. **模型信息 vs 外部信息**：利用模型内部信息的方法（Unlearning、LPIPS）互相关性高，外部嵌入方法（CLAP、CLEWS）互相关性也高，两组之间相关性中等
4. **RPS 捕获不同模式**：RPS 与所有方法相关性都低，说明其捕获的归因模式独特
5. **模型整体性能不受影响**：遗忘后 $\text{FD}_{openl3}$ 保持 110.5（原始值），验证了正则化有效

## 亮点与洞察

- **领域首创**：首次在大规模文本到音乐 DiT 上探索 unlearning-based TDA，面对的是真实规模（115K 音轨，4356 小时）和多样风格的挑战
- **混合掩码策略**巧妙解决了变长音频处理中的零填充干扰问题——遗忘时排除无关静音，评估时保持训练一致性
- **实验设计严谨**：先通过自影响实验验证方法有效性（是否能正确识别目标样本），再用于实际归因分析
- **对 AI 伦理的贡献**：为音乐 AI 领域的版权归因和创作者回馈提供了技术基础

## 局限与展望

1. **计算成本高**：每次遗忘需 ~20 min（H100），全数据集损失计算需 ~5h（8×H100），大规模部署困难
2. **仅单步遗忘**：网格搜索发现 1 步最优但理论上多步可能更精确，需更深入探索
3. **FIM 为对角近似**：丢失了参数间相关性信息，可能影响归因精度
4. **验证数据为私有**：115K 数据集不公开，难以完全复现
5. **缺少人工评估**：归因结果的音乐相关性未经专业音乐人评估
6. **仅测试单一模型架构**：未验证方法在自回归模型或其他音乐生成架构上的效果

## 相关工作与启发

- **与 influence function 方法的互补**：Deng et al. 用 influence function 在小规模 Piano 数据上做归因，本文将 unlearning 方法扩展到大规模多风格场景，两类方法可互相验证
- **通用 TDA 方法的领域迁移**：Wang et al. 的 FIM 正则化遗忘方法原用于图像分类，本文成功适配到扩散模型的音频生成
- **对音乐版权体系的潜在影响**：如果 TDA 足够准确且计算成本降低，可实现自动化的版税分配系统

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将 unlearning TDA 用于大规模音乐生成，混合掩码策略新颖
- 实验充分度: ⭐⭐⭐ — 自影响验证严谨但测试规模有限（40 + 16 样本），缺少人工评估
- 写作质量: ⭐⭐⭐⭐ — 方法推导清晰，实验设计合理，图表信息量大
- 价值: ⭐⭐⭐⭐ — 为音乐 AI 伦理和版权归因开拓了重要方向

<!-- RELATED:START -->

## 相关论文

- [Fast Data Attribution for Text-to-Image Models](fast_data_attribution_for_text-to-image_models.md)
- [Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](../../CVPR2025/image_generation/training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)

<!-- RELATED:END -->
