---
title: >-
  [论文解读] UniRes: Universal Image Restoration for Complex Degradations
description: >-
  [ICCV 2025][图像恢复][复杂退化] 提出 UniRes——一个基于扩散模型的通用图像复原框架，通过多任务训练学习超分辨率、运动去模糊、散焦去模糊和去噪等专家知识，推理时通过灵活组合不同任务的隐空间预测权重来端到端地处理真实世界中的任意复杂退化组合。
tags:
  - ICCV 2025
  - 图像恢复
  - 复杂退化
  - 扩散模型
  - 多任务训练
  - 隐空间组合
  - 真实世界复原
---

# UniRes: Universal Image Restoration for Complex Degradations

**会议**: ICCV 2025  
**arXiv**: [2506.05599](https://arxiv.org/abs/2506.05599)  
**代码**: 无  
**领域**: 图像复原  
**关键词**: 复杂退化, 扩散模型, 多任务训练, 隐空间组合, 真实世界复原

## 一句话总结

提出 UniRes——一个基于扩散模型的通用图像复原框架，通过多任务训练学习超分辨率、运动去模糊、散焦去模糊和去噪等专家知识，推理时通过灵活组合不同任务的隐空间预测权重来端到端地处理真实世界中的任意复杂退化组合。

## 研究背景与动机

真实世界的图像往往同时包含**多种退化**（复杂退化）：物体运动产生运动模糊、聚焦错误产生散焦模糊、高 ISO 产生噪声、低 JPEG 质量因子产生压缩伪影——这些退化在现实中经常**共现**。这给图像复原算法带来了极大挑战：

**训练数据构建困难**：创建包含真实复杂退化的配对 HQ-LQ 数据集极为困难。现有数据集要么退化类型单一，要么场景多样性不足。

**合成退化的泛化鸿沟**：Real-ESRGAN 等方法使用合成退化管道（高斯模糊+噪声+缩放+JPEG压缩）训练，但模型在面对真实图像时经常表现不佳。

**生成先验的不一致性问题**：StableSR、DiffBIR、SUPIR 等方法利用预训练扩散模型作为生成先验，通过冻结骨干网络+微调适配器实现盲复原。然而，这种基于 ControlNet 的条件机制容易产生**像素级结构不一致**（hallucination）——复原输出与输入的像素结构差异显著。

**All-in-one 方法的迭代式局限**：AutoDIR、RestoreAgent 等方法通过判断退化类型再迭代应用不同复原操作，效果受限于单步复原质量和迭代误差累积。

核心洞察：能否训练一组"专家"，每个专家擅长一种退化类型的修复，然后在推理时**灵活组合**这些专家的知识来处理任意复杂退化？这就是 UniRes 的设计思路。

## 方法详解

### 整体框架

UniRes 基于预训练的文本到图像 Latent Diffusion Model (LDM)，包含两个阶段：
- **训练阶段**：在超分辨率、运动去模糊、散焦去模糊和去噪 4 个任务上联合微调 LDM
- **推理阶段**：通过加权组合不同任务的隐空间预测，端到端处理复杂退化

### 关键设计

1. **隐空间拼接条件机制**:

    - **功能**：用 LQ 图像的隐空间编码作为扩散模型的条件输入。
    - **核心思路**：与 ControlNet/Adapter 方法不同，UniRes 将 LQ 图像隐向量 $\boldsymbol{z}_{\text{LQ}}$ 直接与噪声隐向量 $\boldsymbol{z}_t$ **拼接**后送入 UNet：
    $\boldsymbol{\epsilon}_\theta(\boldsymbol{z}_t, \boldsymbol{z}_{\text{LQ}}, \boldsymbol{s})$
      这种方式仅需修改 UNet 第一层卷积的输入通道数，且由于所有 UNet 参数都在 LQ-HQ 配对数据上微调并受到不一致性惩罚，因此**更好地保留输入的像素结构**。
    - **设计动机**：Adapter-Based 方法冻结骨干导致不一致和幻觉问题，拼接机制则让整个网络在训练中学会忠实于输入。

2. **多任务训练与隐空间预测组合**:

    - **功能**：训练时随机采样不同退化任务，推理时灵活组合各任务的噪声预测。
    - **核心思路**：设 $K$ 为任务数，推理时的组合预测为：
    $\tilde{\boldsymbol{\epsilon}}_\theta(\boldsymbol{z}_t, \boldsymbol{z}_{\text{LQ}}; \boldsymbol{w}) = \sum_{k=1}^{K} w_k \cdot \boldsymbol{\epsilon}_\theta(\boldsymbol{z}_t, \boldsymbol{z}_{\text{LQ}}, \boldsymbol{s}_k)$
      其中 $\boldsymbol{s}_k$ 为第 $k$ 个任务的文本提示（如"Super-resolution"、"Motion-deblur"等），$w_k$ 为权重且 $\sum w_k = 1$。这从概念上类似于 Mixture of Experts，每个文本提示激活模型内不同的"专家"知识。
    - **设计动机**：真实世界图像的退化是多种已知退化的复杂组合。通过调整权重，同一模型可以针对不同图像的退化特点动态调配修复策略。例如，模糊严重的夜景照片可以给运动去模糊分配高权重。

3. **DownLQ 保真度-质量权衡机制**:

    - **功能**：通过一个额外的"下采样 LQ"推理任务来控制生成细节的程度。
    - **核心思路**：在组合权重中增加一项 DownLQ，其条件输入是 LQ 图像先下采样（$\times 4$）再双三次上采样回原分辨率的版本。由于信息损失更严重，模型会对此输入生成更多细节。通过调节 DownLQ 权重可以平衡保真度和生成质量。
    - **设计动机**：拼接条件的模型本质上较为保守，不会过度生成细节。DownLQ 提供了一种自然的方式来鼓励细节生成，是 Adapter-Based 方法中 fidelity-quality tradeoff 的优雅替代。

4. **最优组合权重搜索**:

    - **功能**：自动确定每张图像的最优任务组合权重。
    - **核心思路**：定义搜索空间 $\Omega = \{\boldsymbol{w} \in [\gamma, \delta]^K | \sum w_k = 1\}$，通过网格搜索最大化图像质量评估函数：
    $\boldsymbol{w}^* = \arg\max_{\boldsymbol{w} \in \Omega} Q(g(\boldsymbol{x}, \boldsymbol{w}))$
      采用 MUSIQ 作为质量评估函数 $Q(\cdot)$，搜索范围 $[\gamma, \delta] = [-0.2, 1.2]$，步长 0.2。负权重的作用类似于 classifier-free guidance 的负向引导。
    - **设计动机**：不同图像的退化组合不同，统一权重不可能最优。自动搜索可以自适应地找到最适合每张图像的修复策略。搜索复杂度可通过使用频繁权重集（1512 降至 8）或随机森林预测权重来降低。

### 损失函数 / 训练策略

- 基于 WebLI 预训练的 text-to-image LDM（865M 参数），使用 DDPM 目标微调
- 多任务训练采样概率：SR 0.32, 运动去模糊 0.28, 散焦去模糊 0.18, 去噪 0.22
- 图像条件和文本条件随机丢弃率均为 0.1（支持 classifier-free guidance 和盲复原）
- 使用 JAX 在 32 TPU-v5 上训练 200K 步，batch size 256，lr 8e-5
- 输出经 AdaIN 色彩校正

## 实验关键数据

### 主实验（DiversePhotos×1，复杂退化，160 张图像）

| 方法 | ClipIQA ↑ | MUSIQ ↑ | ManIQA ↑ |
|------|----------|---------|----------|
| StableSR | 0.6227 | 61.39 | 0.3992 |
| DiffBIR | 0.6453 | 59.97 | 0.4922 |
| SUPIR | 0.5060 | 51.68 | 0.3745 |
| DACLIP-IR | 0.3497 | 46.16 | 0.2567 |
| **UniRes** | **0.6519** | **68.22** | **0.5021** |

UniRes 在 MUSIQ 上领先第二名 StableSR **6.83 分**，体现了对复杂退化的显著鲁棒性。

### 消融实验（DiversePhotos×1）

| 消融设置 | ClipIQA | MUSIQ | ManIQA | 说明 |
|---------|---------|-------|--------|------|
| UniRes (默认) | 0.6519 | 68.22 | 0.5021 | 完整模型 |
| 仅 SR 训练 | 0.4173 | 47.76 | 0.2921 | 单任务不够 |
| 单任务推理 SR | 0.4640 | 53.54 | 0.3423 | 一个专家不够 |
| 单任务推理 DN | 0.3744 | 39.21 | 0.2202 | 去噪任务最不通用 |
| DownLQ 仅 ×2 | 0.4883 | 55.38 | 0.3480 | 细节不足 |
| 去掉 SR | 0.5366 | 59.70 | 0.3959 | 各任务均有贡献 |
| 去掉 MD | 0.5595 | 61.05 | 0.4273 | 运动去模糊重要 |
| 去掉 DD | 0.5441 | 60.63 | 0.4075 | 散焦去模糊重要 |
| 搜索范围 [0,1] | 0.5667 | 63.24 | 0.4154 | 负权重有帮助 |
| 频繁权重集 (8组) | 0.6613 | 68.02 | 0.5101 | 大幅降低搜索成本 |
| 随机森林预测 | 0.5873 | 61.91 | 0.4257 | 可跳过搜索 |

### 关键发现

- **多任务训练的必要性**：仅用 SR 训练的模型在复杂退化上 MUSIQ 仅 47.76（vs 68.22），差距巨大
- **每个专家都不可或缺**：去掉任何一个任务都导致性能下降，所有任务的知识都对复杂退化修复有贡献
- **负权重的价值**：搜索范围扩展到 $[-0.2, 1.2]$ 比 $[0, 1]$ 高出约 5 分 MUSIQ，负权重起到了 classifier-free guidance 的推离效果
- **搜索复杂度可控**：使用 8 组频繁权重集即可达到接近完整搜索的效果
- 在 Real60 等单一退化基准上也保持竞争力，说明并未以牺牲单任务性能为代价

## 亮点与洞察

- **优雅的MoE思想**：将文本提示作为"专家选择器"，权重组合作为"路由策略"，整个框架像一个端到端的可微分专家混合系统
- **拼接条件 vs Adapter**：论文重新审视了一条被主流忽视的技术路线（隐空间拼接），证明其在保真度方面的优越性
- **DownLQ 机制**：通过故意降低输入质量来激发模型生成更多细节，思路反直觉但有效
- **DiversePhotos 基准**：填补了复杂退化基准测试的空白，每张图像包含至少两种真实退化
- **可扩展的统一公式**：Eq. 2 的框架足够灵活，可以轻松添加新的复原任务或操纵任务

## 局限与展望

- **推理成本高**：每个权重组合需要 $K$ 次前向传播（6 个专家 = 6 次），网格搜索更是将推理时间放大上千倍
- **仅关注相机退化**：不包括恶劣天气（雨、雪、雾）等退化类型
- **MUSIQ 作为优化目标的局限**：图像质量评估函数本身的偏差可能影响权重优化方向
- **无全参考指标评估复杂退化**：DiversePhotos 缺乏 HQ 参考图像，仅能使用无参考指标
- **暂无代码开源**
- 色彩校正依赖后处理（AdaIN），未与模型端到端训练

## 相关工作与启发

- 与 Mixture of Experts (MoE) 的关系：UniRes 本质上是一种"推理时MoE"，专家共享参数但通过不同文本提示激活不同知识
- 与 Diffusion Soup（模型合并）的联系：在隐空间层面组合预测，而非在参数空间合并权重
- ControlNet 方法的不一致性问题启发了更简单的拼接条件方案
- 对未来工作的启发：可以用退化感知的特征学习来替代网格搜索，实现更高效的权重预测

## 评分

- 新颖性: ⭐⭐⭐⭐ 隐空间预测的加权组合思路简洁优雅，DownLQ 机制有创意，但整体建立在成熟的 LDM 框架上
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面，但缺乏全参考指标验证和推理效率分析；DiversePhotos 基准有价值但规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 论文写作流畅，动机阐述清晰，方法公式化简洁明了
- 价值: ⭐⭐⭐⭐ 对真实世界复杂退化问题提出了有效解决方案，DiversePhotos 填补了基准空白，但推理成本限制了实用性

<!-- RELATED:START -->

## 相关论文

- [MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration](../../ECCV2024/image_restoration/moe-diffir_task-customized_diffusion_priors_for_universal_compressed_image_resto.md)
- [FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration](foundir_unleashing_million-scale_training_data_to_advance_foundation_models_for_.md)
- [Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)

<!-- RELATED:END -->
