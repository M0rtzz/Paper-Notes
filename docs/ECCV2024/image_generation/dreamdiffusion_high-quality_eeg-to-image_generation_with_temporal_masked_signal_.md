---
title: >-
  [论文解读] DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment
description: >-
  [ECCV 2024][图像生成][EEG信号生成图像] 本文提出 DreamDiffusion，利用时序掩码信号建模对EEG编码器进行大规模预训练学习鲁棒的脑电表征，再通过CLIP图像编码器提供额外监督将EEG-文本-图像空间对齐，最终借助预训练的Stable Diffusion从脑电信号直接生成高质量图像，实现便携低成本的"思维转图像"。
tags:
  - ECCV 2024
  - 图像生成
  - EEG信号生成图像
  - 脑信号解码
  - 时序掩码预训练
  - CLIP对齐
  - 扩散模型
---

# DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment

**会议**: ECCV 2024  
**arXiv**: [2306.16934](https://arxiv.org/abs/2306.16934)  
**代码**: https://github.com/bbaaii/DreamDiffusion  
**领域**: 扩散模型 / 脑机接口  
**关键词**: EEG信号生成图像、脑信号解码、时序掩码预训练、CLIP对齐、Stable Diffusion

## 一句话总结

本文提出 DreamDiffusion，利用时序掩码信号建模对EEG编码器进行大规模预训练学习鲁棒的脑电表征，再通过CLIP图像编码器提供额外监督将EEG-文本-图像空间对齐，最终借助预训练的Stable Diffusion从脑电信号直接生成高质量图像，实现便携低成本的"思维转图像"。

## 研究背景与动机

**领域现状**：图像生成领域在文本到图像方向取得了巨大突破（DALL-E, Stable Diffusion等）。最近的研究开始探索直接从脑信号生成图像——如MinD-Vis利用fMRI信号生成图像取得了不错效果。这类"思维到图像"的研究在神经科学和人机交互领域具有广阔前景。

**现有痛点**：(1) 现有基于fMRI的方法虽然效果较好，但fMRI设备昂贵、不便携、需要专业人员操作，极大限制了实际应用；(2) EEG（脑电图）虽然便携且成本低，但信号噪声大、信息量有限、个体差异显著，直接从EEG生成图像极具挑战；(3) EEG-图像配对数据极其稀少，难以端到端训练高质量的条件生成模型；(4) EEG信号的特征空间与Stable Diffusion中已对齐的文本-图像空间差异巨大。

**核心矛盾**：要实现从EEG直接生成图像，需要解决两个根本性问题——如何从噪声大且数据量少的EEG信号中提取有效的语义表征？如何将EEG表征与预训练扩散模型的文本-图像空间对齐？

**本文目标** (1) 如何利用大量无标注EEG数据（无需配对图像）学习鲁棒的EEG表征？(2) 如何用极少的EEG-图像配对数据将EEG空间与CLIP文本-图像空间对齐？

**切入角度**：作者注意到EEG信号具有强烈的时序特性，不同于fMRI的空间特性。因此不同于MAE和MinD-Vis对空间信息进行掩码，本文对时间域进行掩码建模。另外，利用CLIP图像编码器作为桥梁——因为CLIP的文本和图像空间已经对齐，将EEG拉近CLIP图像空间就等价于拉近文本空间，从而适配Stable Diffusion。

**核心 idea**：用时序掩码预训练从大规模噪声EEG数据中学习鲁棒表征，再用CLIP图像编码器搭桥将EEG空间与SD的文本-图像空间对齐，实现从EEG直接生成高质量图像。

## 方法详解

### 整体框架

DreamDiffusion的方法流程分为三个阶段：(1) 时序掩码信号预训练阶段——使用大量无配对的EEG数据（来自MOABB平台的约120,000个样本、400+被试）通过掩码自监督学习预训练EEG编码器；(2) Stable Diffusion微调阶段——使用少量EEG-图像配对数据（ImageNet-EEG数据集）微调EEG编码器和SD的交叉注意力层；(3) CLIP对齐阶段——利用CLIP图像编码器提供额外监督，将EEG嵌入拉近CLIP图像嵌入以更好地适配SD。推理时只需输入EEG信号，通过编码器得到条件嵌入，然后由SD生成对应图像。

### 关键设计

1. **时序掩码信号建模（Temporal Masked Signal Modeling）**:

    - 功能：从大规模无标注EEG数据中学习通用且鲁棒的EEG表征
    - 核心思路：EEG信号是128通道 × 时间步的二维数据。不同于MAE对空间维度掩码，本文考虑EEG信号的时序特性，在时间维度上将信号分成token（每4个相邻时间步为一个token），随机掩码75%的token，然后使用类ViT-Large的编码器-解码器架构重建被掩码的时间段。重建损失为MSE，仅在被掩码的patch上计算。预训练500个epoch后丢弃解码器，仅保留编码器用于下游任务
    - 设计动机：EEG信号高度噪声且个体差异大，传统监督学习难以学到鲁棒表征。自监督掩码预训练可以利用大量无配对的EEG数据。选择时序掩码而非空间掩码是因为EEG的时间分辨率高但空间分辨率低，时序变化中包含更丰富的语义信息。75%的掩码比例实验证明是最优的，这与NLP中常用的低掩码比例不同

2. **Stable Diffusion微调（Fine-tuning with SD）**:

    - 功能：利用预训练的SD生成能力，将EEG条件接入扩散模型
    - 核心思路：预训练的EEG编码器输出经过投影层变换为与SD文本嵌入同维度的条件嵌入 $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$。通过交叉注意力机制将EEG条件信息注入U-Net中，其中 $Q = W_Q \cdot \varphi_i(z_t)$, $K = W_K \cdot \tau_\theta(y)$, $V = W_V \cdot \tau_\theta(y)$。微调时优化EEG编码器和U-Net的交叉注意力头参数，冻结SD其他部分。使用标准SD损失 $\mathcal{L}_{SD} = \mathbb{E}_{x,\epsilon,t}[\|\epsilon - \epsilon_\theta(x_t, t, \tau_\theta(y))\|_2^2]$
    - 设计动机：SD已经在大规模文本-图像数据上学习了强大的图像生成能力，通过微调交叉注意力层可以将EEG条件替代文本条件接入已有的生成pipeline。冻结U-Net的大部分参数可以保持生成质量同时避免在小数据集上过拟合

3. **CLIP空间对齐（CLIP Alignment）**:

    - 功能：进一步优化EEG嵌入使其与CLIP的图像-文本空间对齐
    - 核心思路：由于SD使用CLIP文本编码器生成条件嵌入，EEG嵌入需要尽可能接近CLIP空间才能有效驱动生成。本文利用CLIP图像编码器 $E_I$ 提取配对图像的CLIP嵌入，然后通过投影层 $h$ 将EEG嵌入映射到同一空间。对齐损失定义为余弦距离：$\mathcal{L}_{clip} = 1 - \frac{E_I(I) \cdot h(\tau_\theta(y))}{|E_I(I)| \cdot |h(\tau_\theta(y))|}$。CLIP模型在此过程中保持冻结
    - 设计动机：直接用有限的EEG-图像配对数据端到端微调SD难以准确对齐EEG和文本空间。CLIP作为桥梁，由于其文本和图像空间已高度对齐，将EEG拉近CLIP图像空间就隐式地拉近了文本空间。实验证明即使没有预训练，仅用CLIP对齐也能获得合理结果，凸显了CLIP监督的重要性

### 损失函数 / 训练策略

训练包含三个阶段的损失：(1) 预训练阶段使用MSE重建损失；(2) 微调阶段使用SD扩散损失 $\mathcal{L}_{SD}$；(3) 对齐阶段在SD损失基础上加入CLIP对齐损失 $\mathcal{L}_{clip}$。使用SD 1.5版本，EEG信号预处理时滤波至5-95Hz、截断到512长度，编码器基于ViT-Large架构。预训练500 epochs，微调300 epochs。所有实验使用Subject 4的数据。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 DreamDiffusion | Brain2Image | 说明 |
|---|---|---|---|---|
| ImageNet-EEG | 50-way Top-1 Acc (%) | 45.8 | - | 使用预训练ImageNet分类器评估语义正确性 |
| ImageNet-EEG | 图像质量 | 显著优于 | 低质量模糊 | 定性对比DreamDiffusion远胜Brain2Image |

### 消融实验

| 配置 | Top-1 Acc (%) | 说明 |
|---|---|---|
| 完整模型（MSM预训练+CLIP+大编码器） | 45.8 | 最佳配置 |
| 无预训练+无CLIP+大编码器（Model 1） | 4.2 | 验证预训练和CLIP的必要性 |
| 无预训练+无CLIP+小编码器（Model 2） | 3.7 | 小编码器也无法避免过拟合 |
| 无预训练+有CLIP+大编码器（Model 3） | 32.3 | CLIP对齐单独就有很大帮助 |
| 无预训练+有CLIP+小编码器（Model 4） | 24.5 | 大编码器有优势 |
| 有预训练+有CLIP+mask ratio 0.25（Model 5） | 19.7 | 掩码比例过低 |
| 有预训练+有CLIP+mask ratio 0.50（Model 6） | - | 中等效果 |
| 有预训练+有CLIP+mask ratio 0.75 | 45.8 | 最优掩码比例 |

### 关键发现

- 时序掩码预训练是关键：无预训练时准确率仅4.2%，有预训练后达45.8%
- CLIP对齐极其重要：即使无预训练，仅加CLIP监督就能从4.2%提升到32.3%
- 75%的掩码比例最优，与NLP中的低掩码比例不同，说明EEG信号有类似视觉信号的冗余特性
- 大编码器（297M参数）显著优于小编码器（18.3M），说明EEG信号的建模需要足够的模型容量
- 部分生成失败案例中，形状或颜色相似的类别会被混淆，说明脑电信号在类别级提供的是粗粒度信息

## 亮点与洞察

- 首次实现从EEG信号直接生成高质量图像，相比fMRI大幅降低成本和使用门槛
- 时序掩码预训练的设计很有洞察力——EEG时间分辨率高但空间分辨率低，时序掩码比空间掩码更合适
- 利用CLIP作为EEG-文本-图像空间对齐的桥梁非常巧妙，避免了直接对齐的困难
- 数据来源设计好——用MOABB平台400+被试的数据预训练，增强了跨个体的泛化性
- 方法名"DreamDiffusion"很有吸引力，暗示了可视化梦境的潜在应用

## 局限与展望

- EEG信号当前只能提供类别级别的粗粒度语义信息，无法捕获具体的视觉细节（如颜色相似的不同物体容易混淆）
- 实验仅使用Subject 4的数据，跨被试的泛化性未被验证
- EEG-图像配对数据量极小（2000图×6被试），限制了微调效果
- 生成的图像与原始刺激图像的像素级对应关系较弱，更多是语义级别的匹配
- 可以考虑引入更多的EEG通道选择策略或注意力机制来提取更精细的信号特征
- 未来可以探索EEG+fMRI的互补融合或联合训练方式
- 可以考虑引入对比学习来增强EEG预训练的效果

## 相关工作与启发

- **MinD-Vis**：基于fMRI的脑信号图像生成，使用SC-MBM+DC-LDM，效果好但设备昂贵
- **Brain2Image**：早期基于EEG的图像生成工作，使用LSTM+GAN/VAE，质量有限
- **Stable Diffusion**：强大的文本到图像扩散模型，本文利用其作为生成引擎
- **MAE**：掩码自编码器，本文借鉴其掩码预训练思路但改为时序掩码
- **CLIP**：视觉-语言对齐模型，本文用其图像编码器搭桥对齐EEG空间
- 启发：预训练+CLIP对齐的范式可以推广到其他模态（如肌电信号EMG、眼动信号等）到图像的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次实现高质量EEG到图像的生成，时序掩码预训练和CLIP桥接设计新颖
- 实验充分度: ⭐⭐⭐ 消融实验详尽，但缺少跨被试评估和更多定量指标
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法流程易懂，图表丰富
- 价值: ⭐⭐⭐⭐ 开创性工作，推动便携低成本"思维到图像"技术的发展

<!-- RELATED:START -->

## 相关论文

- [MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)
- [Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)
- [RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models](rodinhd_high-fidelity_3d_avatar_generation_with_diffusion_models.md)

<!-- RELATED:END -->
