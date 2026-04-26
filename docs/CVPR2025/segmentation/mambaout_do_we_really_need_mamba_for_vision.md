---
title: >-
  [论文解读] MambaOut: Do We Really Need Mamba for Vision?
description: >-
  [CVPR 2025][图像分割][Mamba] 本文通过概念分析指出 Mamba 的 SSM 机制适用于长序列+自回归任务，而 ImageNet 图像分类两者都不满足，因此构建了去掉 SSM 的 MambaOut（纯 Gated CNN）系列模型，在图像分类上全面超越所有视觉 Mamba 模型，有力证明了 SSM 对视觉分类是不必要的。
tags:
  - CVPR 2025
  - 图像分割
  - Mamba
  - SSM
  - Gated CNN
  - 视觉识别
  - 图像分类
---

# MambaOut: Do We Really Need Mamba for Vision?

**会议**: CVPR 2025  
**arXiv**: [2405.07992](https://arxiv.org/abs/2405.07992)  
**代码**: https://github.com/yuweihao/MambaOut  
**领域**: 图像分类 / 视觉骨干网络  
**关键词**: Mamba, SSM, Gated CNN, 视觉识别, 图像分类

## 一句话总结
本文通过概念分析指出 Mamba 的 SSM 机制适用于长序列+自回归任务，而 ImageNet 图像分类两者都不满足，因此构建了去掉 SSM 的 MambaOut（纯 Gated CNN）系列模型，在图像分类上全面超越所有视觉 Mamba 模型，有力证明了 SSM 对视觉分类是不必要的。

## 研究背景与动机

1. **领域现状**：Mamba 作为一种基于 SSM 的 RNN-like 架构，凭借线性复杂度在 NLP 中表现出色，随后被引入视觉任务（Vision Mamba、VMamba、PlainMamba 等），试图替代 Transformer 的二次复杂度 attention。
2. **现有痛点**：然而，视觉 Mamba 模型在实际性能上令人失望——与卷积模型和 attention 模型相比始终落后。例如 CAFormer-M36 使用 7 年前的分离卷积+vanilla attention 就能超过所有同等大小的视觉 Mamba 模型 1% 以上。
3. **核心矛盾**：社区一直在给 Mamba 加各种视觉改进（双向扫描、局部归纳偏置等），却没人从根本上追问：SSM 对视觉任务到底是不是必要的？
4. **本文目标**：从 Mamba 的 RNN 本质出发，分析 SSM 适合的任务特征，然后检验视觉任务是否符合这些特征。
5. **切入角度**：作者从记忆机制出发——SSM 的固定大小隐状态是有损记忆，只有在长序列时才能体现优势（attention 会爆内存）；同时 SSM 的递归本质决定了它是 causal mode（只能看到之前的 token），适合自回归任务而非理解任务。
6. **核心 idea**：SSM 适合长序列+自回归任务，ImageNet 分类两个特征都不满足，所以 SSM 是不必要的——去掉 SSM 的 Gated CNN 就能超越视觉 Mamba。

## 方法详解

### 整体框架
MambaOut 采用类似 ResNet 的 4 阶段层级架构，每个阶段堆叠 Gated CNN block。输入图像经过 patch embedding 后依次通过 4 个 stage，每个 stage 之间通过下采样降低分辨率、提升通道数。最终通过全局平均池化 + 分类头输出预测。核心关注点不在于提出新架构，而在于**去掉 SSM 后验证假设**。

### 关键设计

1. **Gated CNN Block（核心组件）**:
    - 功能：作为 Mamba block 的"去 SSM 版本"，用纯卷积实现 token mixing
    - 核心思路：对输入 $X$ 先做 LayerNorm，然后通过一个线性层投影到两个分支——一个经过 depthwise conv（7×7 kernel，部分通道）做 token mixing，另一个通过激活函数 GELU 做 gating；两者逐元素相乘后再通过线性层投影回原维度，加残差连接。公式为 $Y = (\text{TokenMixer}(X'W_1) \odot \sigma(X'W_2))W_3 + X$
    - 设计动机：Mamba block 本质上就是在 Gated CNN 的基础上加了 SSM。移除 SSM 后，Gated CNN block 保留了门控卷积的表达能力，可以直接验证 SSM 的贡献

2. **部分通道卷积策略**:
    - 功能：只在部分通道上进行 depthwise conv，提高实际推理速度
    - 核心思路：借鉴 InceptionNeXt，将 hidden 维度拆为三部分——gating 分支、identity 分支和卷积分支，只有卷积分支做 depthwise conv，其余直接 pass through，最后 concat
    - 设计动机：全通道 depthwise conv 虽然 FLOPs 低但实际速度慢（内存访问瓶颈），部分通道策略在几乎不损失精度的情况下大幅提升吞吐量

3. **概念性分析框架（双特征判据）**:
    - 功能：从理论层面判断 Mamba 对某类任务是否必要
    - 核心思路：(1) 长序列特征：当 token 数量 $L > 6D$（D 为通道数）时，attention 的二次项才主导计算量。ImageNet 224² 分辨率只有 196 个 token，远小于阈值 2304（ViT-S）；检测分割约 4K tokens，接近阈值。(2) 自回归特征：视觉理解任务是 fully-visible mode（模型一次看到整张图），而 SSM 天然是 causal mode（只能看当前及之前的 token），对理解任务反而有害——实验证明给 ViT 加 causal mask 后精度下降
    - 设计动机：为实验提供清晰的理论预期：分类不需要 SSM（Hypothesis 1），检测分割值得探索 SSM（Hypothesis 2）

### 损失函数 / 训练策略
采用标准 DeiT 训练方案（无蒸馏）：RandAugment、Mixup、CutMix、Random Erasing、label smoothing、stochastic depth 等。优化器为 AdamW，学习率 $lr = \frac{batchsize}{1024} \times 10^{-3}$，batch size 4096，在 TPU v3 上训练。

## 实验关键数据

### 主实验

| 模型 | Token Mixer | Params | MACs | Top-1 Acc |
|------|------------|--------|------|-----------|
| MambaOut-Femto | Conv | 7M | 1.2G | 78.9% |
| EfficientVMamba-S | Conv+SSM | 11M | 1.3G | 78.7% |
| MambaOut-Tiny | Conv | 27M | 4.5G | 82.7% |
| VMamba-T | Conv+SSM | 22M | 5.6G | 82.2% |
| LocalVMamba-T | Conv+SSM | 26M | 5.7G | 82.7% |
| MambaOut-Small | Conv | 48M | 9.0G | 84.1% |
| VMamba-S | Conv+SSM | 44M | 11.2G | 83.5% |
| LocalVMamba-S | Conv+SSM | 50M | 11.4G | 83.7% |
| MambaOut-Base | Conv | 85M | 15.8G | 84.2% |
| VMamba-B | Conv+SSM | 75M | 18.0G | 83.7% |

MambaOut 在所有尺度上均超越视觉 Mamba 模型，且 MACs 更低。

### 消融实验

| 实验设定 | 结论 |
|---------|------|
| ViT causal vs fully-visible | 加 causal mask 后 ViT 精度显著下降，证明视觉理解不需要 causal mixing |
| 检测/分割任务 | MambaOut 无法匹配 SOTA Mamba 模型（VMamba-T: 47.3 AP vs MambaOut: 低于此），支持 Hypothesis 2 |
| ImageNet 分类 | MambaOut 全面超越 Mamba 模型，支持 Hypothesis 1 |

### 关键发现
- **SSM 对 ImageNet 分类完全不必要**，纯 Gated CNN 去掉 SSM 后反而更好，说明 SSM 在短序列理解任务中是负面的
- **SSM 在检测/分割任务中仍有价值**，因为这些任务的 token 序列较长（~4K），SSM 的线性复杂度优势可以发挥
- 当前视觉 Mamba 模型与 SOTA 卷积/attention 混合模型（如 CAFormer-M36: 85.2%）仍有 >1% 的差距

## 亮点与洞察
- **从第一性原理分析**：不盲目跟风改进 Mamba for Vision，而是退一步追问"Mamba 的 SSM 到底适合什么任务"，从 RNN 记忆机制和 token mixing 模式两个维度给出清晰判据。这种思维方式可迁移到任何"XX for YY"的跨领域迁移问题
- **Occam's Razor 实践**：MambaOut 作为最简 baseline，用"减法"而非"加法"证明观点。去掉 SSM 效果更好，比复杂改进更有说服力
- **长序列阈值公式**：$L > 6D$ 的判据简洁实用，可以快速判断任何视觉任务是否受益于线性复杂度 token mixer

## 局限与展望
- 作者只验证了分类/检测/分割三类任务，对视频理解、点云等真正的长序列视觉任务未做验证
- MambaOut 的 Gated CNN block 在下游任务（检测/分割）上不如 SSM 模型，说明纯卷积的全局建模能力确实有局限
- 论文未讨论最新的 Mamba-2 等改进 SSM 是否能改变结论
- 双向 SSM（bidirectional branches）虽然不完美，但论文对其分析略显简单——每个 branch 仍然是 causal 不代表组合后还是 causal

## 相关工作与启发
- **vs VMamba**: VMamba 使用 Cross-Scan 四向扫描，MambaOut 直接去掉 SSM 用 depthwise conv，在分类上 MambaOut 以更低 MACs 取胜，但在检测上 VMamba 更强
- **vs MetaFormer/CAFormer**: MetaFormer 系列证明了 token mixer 甚至可以是 pooling，与 MambaOut 的"SSM 不必要"结论一脉相承。CAFormer 用简单卷积+attention 远超所有 Mamba 模型
- **vs ConvNeXt**: 都是纯卷积骨干，MambaOut 的 Gated CNN block 更接近 Mamba 的结构设计（门控+depthwise conv），但核心贡献在于概念性分析而非架构创新

## 评分
- 新颖性: ⭐⭐⭐⭐ 概念分析框架新颖，但模型本身无太多创新
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类/检测/分割，多尺度对比全面，但缺少更多任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从理论到实验环环相扣，致敬 Kobe "Mamba Out" 的立意也很有趣
- 价值: ⭐⭐⭐⭐ 为 Mamba for Vision 提供了重要反思和 baseline，但不影响 Mamba 在长序列任务中的价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](../../ICCV2025/segmentation/tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[AAAI 2026\] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation](../../AAAI2026/segmentation/do_we_need_perfect_data_leveraging_noise_for_domain_generalized_segmentation.md)
- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](../../ICCV2025/segmentation/vssd_vision_mamba_with_non-causal_state_space_duality.md)

<!-- RELATED:END -->
