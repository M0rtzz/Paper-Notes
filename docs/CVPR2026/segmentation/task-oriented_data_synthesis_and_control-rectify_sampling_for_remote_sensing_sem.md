---
title: >-
  [论文解读] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation
description: >-
  [CVPR 2026][图像分割][遥感语义分割] 本文提出TODSynth框架，通过MM-DiT的统一三模态注意力实现文本-图像-掩码联合控制的遥感图像合成，并创新性地提出控制-校正流匹配（CRFM）方法，在采样阶段利用下游分割模型的语义损失动态调整生成轨迹，使合成数据在FUSU-4k和LoveDA上分别提升4.14%和2.08%的mIoU。
tags:
  - CVPR 2026
  - 图像分割
  - 遥感语义分割
  - 数据合成
  - 可控生成
  - 扩散模型
  - 流匹配
---

# Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation

**会议**: CVPR 2026  
**arXiv**: [2512.16740](https://arxiv.org/abs/2512.16740)  
**代码**: [GitHub](https://github.com/Yunkai-Yang/crfm)  
**领域**: 分割 / 遥感图像  
**关键词**: 遥感语义分割, 数据合成, 可控生成, 扩散模型, 流匹配

## 一句话总结

本文提出TODSynth框架，通过MM-DiT的统一三模态注意力实现文本-图像-掩码联合控制的遥感图像合成，并创新性地提出控制-校正流匹配（CRFM）方法，在采样阶段利用下游分割模型的语义损失动态调整生成轨迹，使合成数据在FUSU-4k和LoveDA上分别提升4.14%和2.08%的mIoU。

## 研究背景与动机

**领域现状**：遥感语义分割是土地利用分类、环境监测的基础任务，但构建大规模像素级标注数据集成本极高。近年来，基于扩散模型的数据合成成为扩充训练集的有前景方案，ControlNet等方法可以从语义掩码生成对应图像。

**现有痛点**：（1）**控制方案不成熟**：DiT架构的生成模型（如SD v3.5）已显著优于UNet架构，但如何有效地在DiT中注入语义掩码控制仍是开放问题。适配器方式的cross-attention控制效率低、存在模态冲突。（2）**采样质量不稳定**：即使控制方案合理，扩散/流匹配采样过程的随机性仍会导致生成图像在局部区域偏离掩码约束（语义漂移），降低合成数据对下游任务的有效性。（3）**后处理方案有限**：现有方法（如CLIP打分、FreeMask自适应过滤）都是生成后的补救措施，在复杂场景或少样本类别中，严格过滤会丢弃有用标注。

**核心矛盾**：生成模型的随机性与下游任务需要的确定性语义控制之间的矛盾。遥感图像的域差异大、缺少预训练DiT模型、精细文本描述稀缺，进一步加剧了这一矛盾。

**本文目标** （1）找到适合遥感M2I（mask-to-image）任务的DiT控制方案；（2）在采样过程中（而非生成后）纠正语义偏移，提升合成数据的任务相关性。

**切入角度**：作者观察到直接优化隐变量 $z_t$ 会导致模式崩溃，而优化速度场 $v_\Theta$ 则提供稳定的连续校正。基于此，在流匹配的早期高可塑阶段注入下游分割模型的梯度信号来校正生成轨迹。

**核心 idea**：用三模态联合注意力做架构级控制，用下游分割损失的梯度在采样早期做速度场校正（CRFM），实现任务导向的遥感数据合成。

## 方法详解

### 整体框架

TODSynth分为三个阶段：（1）训练阶段：在SD v3.5基础上，用统一三模态注意力（Tri-Attention）的MM-DiT模型，以文本和语义掩码为条件训练遥感图像生成器；（2）采样阶段：用CRFM方法在生成过程的早期步骤中，利用预训练分割模型的交叉熵损失梯度校正速度场；（3）下游训练：将合成数据与真实数据混合训练分割模型。

### 关键设计

1. **统一三模态注意力（Tri-Attention）**:

    - 功能：在DiT架构中实现文本、图像、语义掩码三种模态的深度融合控制
    - 核心思路：在MM-DiT原有的文本-图像双模态联合注意力基础上，增加第三个模态流处理掩码序列 $h^m$。三个模态各自有独立的 $W_q, W_k, W_v$ 投影矩阵，token拼接后在同一个注意力中计算：$h_o^t, h_o^z, h_o^m = \text{Attn}([h^t W_q^t, h^z W_q^z, h^m W_q^m], ...)$。这使得掩码信息可以直接与文本embedding交互，增强全局语义理解
    - 设计动机：Mask-adapter方式中掩码不与文本融合，语义利用不充分且掩码表示在去噪过程中固定不变。Siamese方式中M2I的纯掩码缺少局部文本描述，削弱了解耦的优势。Tri-Attention最简洁地实现了三模态交叉注意力

2. **控制-校正流匹配（CRFM）**:

    - 功能：在采样阶段动态修正生成轨迹，确保合成图像更忠实于语义掩码约束
    - 核心思路：在流匹配采样的早期步骤（高可塑阶段），先用当前状态 $z_t$ 和预测速度场 $v^P$ 预估最终生成结果 $z_0^t = z_t - \sigma_t v^P$，通过VAE解码器得到预合成图像 $x_0^t$，将其输入预训练分割网络计算交叉熵损失 $\mathcal{L}_{CE}(\mathcal{S}(x_0^t), C^m)$，对速度场求梯度得到校正向量 $v_{rec}' = -\nabla_{v_t} \mathcal{L}_{CE}$。最终更新速度场为 $v' = v^P + \alpha \cdot v_{rec}'$
    - 设计动机：（1）直接优化 $z_t$ 会导致模式崩溃（生成样本多样性丧失），而优化速度场通过ODE积分间接更新 $z_t$，提供更稳定的校正；（2）只在早期步骤校正，因为早期随机性高、可塑性强，且分割模型预测误差在粗粒度调整时影响可控；晚期校正可能放大分割模型的误差，产生对抗性扰动

3. **全参数微调图像和掩码分支**:

    - 功能：弥补SD v3.5在遥感域上缺乏预训练的域差异
    - 核心思路：由于不存在基于DiT在遥感数据上预训练的生成模型，冻结backbone+适配器的策略效果有限。本文选择全参数微调图像分支和掩码分支，使模型充分适应遥感图像的特殊分布
    - 设计动机：遥感图像与自然图像域差异大（俯视视角、光谱特性、尺度变化），需要更充分的适配

### 损失函数 / 训练策略

训练阶段使用标准的Rectified Flow损失（速度场预测的MSE损失）。CRFM的校正强度由超参数 $\alpha$ 控制。后处理使用FreeMask的像素级过滤策略。合成/真实数据比例为3:1。模型在8×RTX 4090上训练200K步，分辨率512×512，AdamW优化器，学习率$10^{-5}$。

## 实验关键数据

### 主实验

| 方法 | 后处理 | 合成/真实 | FUSU-4k OA | FUSU-4k mIoU | FUSU-4k mAcc |
|------|--------|----------|------------|-------------|-------------|
| Baseline (仅真实) | - | - | 74.27 | 45.27 | 56.44 |
| ControlNet (SD v1.5) | × | ×10 | 73.85 | 45.13 | 56.77 |
| FreeMask | FM | ×5 | 74.23 | 45.83 | 56.29 |
| SynthEarth | CLIP | ×5 | 75.35 | 47.53 | 58.91 |
| SD v3.5 (Tri-Attn) | FM | ×3 | 75.41 | 48.57 | 61.67 |
| **TODSynth (Ours)** | FM | **×3** | **75.66** | **49.41** | **63.27** |

LoveDA数据集：TODSynth相比baseline提升 OA +1.60% / mIoU +2.08% / mAcc +2.22%。

### 消融实验

控制策略对比（FUSU-4k）：

| 方法 | OA | mIoU | mAcc |
|------|-----|------|------|
| ControlNet (SD v1.5) | 73.85 | 45.13 | 56.77 |
| Mask-adapter | 74.94 | 47.41 | 59.62 |
| Siamese MM-attention | 74.94 | 48.46 | 61.44 |
| **Tri-Attention** | **75.41** | **48.57** | **61.67** |

CRFM步数消融（step=23）：

| CRFM步数 | mIoU | mAcc | FID |
|----------|------|------|-----|
| 0 (无校正) | 48.57 | 61.67 | 35.85 |
| 2 | 48.80 | 61.05 | 34.86 |
| **4** | **49.41** | **63.27** | 38.65 |
| 6 | 48.74 | 61.30 | 66.95 |

### 关键发现
- **DiT >> UNet**：同样是可控生成，MM-DiT方法大幅优于UNet-based的ControlNet和FreeMask，即使SynthEarth是专门的遥感生成基础模型
- **CRFM有效但需控制步数**：4步校正在mIoU/mAcc上最优；过多校正步（6步）导致FID急剧上升（66.95），说明过度校正会破坏图像质量
- **像素级过滤 > 图像级过滤**：FreeMask的像素级过滤显著优于CLIP的图像级过滤，更精细的筛选适合分割任务
- TODSynth用**更少的合成数据**（×3 vs ×5/×10）取得更好效果，体现了任务导向合成的高效性
- 直接优化隐变量导致模式崩溃，验证了校正速度场而非隐变量的设计必要性

## 亮点与洞察
- **速度场校正 vs 隐变量优化**：这是本文最核心的洞察。在流匹配框架中，优化速度场而非直接修改隐变量，避免了模式崩溃，提供了稳定的轨迹校正。这一思路可推广到其他条件生成任务
- **任务反馈驱动采样**：不像传统方法在生成后筛选，而是在生成过程中利用下游任务信号引导采样——从"生成后选优"到"生成中纠偏"的范式转变
- **早期可塑性窗口**：发现流匹配早期步骤是校正的最佳时机，晚期校正反而有害，这与扩散模型中"早期决定语义、晚期决定细节"的观察一致
- 三模态注意力的简洁实现证明了"统一融合优于解耦处理"在遥感M2I场景下的适用性

## 局限与展望
- CRFM依赖预训练分割模型的质量——如果分割模型本身在目标域上不好，校正信号可能不准确
- 目前 $\alpha$ 和CRFM步数需要手动调参，自适应调节策略可能更鲁棒
- 仅在两个遥感数据集上验证，是否对医学图像等其他标注稀缺域同样有效有待验证
- 512×512分辨率对于高分辨率遥感图像可能不够，需要探索更高分辨率的生成方案
- 全参数微调计算成本高（8×4090），LoRA等轻量微调方案的效果对比缺失

## 相关工作与启发
- **vs ControlNet**: 基于UNet的控制方案，在遥感域上效果有限。本文用DiT的Tri-Attention显著更优
- **vs FreeMask**: 后处理过滤方案，与CRFM互补。本文证明将CRFM叠加在FreeMask之上可进一步提升
- **vs SynthEarth**: 遥感生成基础模型，使用CLIP打分过滤。本文用更少数据量（×3 vs ×5）取得更好效果
- **vs 训练无关L2I编辑**: 直接优化隐变量的方法会导致模式崩溃。本文的速度场校正提供了更优的替代方案

## 评分
- 新颖性: ⭐⭐⭐⭐ CRFM速度场校正思路新颖，"生成中纠偏"的范式值得关注
- 实验充分度: ⭐⭐⭐⭐ 控制策略和CRFM超参数的消融实验充分，但仅两个数据集略显不足
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验框架完整
- 价值: ⭐⭐⭐⭐ 对遥感数据合成有实际价值，CRFM思路可推广到其他域

<!-- RELATED:START -->

## 相关论文

- [SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semanticguided_modalityaware_segmentation_for.md)
- [General and Task-Oriented Video Segmentation](../../ECCV2024/segmentation/general_and_task-oriented_video_segmentation.md)
- [Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](../../ACL2025/segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)

<!-- RELATED:END -->
