---
title: >-
  [论文解读] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution
description: >-
  [CVPR 2026][图像生成][真实图像超分辨率] FRAMER 提出频率对齐的自蒸馏训练框架，通过将最终层特征图作为教师监督中间层，并按低频/高频分别施加 IntraCL 和 InterCL 对比损失，配合自适应权重调节(FAW)和对齐门控(FAM)，在不改变网络结构和推理流程的情况下，显著提升扩散模型在真实图像超分辨率任务的高频细节恢复能力。
tags:
  - CVPR 2026
  - 图像生成
  - 真实图像超分辨率
  - 自蒸馏
  - 频率感知
  - 扩散先验
  - 即插即用
---

# FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2512.01390](https://arxiv.org/abs/2512.01390)  
**代码**: https://cmlab-korea.github.io/FRAMER/  
**领域**: 扩散模型 / 图像生成  
**关键词**: 真实图像超分辨率, 自蒸馏, 频率感知, 扩散先验, 即插即用

## 一句话总结
FRAMER 提出频率对齐的自蒸馏训练框架，通过将最终层特征图作为教师监督中间层，并按低频/高频分别施加 IntraCL 和 InterCL 对比损失，配合自适应权重调节(FAW)和对齐门控(FAM)，在不改变网络结构和推理流程的情况下，显著提升扩散模型在真实图像超分辨率任务的高频细节恢复能力。

## 研究背景与动机

1. **领域现状**：真实图像超分辨率(Real-ISR)旨在从含混合未知退化的低分辨率图像恢复高分辨率图像。扩散模型已超越GAN成为主流方案，利用预训练T2I模型(如SD2的U-Net、SD3的DiT)的丰富先验是有前景的方向。
2. **现有痛点**：扩散模型在重建精细高频(HF)细节时表现不佳，容易产生过度平滑的结果。标准噪声预测损失对所有层和频率施加同一监督，忽略了模型内部的频率层级特性。
3. **核心矛盾**：作者追溯到一个根本性的低频(LF)偏置——来自两个特性：(a) 自然图像频率分布本身LF占主导，在LR输入中更严重，噪声预测损失偏向LF以降低整体损失；(b) 网络深度方向存在"低频优先、高频延后"的层级结构——LF特征在早期层即稳定，HF特征仅在最终层附近才收敛。
4. **本文目标** 如何在不改变推理架构的前提下，让训练过程中对LF和HF施加有针对性的监督，纠正LF偏置？
5. **切入角度**：自蒸馏——将最终层特征图作为教师、中间层作为学生。与外部频域损失相比，师生在同一特征空间中避免了域不匹配问题。
6. **核心 idea**：按频率分解自蒸馏信号，对LF施加样本内对比学习稳定结构，对HF施加跨样本对比学习锐化细节，并用自适应机制匹配模型内部频率层级。

## 方法详解

### 整体框架
FRAMER是纯训练策略，在标准噪声预测损失基础上增加辅助自蒸馏项。推理时使用原始backbone无任何修改。对每个去噪步，将最终层特征图视为教师，所有中间层为学生。师生特征图通过FFT掩码分解为LF/HF两个频带。LF频带用IntraCL(样本内对比损失)稳定全局结构；HF频带用InterCL(跨样本对比损失)锐化实例特定细节。FAW和FAM分别自适应调节各层各频带的蒸馏强度。

### 关键设计

1. **Intra Contrastive Loss (IntraCL) — 低频稳定**:

    - 功能：稳定LF频带的全局共享结构表示
    - 核心思路：对每个中间层 $i$，计算其LF表示 $\mathbf{F}_{LF}^{(i)}$ 与教师LF表示 $\mathbf{F}_{LF}^{(n)}$ 的余弦相似度作为正样本对，与随机采样的另一层 $j$ 的LF表示作为负样本对。损失形式为log-softmax：$\mathcal{L}_{IntraCL}^{(i)} = -\log \frac{\exp(s_{+,LF}^{(i)})}{\exp(s_{+,LF}^{(i)}) + \exp(s_{-,LF}^{(i)})}$。不使用跨样本负样本，因为LF特征跨样本相似度高，批内负样本会成为假阴性。
    - 设计动机：LF特征在训练样本间高度相似(共享结构信息)，使用批内负样本会引入假阴性。样本内对比足以通过层间差异推动学生向教师收敛。

2. **Inter Contrastive Loss (InterCL) — 高频锐化**:

    - 功能：锐化HF频带的实例特定细节表示
    - 核心思路：拉近学生HF表示与教师HF表示，同时推远两类负样本：(i) 同图像随机层HF表示(强制层间递进)；(ii) 批内其他图像的HF表示(鼓励实例判别)。$\mathcal{L}_{InterCL}^{(i)} = -\log \frac{\exp(s_{+,HF}^{(i)})}{\exp(s_{+,HF}^{(i)}) + \exp(s_{-,HF}^{(i)}) + S_{neg}^{(i)}}$。
    - 设计动机：HF特征跨样本相似度低(实例特异细节)，批内负样本是信息丰富的真阴性。直接对抗LF偏置，为缓慢收敛的HF组件提供定向优化信号。

3. **Frequency-based Adaptive Weight (FAW) — 自适应权重**:

    - 功能：根据各层各频带与教师的差异自适应调节蒸馏权重
    - 核心思路：计算每层LF/HF的FFT幅度均值 $E_{LF}^{(i)}$、$E_{HF}^{(i)}$，与最终层的相对差异 $\Delta^{(i)}$，权重为逆差异公式 $w^{(i)} = 1/(1+\Delta^{(i)})$。与教师差异小的频带获得更高权重，早期层LF权重高于HF。
    - 设计动机：匹配"低频优先、高频延后"的层级结构，避免给已收敛的LF层冗余梯度、给未成熟的HF层不足信号。

### 损失函数 / 训练策略
最终训练目标为 $\mathcal{L}_{total} = \mathcal{L}_{noise} + \sum_i \mathcal{L}_{FRAMER}^{(i)}$，其中FRAMER项为FAW和FAM门控后的IntraCL+InterCL加权和。FAM通过学生-教师对齐分数(经ReLU和stop-gradient)门控蒸馏强度，防止早期层不成熟时的崩溃。所有辅助头在测试时移除，零推理开销。

## 实验关键数据

### 主实验

| 数据集 | 指标 | FRAMER_U (Ours) | PiSA-SR (基线) | 提升 | FRAMER_D (Ours) | DiT4SR (基线) | 提升 |
|--------|------|-----------------|----------------|------|-----------------|---------------|------|
| DrealSR | PSNR↑ | 26.96 | 26.18 | +3.0% | 24.73 | 23.64 | +4.6% |
| DrealSR | SSIM↑ | 0.786 | 0.752 | +4.5% | 0.687 | 0.640 | +7.3% |
| DrealSR | LPIPS↓ | 0.333 | 0.368 | +9.5% | 0.412 | 0.442 | +6.8% |
| DrealSR | MANIQA↑ | 0.595 | 0.490 | +21.4% | 0.514 | 0.441 | +16.6% |
| RealSR | PSNR↑ | 24.81 | 24.02 | +3.3% | 23.23 | 21.94 | +5.9% |
| RealSR | MANIQA↑ | 0.484 | 0.412 | +17.5% | 0.564 | 0.459 | +22.9% |

### 消融实验
论文中消融验证了最终层教师和随机层负样本的有效性(详细数据在补充材料中)。核心发现：

| 配置 | 效果说明 |
|------|---------|
| 仅噪声预测损失(基线) | LF偏置严重，HF恢复不足 |
| + IntraCL | LF稳定性提升，结构更一致 |
| + InterCL | HF细节显著锐化 |
| + FAW | 层级感知权重分配，整体均衡提升 |
| + FAM | 防止早期层崩溃，训练更稳定 |

### 关键发现
- FRAMER在感知指标(MANIQA、MUSIQ)上提升最为显著，DrealSR上MANIQA提升21.4%，证实HF恢复能力大幅增强
- 跨U-Net和DiT两种架构均有效，验证了架构无关性
- 在更具挑战性的RealLR200和RealLQ250数据集上优势更明显
- 训练开销极小(仅增加辅助损失计算)，推理完全无开销

## 亮点与洞察
- **频率层级发现的深入性**：论文不仅指出LF偏置问题，还通过层级余弦相似度分析揭示了"低频优先、高频延后"的现象，为分频自蒸馏提供了有力的实验依据
- **LF/HF对比学习的差异化设计**：LF用样本内对比(避免假阴性)，HF用跨样本对比(利用真阴性)，这种基于特征跨样本相似度分析的差异化设计非常精巧
- **即插即用的实用价值**：不改变架构、不增加推理开销，可直接应用于任意扩散backbone的SR训练，实用性极强

## 局限与展望
- 频率分解使用固定的FFT二值掩码划分LF/HF，可能不是最优划分方式，可探索可学习的频率分割
- U-Net中需要额外1x1卷积和resize操作对齐特征维度，增加了集成复杂性
- 论文未探索在其他低级视觉任务(去噪、去模糊)上的效果
- FAW/FAM引入的超参数(如epsilon)可能需要针对不同backbone微调
- LF/HF二分法可能过于粗糙，三频或连续频率分解可能获得更好效果
- 消融实验的详细数据放在补充材料中，主文缺乏具体的组件增量数据

## 相关工作与启发
- **vs SeeSR**: SeeSR等方法对所有层和频率施加统一的噪声预测损失，未利用内部频率层级；FRAMER通过分频自蒸馏直接弥补此缺陷
- **vs 频率感知扩散方法**: 现有频率感知方法(FreeU等)依赖固定的推理时调制；FRAMER在训练阶段根据各层实际状态自适应调节监督
- **vs 自蒸馏方法**: 传统自蒸馏对齐整个特征图，隐式继承LF偏置；FRAMER通过频率分离显式对抗LF偏置

## 评分
- 新颖性: ⭐⭐⭐⭐ 频率对齐自蒸馏的框架设计新颖，但单个技术组件(对比学习、自适应权重)较为标准
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、六个指标、跨U-Net/DiT架构、详尽的消融分析
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，从观察到方法的逻辑链完整，图表直观
- 价值: ⭐⭐⭐⭐ 即插即用的训练策略具有广泛适用性，可直接提升现有SR方法

<!-- RELATED:START -->

## 相关论文

- [OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_processaware_online_alignment_for_generative.md)
- [Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](../../ECCV2024/image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)
- [Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](../../CVPR2025/image_generation/self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)

<!-- RELATED:END -->
