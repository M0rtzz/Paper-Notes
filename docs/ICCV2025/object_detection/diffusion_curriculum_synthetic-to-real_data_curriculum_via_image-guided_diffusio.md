---
title: >-
  [论文解读] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion
description: >-
  [ICCV 2025][目标检测][扩散模型] 利用扩散模型的图像引导强度控制生成从合成到真实的连续谱系数据，设计"扩散课程学习（DisCL）"策略在训练不同阶段自适应选择最优引导级别的合成数据，有效解决长尾分类和低质量数据学习问题。
tags:
  - ICCV 2025
  - 目标检测
  - 扩散模型
  - 课程学习
  - 合成数据
  - 长尾分类
  - 低质量数据
---

# Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion

**会议**: ICCV 2025  
**arXiv**: [2410.13674](https://arxiv.org/abs/2410.13674)  
**代码**: [tianyi-lab/DisCL](https://github.com/tianyi-lab/DisCL)  
**领域**: Object Detection / Data Augmentation  
**关键词**: 扩散模型, 课程学习, 合成数据, 长尾分类, 低质量数据

## 一句话总结

利用扩散模型的图像引导强度控制生成从合成到真实的连续谱系数据，设计"扩散课程学习（DisCL）"策略在训练不同阶段自适应选择最优引导级别的合成数据，有效解决长尾分类和低质量数据学习问题。

## 研究背景与动机

深度学习在实际场景中常面临数据质量和数量的双重挑战：野生动物相机、交通摄像头等设备采集的图像存在光照差、运动模糊、遮挡等问题；类别不平衡导致模型在尾部类上表现极差。

现有解决方案的不足：
- 传统数据增强（翻转、裁剪等）产生的新样本差异有限
- 文本引导的扩散模型虽能生成高质量多样数据，但**仅靠文本控制无法保证合成图像与原始图像的相似性**，导致分布外数据反而损害模型性能
- 已有方法（如ALIA、LDMLR）在合成-真实分布差距控制上仍有缺陷

核心洞察：**扩散模型中的图像引导强度 $\lambda$ 天然提供了合成到真实的连续插值谱**。低 $\lambda$ 生成多样且原型化的简单图像，高 $\lambda$ 生成与原始图像相似但可能继承原始缺陷的图像。

## 方法详解

### 整体框架

DisCL 包含两个阶段：
1. **合成-真实数据生成**：识别困难样本，利用图像引导级别 $\lambda \in [0,1)$ 生成完整的合成-真实插值谱
2. **生成式课程学习**：根据训练阶段需求，选择对应引导级别的合成数据进行训练

### 关键设计

1. **图像引导控制的合成-真实插值**：

    - 基于 Stable Diffusion XL，修改 Classifier-free guidance 中的初始扩散步 $t(\lambda) = \lfloor(1-\lambda)T\rfloor$
    - 噪声初始化公式：$z_{t(\lambda)} = \sqrt{\tilde{\alpha}_{t(\lambda)}} z_{real} + \sqrt{1-\tilde{\alpha}_{t(\lambda)}} \epsilon$
    - $\lambda = 0$ 对应纯文本引导（最多样但最远离原始）；$\lambda \rightarrow 1$ 对应接近原始图像（最相似但多样性最低）
    - 通过 CLIPScore 阈值过滤低保真度合成图像（目标对象缺失或被遮挡的情况）

2. **长尾分类的非自适应课程（Diverse-to-Specific）**：

    - 核心思路：尾部类数据稀缺，先用低引导级别数据增加多样性和数量，再逐步提高引导级别让模型适应真实分布
    - 训练初期使用 $\lambda \rightarrow 0$ 的合成数据（原型化特征，高多样性）
    - 训练后期渐进切换到 $\lambda \rightarrow 1$ 的数据（贴近真实分布）
    - 逐步弥合合成-真实差距，避免分布突变

3. **低质量数据的自适应课程（Adaptive）**：

    - 困难样本通过预训练分类器的真实类概率判定（概率越低越困难）
    - 每个 epoch 根据"学习进展"（验证子集上真实类置信度提升量）自适应选择下一轮的引导级别 $\lambda$
    - 选择使当前训练阶段进展最大的 $\lambda$，确保在每个阶段学习最有信息量的数据
    - 避免非自适应课程在低质量场景下因过早引入分布外数据导致的负迁移

### 训练策略

- 困难样本识别：长尾任务基于类别频次；低质量任务基于分类器置信度
- 合成数据量级：长尾任务中尾部类合成数据为原始的 3-4 倍效果最佳
- 训练骨干：ImageNet-LT 使用 ResNet-10；iWildCam 使用 CLIP ViT-B/16 和 ViT-L/14
- 使用 DDIM 作为噪声调度器

## 实验关键数据

### 主实验：ImageNet-LT 长尾分类

| 方法 | 课程策略 | Many | Medium | Few | Overall |
|------|---------|------|--------|-----|---------|
| CE baseline | N/A | 57.70% | 26.60% | 4.40% | 35.80% |
| CE + CUDA | N/A | 57.49% | 28.16% | 6.58% | 36.30% |
| CE + LDMLR | N/A | 57.20% | 29.20% | 7.30% | 37.20% |
| BS + CUDA | N/A | 51.16% | 37.35% | 19.28% | 40.03% |
| **CE + DisCL** | Diverse-to-Specific | 56.78% | 30.73% | **23.64%** | 39.82% |
| **BS + DisCL** | Diverse-to-Specific | 52.68% | 37.68% | 21.36% | **41.33%** |

DisCL 将尾部类准确率从 4.40% 提升至 23.64%（+19.24%），整体提升 4.02%。

### 主实验：iWildCam 低质量数据分类

| 方法 | 课程策略 | OOD F1 | ID F1 |
|------|---------|--------|-------|
| FLYP | N/A | 35.5 | 52.2 |
| FLYP + ALIA | N/A | 36.9 | 52.6 |
| **FLYP + DisCL** | Adaptive | **38.2** | **54.3** |
| **FLYP + DisCL + WE** | Adaptive | **38.7** | **54.6** |

OOD 和 ID 分别提升 2.7% 和 2.1%。

### 消融实验

| 配置 | Few-class (IN-LT) | 说明 |
|------|-------------------|------|
| Text-only Guidance (λ=0) | 17.90% | 仅文本引导，多样但分布偏差大 |
| All-Level Guidance | 19.17% | 所有级别混合，无课程策略 |
| DisCL Specific-to-Diverse | 18.36% | 反向课程，易过拟合真实分布 |
| DisCL Adaptive | 16.78% | 自适应在长尾场景下验证集太小不适用 |
| **DisCL Diverse-to-Specific** | **23.64%** | 最优策略，渐进弥合分布差距 |

| 配置 | OOD F1 (iWildCam) | 说明 |
|------|-------------------|------|
| Easy-to-Hard (非自适应) | 35.2 | 低引导到高引导的固定策略 |
| Random | 35.9 | 随机选择引导级别 |
| **Adaptive** | **38.2** | 根据学习进展自适应选择 |

### 关键发现

- 合成数据缩放实验表明，尾部类 3-4 倍合成数据是最优平衡点，超过后头部类准确率轻微下降
- Diverse-to-Specific 和 Adaptive 策略分别最适合长尾和低质量场景，说明**课程策略需与任务特性匹配**
- CLIPScore 阈值在低质量数据任务中影响显著（合成图像质量方差大），在 ImageNet 中影响较小（Stable Diffusion 对 ImageNet 类别生成质量稳定）

## 亮点与洞察

- 通过图像引导级别控制合成-真实插值的思路极具通用性，将扩散模型从"数据增强工具"提升为"可控课程学习引擎"
- 两种课程策略（非自适应 vs 自适应）的设计反映了对不同任务本质差异的深刻理解
- 实验设计系统全面，覆盖 ImageNet-LT、CIFAR100-LT、iNaturalist2018、iWildCam 四个数据集

## 局限与展望

- 合成数据质量受扩散模型能力和 CLIP 对齐能力制约
- 文本提示仅基于类别名称，未利用图像描述信息
- 合成图像中目标对象的位置和尺寸与真实图像的差异会加大分布差距
- 未探索更复杂的检测/分割任务，仅在分类任务上验证

## 相关工作与启发

- **CUDA** 是最早将课程学习与数据增强结合用于长尾学习的工作，但仅使用工程化增强
- **LDMLR** 使用扩散模型训练专门的长尾采样器，DisCL 则直接利用预训练扩散模型的图像引导
- 与对比学习增强（如 DoCL）的区别在于 DisCL 利用生成模型而非真实样本的动态选择

## 评分

- 新颖性: ⭐⭐⭐⭐ 图像引导级别 × 课程学习的交叉创新简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集 + 详尽消融 + 多种损失函数验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰，公式化表达规范，图示易懂
- 价值: ⭐⭐⭐⭐ 为利用扩散模型增强数据提供了通用框架，具有广泛应用前景

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)
- [\[ICCV 2025\] DiffDoctor: Diagnosing Image Diffusion Models Before Treating](diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](../../CVPR2026/object_detection/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[CVPR 2025\] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](../../CVPR2025/object_detection/generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)

<!-- RELATED:END -->
