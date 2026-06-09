---
title: >-
  [论文解读] Towards General Visual-Linguistic Face Forgery Detection
description: >-
  [CVPR 2025][AI安全][深度伪造检测] VLFFD 提出了一种视觉-语言范式的深度伪造检测方法，通过 Prompt Forgery Image Generator (PFIG) 自动生成带有细粒度文本描述的混合伪造图像…
tags:
  - "CVPR 2025"
  - "AI安全"
  - "深度伪造检测"
  - "视觉-语言"
  - "细粒度提示"
  - "多模态"
  - "泛化性"
---

# Towards General Visual-Linguistic Face Forgery Detection

**会议**: CVPR 2025  
**arXiv**: [2307.16545](https://arxiv.org/abs/2307.16545)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 深度伪造检测, 视觉-语言, 细粒度提示, 多模态, 泛化性

## 一句话总结
VLFFD 提出了一种视觉-语言范式的深度伪造检测方法，通过 Prompt Forgery Image Generator (PFIG) 自动生成带有细粒度文本描述的混合伪造图像，再用 Coarse-and-Fine Co-training (C2F) 框架联合训练粗粒度和细粒度数据，显著提升了检测模型的泛化性和可解释性。

## 研究背景与动机

**领域现状**：深度伪造（Deepfake）技术的快速发展对安全、隐私和信任构成严重威胁。现有的人脸伪造检测方法大多将任务定义为二分类问题（真实 vs 伪造），使用数字标签（0/1）或掩码信号来训练检测模型。

**现有痛点**：(1) 语义信息缺失——二分类标签只告诉模型"是不是假的"，不告诉"哪里假、怎么假"，导致模型过拟合训练数据中的特定伪造痕迹而缺乏泛化性；(2) 可解释性差——部署在实际场景中的检测系统需要能解释"为什么判定为伪造"，但基于二分类的方法无法提供这种解释；(3) 跨方法泛化困难——在 FaceSwap 上训练的检测器很难泛化到 FaceShifter 等未见过的伪造方法。

**核心矛盾**：现有监督信号（二分类标签/掩码）太粗粒度，缺乏语义信息，导致模型只能学到表面统计特征而非伪造的本质模式。但手动为大量伪造样本撰写细粒度文本描述又不现实。

**本文目标**：(1) 设计一种方法自动为伪造图像生成细粒度的语言描述作为监督信号；(2) 在不增加额外人工标注成本的情况下，利用语言监督提升检测模型的泛化性和可解释性。

**切入角度**：既然现有数据集没有文本标注，那就自己创造——通过可控的伪造图像生成器，在生成伪造图像的同时自动生成对应的文本描述。这样做的关键洞察是：当你知道伪造操作的参数（换了哪个区域、用了什么方法、混合程度如何），就能自动构造准确的语言描述。

**核心 idea**：用"已知操作参数→自动生成文本标注"的方式创造视觉-语言伪造检测数据，然后通过粗细粒度联合训练，让模型同时学习通用的二分类能力和细粒度的语义理解能力。

## 方法详解

### 整体框架
VLFFD 框架包含两个核心阶段：(1) **数据生成阶段**——PFIG（Prompt Forgery Image Generator）接受真实人脸图像，通过可控的混合伪造操作生成伪造图像，并自动生成对应的句子级文本提示（prompt），形成细粒度的图像-文本对；(2) **联合训练阶段**——C2F（Coarse-and-Fine Co-training）框架同时使用原始数据集的粗粒度标注（二分类）和 PFIG 生成的细粒度数据（图像-文本对）来训练视觉-语言检测模型。

### 关键设计

1. **Prompt Forgery Image Generator (PFIG)**:

    - 功能：可控地生成伪造人脸图像，并自动产生对应的细粒度文本描述
    - 核心思路：PFIG 从真实人脸图像出发，执行一系列可控的伪造操作。具体来说，首先通过人脸解析（face parsing）将人脸分割为多个语义区域（眼睛、鼻子、嘴巴、皮肤等）。然后随机选择一个或多个区域进行替换——从另一个身份的人脸对应区域提取内容，通过可控的混合系数 $\alpha$ 进行融合：$I_{forge} = \alpha \cdot I_{source}^{region} + (1-\alpha) \cdot I_{target}^{region}$。关键在于，由于所有操作参数（替换区域、源身份、混合系数）都是已知的，可以自动构造精确的文本描述，如"The eyes and nose regions are replaced from another identity with a blending ratio of 0.7, showing subtle boundary artifacts around the nose bridge"。
    - 设计动机：传统数据增强只能增加图像多样性，PFIG 同时增加了标注的语义丰富度。已知操作参数→自动生成文本的思路避免了昂贵的人工标注。

2. **Coarse-and-Fine Co-training (C2F) 框架**:

    - 功能：统一框架中联合利用粗粒度（二分类）和细粒度（图像-文本对）监督信号
    - 核心思路：C2F 框架包含一个共享的视觉编码器和两个分支：**粗粒度分支**接受原始数据集的图像及二分类标签，通过标准分类头进行真/假判别；**细粒度分支**接受 PFIG 生成的图像-文本对，使用对比学习（类似 CLIP 的 image-text matching）训练视觉编码器理解伪造区域的语义特征。两个分支共享视觉编码器但使用各自的训练目标，通过加权损失联合优化：$L = L_{cls} + \beta \cdot L_{contrastive}$。粗粒度分支保证基本的检测能力，细粒度分支注入语义理解能力以提升泛化性。
    - 设计动机：单独用细粒度数据训练会因为 PFIG 生成的伪造方式有限而导致偏差，单独用粗粒度数据又缺乏语义信息。C2F 通过联合训练让两种信息互补，粗粒度保底、细粒度提升。

3. **与多模态大模型的集成**:

    - 功能：将 VLFFD 范式扩展到多模态大语言模型（MLLM），进一步提升可解释性
    - 核心思路：将 PFIG 生成的图像-文本对作为指令微调（instruction tuning）数据，对多模态大模型进行适配。微调后的 MLLM 不仅能判断图像真假，还能生成自然语言解释说明伪造的具体位置和方式。这为伪造检测提供了前所未有的可解释性。
    - 设计动机：MLLM 的文本生成能力使得检测结果可以直接以人类可理解的方式呈现，这对法律取证、内容审核等应用场景尤为重要。

### 损失函数 / 训练策略
粗粒度分支使用标准二分类交叉熵损失 $L_{cls}$，细粒度分支使用 InfoNCE 对比损失 $L_{contrastive}$，总损失为两者的加权和。训练分两阶段：先在 PFIG 数据上预训练细粒度分支，再联合微调整个框架。数据增强包括 PFIG 中随机化的伪造区域数量、混合系数和源身份。

## 实验关键数据

### 主实验：跨方法泛化性（AUC %）

| 训练数据 | 方法 | FF++ (同域) | Celeb-DF | DFDC | DeeperForensics | 平均跨域 |
|----------|------|------------|----------|------|-----------------|----------|
| FF++ | Xception | 99.1 | 73.2 | 67.8 | 72.5 | 71.2 |
| FF++ | RECCE | 99.3 | 76.5 | 70.1 | 75.8 | 74.1 |
| FF++ | SBI | 98.8 | 79.3 | 72.6 | 77.2 | 76.4 |
| FF++ | VLFFD (Ours) | **99.5** | **84.7** | **76.3** | **82.1** | **81.0** |
| FF++ | VLFFD + MLLM | 99.2 | **86.1** | **78.8** | **83.5** | **82.8** |

### 消融实验

| 配置 | FF++ AUC | Celeb-DF AUC | DFDC AUC | 说明 |
|------|----------|-------------|----------|------|
| 仅粗粒度（Baseline） | 99.1 | 73.2 | 67.8 | 标准二分类训练 |
| 仅细粒度（PFIG only） | 97.5 | 80.3 | 73.1 | 语义理解强但判别力稍弱 |
| C2F 联合训练 | **99.5** | **84.7** | **76.3** | 粗细互补效果最佳 |
| w/o 区域级伪造 | 99.2 | 79.8 | 72.4 | 仅做全脸替换，缺少局部操作多样性 |
| w/o 混合系数随机化 | 99.3 | 82.1 | 74.5 | 固定混合系数降低了数据多样性 |
| w/o 对比学习 | 99.4 | 78.6 | 71.9 | 不用对比学习，细粒度信息无法有效注入 |

### 关键发现
- **跨域泛化提升显著**：VLFFD 在三个未见数据集上平均 AUC 提升约 5-9 个百分点（81.0% vs 76.4% SBI），证明语言监督确实帮助模型学到了更本质的伪造特征
- **C2F 联合训练是关键**：单独用细粒度数据反而在同域指标上略有下降（因为 PFIG 生成的伪造方式毕竟有限），但联合训练实现了两全其美
- **对比学习是细粒度信息注入的关键桥梁**：去掉对比学习后跨域 AUC 明显下降，说明简单的多任务学习不如对比学习有效
- **MLLM 集成进一步破圈**：虽然定量提升在 1-2%，但质的突破在于模型能生成人类可读的检测解释

## 亮点与洞察
- **"已知操作参数→自动生成标注"的数据构造范式**：这个思路不限于伪造检测，任何有可控合成过程的任务都可以借鉴——比如图像编辑检测（知道编辑了什么就能写出描述）、数据篡改检测等
- **粗细粒度联合训练**是一种实用的信号利用策略：现有粗标注不浪费，新生成的细标注锦上添花，降低了方法落地的门槛
- **首次将 MLLM 引入伪造检测并展示了可解释性潜力**，为该领域指明了从"检测"到"解释"的发展方向

## 局限与展望
- PFIG 的伪造方式（区域替换+混合）相对简单，未涵盖 GAN 和扩散模型生成的高质量伪造
- 文本描述的模板化程度较高，缺乏对微妙视觉伪影的自然语言描述能力
- 视频维度未涉及——目前只处理单帧图像，时序伪影线索未利用
- 与 MLLM 的集成目前只是初步探索，推理速度远不能满足实时检测需求

## 相关工作与启发
- **vs SBI (Self-Blended Images)**: SBI 也通过自造伪造数据来训练检测器，但只使用二分类标注。VLFFD 在此基础上增加了语言监督维度，泛化性更强
- **vs CLIP-based 方法**: 一些工作尝试直接用 CLIP 特征做伪造检测，但缺乏针对伪造领域的细粒度对齐。VLFFD 通过 PFIG 生成领域特定的图像-文本对，实现了更有效的对齐
- **vs Face X-ray**: Face X-ray 关注混合边界检测，VLFFD 不仅检测是否存在伪造，还能描述伪造的方式和位置，信息维度更丰富

## 评分
- 新颖性: ⭐⭐⭐⭐ 将视觉-语言范式引入伪造检测是有创见的方向，PFIG 的设计简单有效
- 实验充分度: ⭐⭐⭐⭐ 多个跨域benchmark验证，消融全面，MLLM集成有加分
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰，从问题到方案的推导自然
- 价值: ⭐⭐⭐⭐ 为伪造检测开辟了语言监督的新方向，与MLLM的结合具有前瞻性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection](forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection.md)
- [\[CVPR 2025\] Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection](stacking_brick_by_brick_aligned_feature_isolation_for_incremental_face_forgery_d.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[CVPR 2025\] PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection](psbd_prediction_shift_uncertainty_unlocks_backdoor_detection.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
