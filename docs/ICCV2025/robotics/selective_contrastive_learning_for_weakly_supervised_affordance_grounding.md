---
title: >-
  [论文解读] Selective Contrastive Learning for Weakly Supervised Affordance Grounding
description: >-
  [ICCV 2025][机器人][可供性定位] 提出选择性对比学习方法用于弱监督可供性定位，通过原型级对比学习和像素级对比学习，在目标和部件两个粒度上自适应学习可供性相关线索，有效避免模型关注与动作无关的显著特征，在 AGD20K 和 HICO-IIF 上全面超越了使用更强基础模型（GPT-4、LLAVA 等）的竞争方法。
tags:
  - "ICCV 2025"
  - "机器人"
  - "可供性定位"
  - "弱监督"
  - "对比学习"
  - "部件发现"
  - "CLIP"
---

# Selective Contrastive Learning for Weakly Supervised Affordance Grounding

**会议**: ICCV 2025  
**arXiv**: [2508.07877](https://arxiv.org/abs/2508.07877)  
**代码**: [https://github.com/hynnsk/SelectiveCL](https://github.com/hynnsk/SelectiveCL)  
**领域**: 机器人  
**关键词**: 可供性定位, 弱监督, 对比学习, 部件发现, CLIP

## 一句话总结

提出选择性对比学习方法用于弱监督可供性定位，通过原型级对比学习和像素级对比学习，在目标和部件两个粒度上自适应学习可供性相关线索，有效避免模型关注与动作无关的显著特征，在 AGD20K 和 HICO-IIF 上全面超越了使用更强基础模型（GPT-4、LLAVA 等）的竞争方法。

## 研究背景与动机

**领域现状**：弱监督可供性定位（WSAG）旨在模仿人类从第三人称示范中学习的方式，给定一张目标物体的自我中心（egocentric）图像和若干外部视角（exocentric）的人-物交互示例图像，定位物体上可供特定动作的部位。主流方法通过共享分类器学习跨视角的动作分类，结合蒸馏策略引入部件发现过程。

**现有痛点**：由于可供性相关部件并不总是容易区分的（如自行车座椅 vs 车轮），模型主要依赖分类任务，倾向于关注对分类有区分度但与可供性无关的公共模式。例如模型可能关注自行车的车架而非座椅（"ride" 动作的可供性区域）。蒸馏策略只在能可靠识别部件时才起作用，导致训练不连续，分类偏差加剧。

**核心矛盾**：可供性线索在弱标注下很难可靠提取，而分类目标天然偏向最具区分度的视觉特征——这两者不一定吻合。

**本文目标** 在部件线索不可靠时也能持续提供可供性学习信号，使模型注意力从不相关区域转移到真正的可供性区域。

**切入角度**：超越孤立的部件级学习，引入选择性的原型和像素对比目标——在信息粒度可靠时做部件级学习，信息不可靠时退而求其次做目标级学习，保证训练信号的连续性。

**核心 idea**：用 CLIP 发现目标物体、用跨视角交叉引用提取可供性部件，然后通过选择性对比学习在可靠/不可靠两种情况下都持续区分可供性相关和无关区域。

## 方法详解

### 整体框架

整体流程：egocentric 和 exocentric 图像分别经过 DINO 编码器和投影层提取特征，然后分两条路径：
- **分类分支**：共享分类器在两种视角上学习动作分类，导出 CAM 用于定位
- **对比学习分支**：通过原型对比学习（利用 exocentric 线索）和像素对比学习（利用 egocentric 线索）引导模型关注可供性区域

推理时仅使用 egocentric 图像和动作文本提示，通过CAM 生成定位图并经目标亲和图校准。

### 关键设计

1. **目标发现（Object Discovery）**:

    - 功能：在 egocentric 和 exocentric 图像中定位与动作相关的目标物体
    - 核心思路：利用 CLIP 视觉编码器提取特征，采用 ClearCLIP 策略增强局部判别力，计算特征与动作文本提示的余弦相似度得到目标亲和图 $A_{\text{obj}}^{\text{ego}} \in \mathbb{R}^{B \times H \times W}$ 和 $A_{\text{obj}}^{\text{exo}} \in \mathbb{R}^{B \times E \times H \times W}$
    - 设计动机：目标亲和图提供了粗粒度但可靠的目标定位，作为后续部件发现的基础。即使部件发现失败，也能退回到目标级学习

2. **选择性原型对比学习（Selective Prototypical Contrastive Learning）**:

    - 功能：在 exocentric 视角中发现可供性部件线索，通过对比学习蒸馏到 egocentric 表示
    - 核心思路：首先在 exocentric 图像中发现部件——将 CAM 与目标亲和图结合后阈值化提取交互区域，K-means（K=3）聚类得候选部件质心，与 DINO 自注意力图比较（pIoU 指标）判断可靠性。然后构建正/负原型：
        - 可靠时：$P^+ = \Phi^+(F, A_{\text{part}})$（部件级正原型），anchor 为 egocentric 目标区域特征
        - 不可靠时：$P^+ = \Phi^+(F, A_{\text{obj}})$（目标级正原型），anchor 为全图特征
    - 对比损失 $\mathcal{L}^{\text{proto}}_b = \frac{-1}{|\mathbf{P}^+_b|} \sum_{p \in \mathbf{P}^+_b} \log \frac{\exp(z^{\text{ego}}_b \circ p / \tau)}{\sum_{n \in (\mathbf{P}^+_b \cup \mathbf{P}^-_b)} \exp(z \circ n / \tau)}$
    - 设计动机：相比之前的 pairwise 蒸馏（LOCATE），原型对比学习不仅拉近 ego-exo 表示，还同时推开不同动作类的原型和背景，学到更具判别性的特征。选择性机制保证即使部件不可靠也能学到目标级信号

3. **选择性像素对比学习（Selective Pixel Contrastive Learning）**:

    - 功能：在 egocentric 图像中直接区分可供性相关像素和无关像素
    - 核心思路：利用 CLIP 对显著物体更敏感的特性，通过 exocentric 中目标亲和图的最大响应值来设定阈值 $\rho = \min_{e \in E} \max_{h,w} A_{\text{obj}}^{\text{exo}}$。超过 $\rho$ 的 egocentric 像素被标记为可供性正样本 $Q^+$，其余为负样本 $Q^-$。像素对比损失 $\mathcal{L}^{\text{pix}}_b = \frac{-1}{|Q^+_b|^2} \sum_{z \in Q^+_b} \sum_{p \in Q^+_b} \log \frac{\exp(z \circ p / \tau)}{\sum_{n \in (Q^+_b \cup Q^-_b)} \exp(z \circ n / \tau)}$
    - 设计动机：原型对比学习提供全局引导，但对每个像素仅有隐式监督。像素对比学习直接在细粒度层面区分像素的可供性相关性，补充了精细定位能力。选择性机制在 exocentric 图像清晰对焦时降级到目标级学习

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}^{\text{ce}} + \lambda_1 \mathcal{L}^{\text{proto}} + \lambda_2 \mathcal{L}^{\text{pix}}$，其中 $\lambda_1 = \lambda_2 = 1$。使用 DINO ViT-S/16 作为主干，CLIP ViT-B/16 提供目标亲和图。推理时额外进行 CAM 校准：将二值化的目标亲和图与 CAM 做 Hadamard 积，将激活限制在目标区域内，避免卷积感受野导致的激活扩散。

## 实验关键数据

### 主实验

AGD20K 和 HICO-IIF 数据集上的性能对比：

| 方法 | 模型 | AGD20K-Seen KLD↓ | AGD20K-Unseen KLD↓ | HICO-IIF KLD↓ |
|------|------|------------------|-------------------|--------------|
| Cross-view-AG | ResNet50 | 1.538 | 1.787 | 1.779 |
| LOCATE | DINO | 1.226 | 1.405 | 1.593 |
| WSMA | DINO+CLIP | 1.176 | 1.335 | 1.465 |
| WorldAfford | DINO+CLIP+SAM+GPT-4 | 1.201 | 1.393 | - |
| INTRA | DINOv2+ALBEF+GPT-4 | 1.199 | 1.365 | - |
| **本文** | **DINO+CLIP** | **1.124** | **1.243** | **1.358** |

### 消融实验

| 配置 | AGD20K-Seen KLD↓ | SIM↑ | NSS↑ | 说明 |
|------|------------------|------|------|------|
| (a) 基线（仅分类） | 1.349 | 0.365 | 1.138 | 无对比学习 |
| (c) +目标级原型对比 | 1.271 | 0.392 | 1.153 | 推开背景 |
| (f) +部件级原型对比 | 1.164 | 0.416 | 1.290 | 定位更精细 |
| (h) +目标+部件级像素对比 | 1.142 | 0.415 | 1.303 | 进一步细化 |
| (i) 完整模型+校准 | 1.124 | 0.433 | 1.280 | 最优，KLD 降 16.7% |

### 关键发现

- **在 Unseen 场景提升最为显著**：KLD 从 LOCATE 的 1.405 降到 1.243（降 11.5%），说明对比学习增强了泛化性。论文归因于对比目标显式地将注意力从背景转向可供性区域，在面对未见物体时尤其有效
- **不依赖重型基础模型也能超越**：本文仅用 DINO+CLIP，超越了使用 GPT-4、SAM、LLAVA 等大型模型的方法（WorldAfford、AffordanceLLM、INTRA），说明学习范式比模型规模更重要
- **目标级学习是基础，部件级学习是锦上添花**：从消融看，目标级对比（c）带来的 KLD 降幅（0.078）略低于部件级对比（f 相对 c）的降幅（0.107），但两者结合效果最佳
- CAM 校准步骤提升明显（KLD 从 1.142 降到 1.124），说明约束激活在物体边界内很重要

## 亮点与洞察

- **选择性机制设计巧妙**：不强求每次训练都有准确的部件监督，而是在不可靠时优雅降级到目标级学习。这种「有就精做、没有就粗做、但始终在学」的策略比之前只在可靠时学习的间歇式训练稳定得多
- **利用 CLIP 的「偏见」反而有效**：CLIP 对显著物体的响应更强被利用来区分 egocentric 和 exocentric 中的可供性区域——在 exocentric 中物体小且被遮挡导致响应弱，这个差异恰好可以作为部件发现的信号
- **可迁移到其他弱监督定位任务**：选择性多粒度对比学习的框架可以用于任何需要在不同可靠度的监督下交替学习的场景
- 目标亲和图校准 CAM 的后处理步骤简单有效，是一个可复用的 trick

## 局限与展望

- 阈值 $\alpha$、$\gamma$ 等超参数固定为 0.6，可能对不同数据分布不一定最优
- K-means K=3 的选择假设交互区域由背景、可供性部件和其他元素三部分组成，对于复杂交互场景可能过于简化
- 部件发现依赖 DINO 的自注意力图质量，对于纹理复杂或遮挡严重的场景可能不够准确
- 仅在 egocentric 图像中做像素对比学习，未对 exocentric 进行像素级优化
- 未探索视频形式的动态可供性预测

## 相关工作与启发

- **vs LOCATE**：LOCATE 仅在可靠部件可用时从 exocentric 蒸馏到 egocentric，训练不连续。本文扩展到两个视角的双向线索利用 + 始终有学习信号
- **vs WSMA**：WSMA 通过 CLIP 语义注意力机制解决分类标签表达力不足的问题，但仍依赖分类驱动。本文引入对比学习显式推开背景和非可供性部件
- **vs WorldAfford/INTRA**：这些方法依赖 GPT-4 等大模型获取部件知识，成本高且难以端到端优化。本文用更轻量的方式达到更好效果

## 评分

- 新颖性: ⭐⭐⭐⭐ 选择性多粒度对比学习框架是对 WSAG 范式的有意义改进，思路清晰且通用
- 实验充分度: ⭐⭐⭐⭐ 在标准 benchmark 上全面超越 SOTA，消融充分，但缺少更多真实机器人应用场景的验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清楚，图示丰富，但公式较多需要仔细阅读
- 价值: ⭐⭐⭐⭐ 为弱监督可供性理解提供了更稳健的学习范式，对机器人交互感知有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Weakly-Supervised Learning of Dense Functional Correspondences](weakly-supervised_learning_of_dense_functional_correspondences.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)
- [\[CVPR 2025\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](../../CVPR2025/robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)

</div>

<!-- RELATED:END -->
