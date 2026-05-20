---
title: >-
  [论文解读] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][部件级共分割] 提出 Calico——首个面向部件级语义共分割的大视觉语言模型，通过对应关系提取模块（CEM）和对应关系适配模块（CAM）在多图像间建立部件级语义对应，仅微调 0.3% 参数就在新构建的 MixedParts 基准上全面超越现有方法…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "部件级共分割"
  - "多图像推理"
  - "语义对应"
  - "大视觉语言模型"
  - "参数高效适配"
---

# Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.19331](https://arxiv.org/abs/2412.19331)  
**代码**: [https://plan-lab.github.io/calico](https://plan-lab.github.io/calico)  
**领域**: 多模态VLM  
**关键词**: 部件级共分割, 多图像推理, 语义对应, 大视觉语言模型, 参数高效适配

## 一句话总结

提出 Calico——首个面向部件级语义共分割的大视觉语言模型，通过对应关系提取模块（CEM）和对应关系适配模块（CAM）在多图像间建立部件级语义对应，仅微调 0.3% 参数就在新构建的 MixedParts 基准上全面超越现有方法，mIoU 提升 6.3%、推理加速 51.3%。

## 研究背景与动机

**领域现状**：大视觉语言模型（LVLM）在单图像分割任务上已取得显著进展。LISA、GLaMM 等模型通过在 LLM 中引入分割 token [SEG]，实现了基于文本指令的像素级分割。同时，DINOv2 等自监督 ViT 已被证明能捕捉跨类别的部件级语义对应关系。

**现有痛点**：（1）现有 LVLM 分割模型仅支持单图像操作，无法进行多图像间的比较推理和共分割；（2）部件共分割的现有方法（SCOPS、DFF 等）无法生成语义标签或处理独有部件——它们只能找到共享结构，无法命名或区分独有特征；（3）现有部件分割模型（PartGLEE、VLPart 等）需要用户逐一指定每个部件名称才能分割，无法自动发现和比较多图像中的共有/独有部件。

**核心矛盾**：在多图像场景中同时实现"定位+比较+命名"三合一的部件级共分割，需要模型具备跨图像的语义对应能力和开放式标签生成能力，而这两种能力在现有架构中是分离的。

**本文目标**：定义"部件聚焦语义共分割"新任务——给定多张包含相似物体的图像，自动分割并标注出共有物体、共有部件和独有部件。

**切入角度**：DINOv2 天然具有跨类别的部件级语义对应能力，LVLM 天然具有开放式文本生成和推理能力——如果能将两者融合，就能解决"定位+比较+命名"的三重需求。

**核心 idea**：用 DINOv2 提取部件级语义对应特征注入冻结的 LVLM，通过参数高效的适配模块让 LVLM 学习多图像间的部件级推理和分割。

## 方法详解

### 整体框架

Calico 基于 GLaMM 架构扩展，包含以下 pipeline：多张输入图像分别通过 EVA-CLIP 视觉编码器和 DINOv2 编码器提取全局/语义特征，经 Q-Former 压缩后以交错方式送入 Vicuna-7B LLM。LLM 输出包含 [SEG] token 的文本，[SEG] token 经投影后送入 SAM 解码器生成分割掩码。整体可训练参数仅约 29M，占总参数的 0.3%。

### 关键设计

1. **对应关系提取模块（Correspondence Extraction Module, CEM）**:

    - 功能：将 DINOv2 的部件级语义对应信息融入 EVA-CLIP 的全局视觉特征
    - 核心思路：给定输入图像，分别通过冻结的 EVA-CLIP 获得全局嵌入 $\mathbf{X}_{\text{global}}$，通过冻结的 DINOv2 获得语义嵌入 $\mathbf{X}_{\text{semantic}}$。然后用交叉注意力机制将语义嵌入作为 Key-Value、全局嵌入作为 Query 进行融合：$\mathbf{X}_{\text{global}}' = \mathcal{A}(\mathbf{X}_{\text{global}}, \mathbf{X}_{\text{semantic}})$。输出的增强特征兼具 CLIP 的全局识别能力和 DINOv2 的部件级语义对应能力
    - 设计动机：DINOv2 通过自监督训练学到的特征在部件级粒度上具有强语义对应性（如不同类别物体的"腿"会有相似激活），但缺乏与语言空间的对齐；EVA-CLIP 有强全局特征但缺乏细粒度对应。CEM 精准互补两者

2. **对应关系适配模块（Correspondence Adaptation Module, CAM）**:

    - 功能：将 CEM 输出的语义丰富特征注入 LLM 的中间层，实现指令引导的多图像理解
    - 核心思路：在 LLM 的第 11 层和第 22 层（32 层 LLM 的 1/3 和 2/3 处）各放置一个 CAM。每个 CAM 先将当前层最后一个文本 token（承载指令信息）线性投影为引导嵌入，加到 Q-Former 的 learnable query 上形成上下文引导的 query $\mathbf{q}' = \mathbf{q} + f_{\text{adaptation}}(\mathbf{t}_{S_T}^l)$，再用该 query 从 CEM 增强后的视觉特征中提取信息，最终将提取结果投影回语言空间并加到 LLM 对应层的视觉 token 上
    - 设计动机：两层 CAM 分别在 LLM 不同深度注入语义对应信息，鼓励模型在不同粒度（物体级和部件级）学习跨图像对应关系。消融实验证实这比 1 层或 3 层配置效果更好

3. **Q-Former 视觉压缩 + 交错多图像输入**:

    - 功能：将每张图像的视觉 token 从 256/576 压缩到 32 个，支持高效的多图像处理
    - 核心思路：借鉴 BLIP-2，用 Q-Former 的 32 个 learnable query 通过交叉注意力机制从 EVA-CLIP 特征中提取紧凑的视觉嵌入。多张图像的嵌入以交错方式（image1 tokens + text + image2 tokens + text）送入 LLM，每张图像用唯一标识符（IMAGE1、IMAGE2）区分
    - 设计动机：多图像分割需要处理多组视觉 token，直接使用原始 token 数量（如 GLaMM 的 576 token/图像）会导致计算量爆炸。Q-Former 将 token 数压缩 8-18 倍，使 TFLOPS 降低 32.6%、推理提速 51.3%

### 损失函数 / 训练策略

训练损失为文本损失和分割损失的加权组合：$\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \mathcal{L}_{\text{mask}}$。其中 $\mathcal{L}_{\text{text}}$ 是标准的 causal LM 交叉熵损失；$\mathcal{L}_{\text{mask}} = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}} + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}$。超参数设为 $\lambda_{\text{text}}=1.0$、$\lambda_{\text{focal}}=2.0$、$\lambda_{\text{Dice}}=0.5$。使用 LoRA（rank=8, alpha=16）进行参数高效训练，学习率 4e-4，AdamW 优化器，4 张 A40 GPU 训练 10 个 epoch。

## 实验关键数据

### 主实验

| 方法 | AP50↑ | mIoU↑ | Recall↑ | SS↑ | S-IoU↑ |
|------|-------|-------|---------|-----|--------|
| Cascade (Sparkles+GPT4o+LISA) | 5.7 | 27.9 | 19.0 | 32.2 | 14.8 |
| Multi-Image PartGLEE | 1.2 | 29.3 | 9.7 | 78.5 | 63.3 |
| Multi-Image VLPart | 13.4 | 42.8 | 34.6 | 59.1 | 46.5 |
| Multi-Image GLaMM (微调) | 42.9 | 59.9 | 54.9 | 76.8 | 71.2 |
| Multi-Image LISA (微调) | 41.4 | 59.7 | 55.5 | 78.7 | 72.5 |
| **Calico** | **45.9** | **63.7** | **59.7** | **82.7** | **77.1** |

### 消融实验

| 配置 | AP50 | mIoU | Recall | SS | S-IoU |
|------|------|------|--------|-----|-------|
| w/o Q-Former | 38.5 | 59.2 | 44.8 | 64.5 | 55.6 |
| w/o DINO | 43.9 | 61.7 | 57.1 | 80.2 | 74.5 |
| w/o CEM | 43.6 | 61.6 | 57.5 | 80.8 | 75.2 |
| w/o CAM | 45.9 | 63.3 | 59.7 | 82.0 | 76.5 |
| w/o CEM w/o CAM | 44.1 | 62.7 | 58.1 | 81.6 | 76.3 |
| **Calico (Full)** | **45.9** | **63.7** | **59.7** | **82.7** | **77.1** |

### 关键发现

- **CEM 是核心贡献**：去掉 CEM 后分割指标全面下降（mIoU 从 63.7 降至 61.6），证明 DINOv2 语义对应信息对跨图像部件理解至关重要
- **CAM 需要 CEM 引导才有效**：单独保留 CAM 而去掉 CEM 时（w/o CEM），性能反而低于两者都去掉（w/o CEM w/o CAM），说明 CAM 在无外部语义信号时会注入冗余/噪声信息
- **Q-Former 不可或缺**：去掉 Q-Former 时，由于 CEM 和 CAM 都依赖 Q-Former 架构，性能大幅下降（AP50 从 45.9 降至 38.5）
- **CAM 层数选择**：2 层均匀分布（第 11、22 层）最优，1 层和 3 层配置下性能均有所下降
- **效率优势显著**：使用 32 token/图像（vs LISA 的 256、GLaMM 的 576），TFLOPS 降低 32-35%，推理速度提升 30-51%

## 亮点与洞察

- **任务定义价值高**："部件聚焦语义共分割"是一个定义良好且具有广泛应用前景的新任务，涵盖定位+比较+命名三个子目标。这种多图像多粒度的视觉推理任务设计思路值得借鉴
- **DINOv2+CLIP 融合范式巧妙**：CEM 利用交叉注意力将 DINOv2 的部件级语义对应能力"嫁接"到 CLIP 特征上，避免了重新训练大模型，且冻结两个编码器保证了效率。这种"冻结+融合"的范式可迁移到其他需要多源视觉特征的任务
- **CAM 的指令引导设计**：用 LLM 层的最后一个 token（承载指令语义）来引导视觉特征提取，实现了"根据用户提问动态选择关注的视觉内容"，这种设计思路可以推广到其他需要条件视觉理解的场景

## 局限与展望

- 仅在 MixedParts 这一个自建数据集上评估，缺少在其他部件分割基准上的泛化性验证
- 假设 DINOv2 特征能充分捕捉部件级语义对应，但对于功能相似而外观差异大的部件（如不同材质的"把手"），这一假设可能失效
- 多图像输入目前限于 2 张图像，扩展到更多图像的可扩展性未验证
- MixedParts 数据集虽然规模大（240 万样本），但部件标注来源于已有数据集，可能继承了原始标注的偏差
- 未来可探索将 CEM/CAM 模块迁移到 3D 医学影像的器官/亚结构对比分析中

## 相关工作与启发

- **vs LISA**: LISA 是基于 LLaVA 的单图像分割 LVLM，Calico 在 LISA 基础上增加了多图像能力和部件级对应模块。微调后的 Multi-Image LISA 在 MixedParts 上表现不错但落后于 Calico，说明专用的对应模块是必要的
- **vs GLaMM**: GLaMM 是 Calico 的直接初始化来源，两者共享相同的基础权重。Calico 的 CEM+CAM 模块带来了显著的性能提升（mIoU: 59.9→63.7），证明了对应信息注入的价值
- **vs SCOPS/DFF 等传统部件共分割**: 这些方法无法生成语义标签且需要事先指定部件类别数，Calico 通过 LLM 的文本生成能力自动命名发现的部件，实现了真正的开放式部件共分割

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义部件聚焦语义共分割任务，CEM/CAM 设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面，效率分析详细，但仅在单一数据集上评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法描述系统，实验分析深入
- 价值: ⭐⭐⭐⭐ 任务定义和数据集贡献价值高，对多图像 LVLM 领域有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)

</div>

<!-- RELATED:END -->
