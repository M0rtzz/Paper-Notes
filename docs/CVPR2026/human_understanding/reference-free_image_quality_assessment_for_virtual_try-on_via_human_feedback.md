---
title: >-
  [论文解读] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback
description: >-
  [CVPR2026][人体理解][虚拟试穿] 提出 VTON-IQA，一个无需参考图的虚拟试穿图像质量评估框架，通过构建 62,688 张试穿图像 × 431,800 条人工标注的大规模基准 VTON-QBench，以及交错式交叉注意力（ICA）模块建模服装-人物-试穿图之间的交互关系，实现与人类感知高度对齐的图像级质量预测。
tags:
  - "CVPR2026"
  - "人体理解"
  - "虚拟试穿"
  - "图像质量评估"
  - "无参考评估"
  - "人类反馈对齐"
  - "交叉注意力"
  - "大规模标注基准"
---

# Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback

**会议**: CVPR2026  
**arXiv**: [2603.13057](https://arxiv.org/abs/2603.13057)  
**代码**: [GitHub](https://github.com/litelightlite/VTON-IQA)  
**领域**: 人体理解 / 虚拟试穿质量评估  
**关键词**: 虚拟试穿, 图像质量评估, 无参考评估, 人类反馈对齐, 交叉注意力, 大规模标注基准

## 一句话总结

提出 VTON-IQA，一个无需参考图的虚拟试穿图像质量评估框架，通过构建 62,688 张试穿图像 × 431,800 条人工标注的大规模基准 VTON-QBench，以及交错式交叉注意力（ICA）模块建模服装-人物-试穿图之间的交互关系，实现与人类感知高度对齐的图像级质量预测。

## 背景与动机

1. **实际场景缺乏真实参考图**：在真实电商部署中，同一人穿着目标服装的 ground-truth 图像通常不可得，使得 SSIM/LPIPS 等全参考指标无法使用
2. **分布级指标无法反映单图质量**：FID/KID 仅衡量数据集级别统计相似度，无法评估单张生成图像的感知质量
3. **现有 VTON 评估方法缺乏大规模人类验证**：VTON-VLLM 侧重文本批评而非定量打分，VTBench 使用 LLM 判断但未从大规模人类标注中学习，VTONQA 数据集规模有限（仅 748 对、40 标注者）
4. **缺乏公开可复现的评估基准**：现有方法未开源实现和标准化基准，阻碍可复现评估
5. **试穿质量评估不同于单图 IQA**：需要同时验证服装保真度和人物特征保持，本质上需要跨图交互建模
6. **传统指标对全局变换过度惩罚**：SSIM/LPIPS 对姿态变化和缩放变化敏感，与人类感知不一致

## 方法详解

### 整体框架

VTON-IQA 要解决的是一个很实际的评估难题：电商场景里根本拿不到「同一个人穿着目标服装」的 ground-truth，SSIM/LPIPS 这类全参考指标全部失效，FID/KID 又只能看数据集级别的统计、说不清单张图好不好。它的方案是一个三分支 Transformer：把服装图 $I_G$、人物图 $I_P$、生成的试穿图 $I_V$ 各自做 patch embedding 加 [CLS] token，前 L/2 层各自独立自注意力提特征，后 L/2 层引入 ICA 模块让三路互相交互，最后取 [CLS] 表征做加权余弦打分，输出一个连续质量分 $\hat{s} \in [-1, 1]$。backbone 用 DINOv3 ViT-L/16。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    G["服装图 I_G"] --> S
    P["人物图 I_P"] --> S
    V["试穿图 I_V"] --> S
    S["patch embedding + [CLS]<br/>前 L/2 层各分支独立自注意力"] --> ICA["ICA 交错式交叉注意力<br/>以试穿图为中心的非对称交互（V↔G、V↔P）"]
    ICA --> SC["加权余弦打分<br/>α·cos(c_G,c_V) + (1−α)·cos(c_P,c_V)"]
    SC --> O["质量分 ŝ ∈ [−1,1]"]
```

### 关键设计

**1. 交错式交叉注意力 ICA：以试穿图为中心的非对称交互**

试穿质量评估的本质是跨图判断——既要看服装保真度，又要看人物特征保持，单看一张图判断不了。ICA 在标准 Transformer 块的自注意力和 MLP 之间插入交叉注意力层，并刻意做成非对称：试穿分支同时聚合服装和人物两路信息 $\hat{X}_V = \tilde{X}_V + C_{V \leftarrow G} + C_{V \leftarrow P}$，而服装、人物分支只从试穿分支取信息 $\hat{X}_G = \tilde{X}_G + C_{G \leftarrow V}$、$\hat{X}_P = \tilde{X}_P + C_{P \leftarrow V}$。它显式建模 (V,G) 和 (V,P) 两组双向交互，却刻意避免 G↔P 直接耦合——因为质量判断本就以试穿图为中心，让服装和人物直接对话反而引入无关信息。消融里 ICA 带来 SRCC +0.133、PLCC +0.136 的提升。

**2. 加权余弦打分：分别度量服装与人物两路保真**

提取三路 [CLS] token $c_G, c_P, c_V$ 后，用一个可学习权重 $\alpha$ 把服装保真和人物保真两路余弦相似度加权起来：

$$\tilde{s} = \alpha \cdot \cos(c_G, c_V) + (1-\alpha) \cdot \cos(c_P, c_V)$$

再经可学习仿射变换加 tanh 映射到 $[-1,1]$。$\alpha$ 让模型自己学到这两路在最终质量里各占多少分量，而不是写死某个比例。

**3. VTON-QBench：用大规模人类标注把模型对齐到人的感知**

模型要对齐人的判断，前提是有足够大的人类标注。VTON-QBench 在这一点上把规模拉满：

| 维度 | 规模 |
|------|------|
| 服装-人物对 | 13,153（含合成增强 1.9×） |
| 试穿图像 | 62,688 |
| VTON 模型 | 14（涵盖 GAN/U-Net Diffusion/DiT/商业模型） |
| 合格标注者 | 13,838 |
| 质量标注 | 431,800 |

标注用三级——不自然(1)/轻微不自然但不明显(2)/完全自然(3)，最终分取多标注者均值；数据再经两阶段清洗（dummy 题筛查+异常行为检测，以及剔除 Krippendorff's α ≤ 0.4 的问卷），把 α 从 0.286 提到 0.550。正是这套数据让微调后的模型成对准确率逼近人类水平（A_macro 0.771 vs 人类 0.782）。

### 损失函数 / 训练策略

联合优化两个目标：(1) Bradley-Terry 偏好学习——对同一人-服装对的两张试穿结果建模成对偏好，用软标签交叉熵对齐预测偏好与人类偏好；(2) 分数回归——用 L2 损失约束预测分数与人类评分一致。合起来即

$$\mathcal{L} = -q_{ij}\log p_\theta - (1-q_{ij})\log(1-p_\theta) + \sum_{k}\|\Psi_\theta - S_k\|_2^2$$

## 实验关键数据

### 主实验：与 baseline 对比（VTON-QBench 测试集）

| 方法 | SRCC↑ | PLCC↑ | R²↑ | A_macro↑ | A_micro↑ |
|------|-------|-------|-----|----------|----------|
| SSIM | – | 0.135 | – | 0.596 | 0.593 |
| LPIPS | – | 0.387 | – | 0.701 | 0.695 |
| DINOv3 (zero-shot) | – | 0.261 | – | 0.637 | 0.641 |
| VTON-IQA w/o ICA | 0.617 | 0.615 | 0.372 | 0.722 | 0.747 |
| **VTON-IQA (full)** | **0.750** | **0.751** | **0.553** | **0.781** | **0.790** |

- ICA 模块带来 SRCC +0.133、PLCC +0.136 的显著提升
- 成对准确率接近人类水平（人类 A_macro=0.782, 模型 0.771）

### 14 个 VTON 模型 Benchmark（VITON-HD，unpaired）

| 模型 | VTON-IQA↑ | FID↓ |
|------|-----------|------|
| Nano Banana Pro | **0.315** | 10.309 |
| GPT-Image-1.5 | 0.234 | 12.801 |
| FitDit | 0.189 | 9.893 |
| Qwen-Image-Edit | 0.087 | 10.706 |
| IDM-VTON | 0.039 | 9.093 |
| OOTDiffusion | -0.142 | 9.064 |
| LADI-VTON | -0.864 | 21.515 |

商业模型在人类对齐评分上显著领先；FID/KID 与人类感知不一定一致。

### 消融分析

- **ICA vs 无 ICA**：ICA 对所有指标均有显著提升，验证了跨图交互建模的必要性
- **非对称 vs 对称交互**：非对称设计（避免 G↔P 耦合）更符合试穿质量评估的语义结构
- **任务特定训练 vs zero-shot**：DINOv3 zero-shot → 微调带来 PLCC +0.354，说明 VTON-QBench 上训练至关重要

## 亮点

1. **数据集规模空前**：VTON-QBench 是目前已知最大的虚拟试穿人类主观评估数据集，且计划开源
2. **非对称 ICA 设计精巧**：以试穿图为中心的交互结构符合评估语义，避免无关耦合
3. **成对准确率达人类水平**：A_macro 仅差 0.011，说明模型在偏好排序上已具有实用价值
4. **首次对 14 个 VTON 模型做统一 benchmark**：涵盖 GAN/UNet-Diffusion/DiT/商业模型，结果显示传统指标与人类感知存在系统性偏差
5. **合成数据增强管道完整**：LoRA+FLUX.1-dev 生成 + GPT 过滤 + 人工审核，服装-人物对扩展 1.9 倍

## 局限与展望

1. **三级标注粒度较粗**：仅 3 个等级可能无法捕捉细粒度质量差异，更细的连续评分或多维打分可能更优
2. **相关性指标与人类仍有差距**：SRCC=0.750 vs 人类 0.760，R²=0.489 vs 0.536，绝对值预测仍有提升空间
3. **未涵盖多维质量评估**：仅输出单一整体分数，缺乏对服装纹理/颜色/形状/长度等子维度的诊断能力
4. **backbone 较重**：DINOv3 ViT-L/16 三分支推理开销大，部署时需考虑效率
5. **合成增强依赖商业模型**：伪三元组构建使用 Nano Banana Pro，数据集构建成本和可复现性受限
6. **未探索视频试穿等动态场景**的质量评估

## 与相关工作的对比

| 方法 | 数据规模 | 无参考 | 图像级打分 | 人类标注 | 开源 |
|------|----------|--------|------------|----------|------|
| SSIM/LPIPS | N/A | ✗ | ✓ | ✗ | ✓ |
| FID/KID | N/A | ✓ | ✗（分布级） | ✗ | ✓ |
| VTONQA | 748对/8132图/40人 | ✓ | ✓ | ✓ | ✗ |
| VTON-VLLM | – | ✓ | ✗（文本） | ✓ | ✗ |
| VTBench | – | ✓ | ✓ | 间接 | – |
| **VTON-IQA** | **13K对/63K图/14K人** | **✓** | **✓** | **✓** | **✓** |

VTON-IQA 在数据规模、评估方式完备性和开源承诺上全面超越先前工作。

## 评分

- 新颖性: ⭐⭐⭐⭐ — ICA 的非对称跨图交互设计有新意，大规模 VTON 人类标注基准填补空白
- 实验充分度: ⭐⭐⭐⭐⭐ — 14 个模型 benchmark + 与人类对比 + 消融 + 定性分析，极为完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据集构建描述详尽，公式表达规范
- 价值: ⭐⭐⭐⭐ — 为虚拟试穿社区提供标准化评估基准和工具，工程和学术价值兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)

</div>

<!-- RELATED:END -->
