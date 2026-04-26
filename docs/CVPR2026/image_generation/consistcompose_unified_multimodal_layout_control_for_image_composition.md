---
title: >-
  [论文解读] ConsistCompose: Unified Multimodal Layout Control for Image Composition
description: >-
  [CVPR2026][图像生成][布局控制生成] 提出 LELG（语言嵌入式布局引导生成）范式，将 bounding box 坐标直接编码为文本 token 嵌入语言流，在统一多模态 Transformer 中实现布局可控的多实例图像生成，无需任何布局专用编码器或分支。
tags:
  - CVPR2026
  - 图像生成
  - 布局控制生成
  - 多实例组合
  - 统一多模态模型
  - 语言嵌入式布局
  - Classifier-Free Guidance
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# ConsistCompose: Unified Multimodal Layout Control for Image Composition

**会议**: CVPR2026  
**arXiv**: [2511.18333](https://arxiv.org/abs/2511.18333)  
**代码**: 待确认  
**领域**: 图像生成  
**关键词**: 布局控制生成, 多实例组合, 统一多模态模型, 语言嵌入式布局, Classifier-Free Guidance

## 一句话总结

提出 LELG（语言嵌入式布局引导生成）范式，将 bounding box 坐标直接编码为文本 token 嵌入语言流，在统一多模态 Transformer 中实现布局可控的多实例图像生成，无需任何布局专用编码器或分支。

## 背景与动机

1. **现有统一多模态模型偏重理解**：DALL-E、Stable Diffusion 等生成质量优异，但统一多模态模型主要聚焦视觉理解（grounding），对布局可控生成支持不足，限制了组合式场景合成。
2. **扩散方法架构耦合严重**：GLIGEN、InstanceDiffusion 等依赖布局-图像融合模块或 region-aware U-Net 改造，与 Transformer 生成框架难以兼容。
3. **自回归方法仍局限于布局任务**：LayoutSAM、HiCo、PlanGen 等仅处理布局相关任务，未展示与视觉推理、图像编辑等通用多模态能力的兼容性。
4. **多参考身份保持生成被忽视**：大多数工作仅关注文本条件布局控制，很少考虑多参考、身份保持的多实例组合这一更具挑战性的设定。
5. **大规模数据集缺失**：缺乏同时提供实例级布局、主体对应关系和多样多实例配置的大规模数据集，制约了统一布局感知系统的发展。
6. **核心洞察——将布局编码为语言**：当布局语义表示为文本的一部分时，多模态 Transformer 可通过与感知和推理相同的交错建模学习空间对齐，无需任何架构修改。

## 方法详解

### 整体框架

ConsistCompose 基于 Bagel 的 MoT（Mixture-of-Transformers）架构构建，包含两个 Transformer 专家：一个用于多模态理解，一个用于多模态生成。使用两个视觉编码器：SigLIP 初始化的 ViT 用于语义感知，FLUX 初始化的 VAE 用于图像生成。所有模态（文本、ViT 特征、VAE 潜变量）投影到共享嵌入空间。

### LELG 范式与 ICBP 机制

核心创新是 **语言嵌入式布局引导生成（LELG）**：将归一化 bounding box 直接作为文本 token 插入。每个实例 $i$ 的坐标 $b_i=(x_1^i, y_1^i, x_2^i, y_2^i) \in [0,1]^4$ 以三位小数精度紧跟在对应主语短语之后：

```
"a brown sofa <bbox>[0.123, 0.456, 0.789, 0.987]</bbox>"
```

这一 **Instance Coordinate Binding Prompt（ICBP）** 机制将对象的语言引用与空间坐标绑定在统一生成序列中，无需额外几何损失，模型通过交错文本和视觉 token 上的自注意力隐式学习空间对齐。

### Coordinate-CFG 增强空间控制

引入坐标感知 Classifier-Free Guidance：条件分支使用含坐标 token 的 prompt，无条件分支省略坐标。公式为：

$$\mathbf{v}_t^{\text{coord-cfg}} = \mathbf{v}_t^{\text{coord-uncond}} + s_{\text{coord}} (\mathbf{v}_t^{\text{coord}} - \mathbf{v}_t^{\text{coord-uncond}})$$

额外引入速度归一化保证引导幅度稳定。$s_{\text{coord}}$ 越大布局越严格，过大则质量下降——COCO-Position 上最优 1.6，MS-Bench 上最优 0.4–0.8。

### 训练目标与策略

- **Flow Matching 损失**：$\mathcal{L}_{\text{FM}}$ 用于图像生成，预测潜变量间的速度场
- **自回归 LM 损失**：$\mathcal{L}_{\text{LM}}$ 用于多模态理解的 next-token prediction
- **总损失**：$\mathcal{L} = \lambda_{\text{FM}} \mathcal{L}_{\text{FM}} + \lambda_{\text{LM}} \mathcal{L}_{\text{LM}}$，不引入额外坐标回归损失
- **两阶段训练**：① 对齐阶段混合通用多模态理解数据 + ConsistCompose3M；② 混合 SFT 阶段联合训练理解、生成、编辑和多主体参考生成数据

### ConsistCompose3M 数据集

构建 340 万样本大规模数据集，包含两部分：

- **T2I 组件**（260 万）：重处理 LayoutSAM，通过 ICBP 机制将实例级布局注释注入 prompt
- **参考条件组件**（80 万）：复用 Subjects200K 和 UNO 的主体资产，在多样布局下重组为多主体场景，经 CLIP/DINO 相似度过滤确保身份一致性

## 实验关键数据

### COCO-Position 布局控制

| 方法 | Avg Instance SR(%) | Avg Image SR(%) | mIoU | AP | AP50 | AP75 |
|------|-------------------|-----------------|------|------|------|------|
| GLIGEN | 82.6 | 52.1 | 69.0 | 40.5 | 75.9 | 39.1 |
| InstanceDiffusion | 87.8 | 65.5 | 78.1 | 57.2 | 83.6 | 65.5 |
| MIGC++ | 86.8 | 63.4 | 74.9 | 48.3 | 79.2 | 52.6 |
| PlanGen | 82.5 | 50.3 | 66.2 | 31.9 | 74.0 | 21.5 |
| **ConsistCompose** | **92.6** | **76.1** | **85.3** | **70.9** | **89.1** | **76.9** |

ConsistCompose 在所有指标上均取得最优。mIoU 提升 7.2%（78.1→85.3），AP 提升 13.7%（57.2→70.9），高实例数场景（L4-L6）优势尤为显著。

### MS-Bench 多参考身份保持

| 方法 | CLIP-T | DINO | mIoU | AP |
|------|--------|------|------|-----|
| GLIGEN | 0.309 | 0.454 | 0.868 | 0.751 |
| MS-Diffusion | 0.336 | 0.555 | 0.466 | 0.108 |
| MUSE | 0.320 | 0.619 | 0.698 | 0.352 |
| **ConsistCompose** | 0.333 | **0.660** | **0.889** | **0.789** |

在身份保持（DINO）和空间精度（mIoU、AP）上同时取得最优，打破了先前方法在这两个维度上的此消彼长。

### 消融：通用能力保持

布局训练对通用能力无明显损害：MMBench 81.4（与 Bagel Base 持平），GenEval 0.88（略优于 Base 的 0.86），DreamBench 单/多目标 DINO 分别达 0.677/0.506，均为最优。

## 亮点

- **极简统一设计**：布局信息完全编码为文本 token，无需布局编码器、区域注意力或任务分支，优雅地统一了空间控制与多模态理解
- **Coordinate-CFG 可调控**：通过推理时 CFG 尺度灵活控制布局精度与视觉质量的平衡，不同任务可独立设置最优值
- **大规模数据集贡献**：ConsistCompose3M 填补了布局+身份联合标注数据集的空白
- **通用能力无损**：布局训练后 MMBench/GenEval 等通用指标持平甚至略优，证明统一训练不牺牲已有能力

## 局限与展望

- 当前仅支持 bounding box 级空间控制，不支持更精细的语义分割、关键点等空间描述
- Coordinate-CFG 需要额外推理步骤（含/不含坐标的双路推理），增加了采样开销
- MMMU 分数（42.3）较 Bagel Base（46.4）有所下降，暗示强布局训练可能轻微影响跨域推理
- 三位小数精度的坐标离散化是否为最优粒度仍不清楚
- 基于 Bagel 的 MoT 架构规模较大，部署成本和训练资源需求高
- 未探索交互式布局编辑或渐进式场景构建等更丰富的应用场景

## 与相关工作的对比

- **vs GLIGEN/InstanceDiffusion**：基于 U-Net 的布局融合模块，架构耦合度高；ConsistCompose 以纯文本接口实现空间控制，更轻量且通用
- **vs PlanGen/LayoutSAM**：使用结构化空间 token 或独立模态处理布局，仅限布局任务；ConsistCompose 统一布局、理解、生成
- **vs MS-Diffusion/MUSE**：多参考生成方法在空间精度或身份保持上有偏科；ConsistCompose 二者兼顾
- **vs OmniGen2/Bagel**：统一多模态模型但缺乏显式布局控制；ConsistCompose 在 Bagel 基础上通过 LELG 补全了空间控制能力

## 评分

- 新颖性: ⭐⭐⭐⭐ — LELG 范式将布局编码为语言 token 的思路简洁而有效，Coordinate-CFG 机制设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖 COCO-Position、MS-Bench、GenEval、DreamBench 等多个 benchmark，消融充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，符号一致，图表质量高
- 价值: ⭐⭐⭐⭐ — 为统一多模态模型增加布局控制能力提供了可行路径，数据集贡献有持续价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)

<!-- RELATED:END -->
