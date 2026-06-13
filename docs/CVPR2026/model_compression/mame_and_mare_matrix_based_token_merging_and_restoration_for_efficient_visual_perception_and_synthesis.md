---
title: >-
  [论文解读] MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis
description: >-
  [CVPR 2026 Findings][模型压缩][token merging] 提出 MaMe，一种基于全矩阵运算的免训练可微分 token 合并方法，以及其逆操作 MaRe 用于 token 恢复，在图像分类、视频识别和图像生成等任务中实现高效加速且性能损失极小。 Vision Transformer (ViT) 的自…
tags:
  - "CVPR 2026 Findings"
  - "模型压缩"
  - "token merging"
  - "token restoration"
  - "Transformer"
  - "matrix operation"
  - "图像生成"
---

# MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis

**会议**: CVPR 2026 Findings  
**arXiv**: [2604.13432](https://arxiv.org/abs/2604.13432)  
**代码**: [github.com/cominder/mame](https://github.com/cominder/mame)  
**领域**: 模型压缩/高效推理  
**关键词**: token merging, token restoration, efficient transformer, matrix operation, image generation

## 一句话总结

提出 MaMe，一种基于全矩阵运算的免训练可微分 token 合并方法，以及其逆操作 MaRe 用于 token 恢复，在图像分类、视频识别和图像生成等任务中实现高效加速且性能损失极小。

## 研究背景与动机

Vision Transformer (ViT) 的自注意力机制复杂度为 $\mathcal{O}(N^2)$，限制了大规模 ViT 在资源受限设备上的部署。现有 token 压缩方法存在多个局限：Top-K 操作不可微分导致端到端训练困难；k-means 等聚类方法计算密集实际较慢；许多方法引入额外可学习参数增加模型复杂度；部分方法依赖特定架构（如依赖 class token）。ToMe 虽为免训练方法但依赖 GPU 不友好的排序和散射写入操作。

## 方法详解

### 整体框架

MaMe 要解决的是 ViT 里 token 太多、注意力 $\mathcal{O}(N^2)$ 太贵，而已有压缩方法要么不可微（Top-K）、要么慢（k-means）、要么得加参数或排序散射（ToMe）不适合 GPU。它的思路是把整套 token 合并写成纯矩阵运算：先把 token 序列切成不相交的目标集（dst）和源集（src），用归一化余弦相似度构一张融合矩阵 $W^F$，把源 token 按权重聚到目标 token 上，同时把跟谁都不像的"独特"源 token 原样留下。逆操作 MaRe 再按存下的同一张矩阵把合并掉的 token 还原回原长度。两者搭配的用法因任务而异：感知任务（分类/识别）只用 MaMe，注意力算完后合并、序列越往后越短；生成任务则在每个 Transformer block 内"注意力前用 MaMe 合并、注意力后用 MaRe 还原"，让昂贵的自注意力只作用在短序列上，而 block 间的 token 数和空间分辨率保持不变。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入 token 序列 X<br/>（排除 class 等特殊 token）"] --> B["切分目标集 dst / 源集 src<br/>（交替或随机，互不相交）"]
    B --> C["自适应权重精修<br/>余弦相似度 → ReLU(τ) → 列归一化<br/>→ 动态列阈值 ζ → 重归一化得 W^F"]
    C --> D["token 聚合<br/>源 token 按 W^F 加权并入目标 token"]
    C --> E["独特 token 保留<br/>与所有 dst 相似度为 0 的源 token 原样留下"]
    D --> F["压缩后序列 X′"]
    E --> F
    F -->|感知任务| G["直接分类 / 识别"]
    F -->|生成任务| H["MaRe token 恢复<br/>用存下的 W^F 逆变换还原回原长度"]
    H --> I["block 内还原序列 → 继续生成"]
```

### 关键设计

**1. 自适应权重精修：把合并写成 GPU 友好的矩阵运算**

ToMe 依赖排序和散射写入，在 GPU 上并不友好。MaMe 全程只用矩阵算子：先算源-目标余弦相似度矩阵，用 ReLU 加位移阈值 $\tau$ 滤掉弱连接，列归一化后再引入动态列级阈值 $\zeta_j$（取该列非零权重的均值）做第二次 ReLU 剪枝并重归一化。两道阈值让融合矩阵既稀疏又自适应——$\tau$ 太小会过度合并、太大则留太多 token，由它控制融合稀疏度，而整个过程没有任何排序或散射，天然并行。

**2. 独特 token 保留：不把"谁都不像"的 token 硬塞进别人**

如果某个源 token 与所有目标 token 的融合权重之和为零，说明它和谁都不相似，强行合并就会污染目标 token。MaMe 直接把这类 token 原样保留不合并；批处理时用 union 策略统一各样本要保留的位置、把对应列置零防止融合，保证一个 batch 内行为一致。消融显示移除这一机制后性能明显下降，说明这些独特 token 往往携带了不可替代的信息。

**3. MaRe token 恢复：让压缩在生成流水线里可逆**

分类任务合并完直接分类即可，但图像生成需要保持 token 数和空间分辨率不变。MaRe 作为 MaMe 的逆操作，同样基于全矩阵运算：它复用合并时存下的融合矩阵 $W^F$，按每个源 token 当初贡献给目标 token 的比例，把信息从合并后的目标 token 反向分配回原来的位置（被保留的独特 token 直接放回）。集成到生成模型时是在**每个 Transformer block 内部**完成"注意力前 MaMe 合并、注意力后 MaRe 还原"（$x'_l = \text{MaRe}(\text{MSA}(\text{MaMe}(\text{LN}(x_{l-1})))) + x_{l-1}$），昂贵的自注意力只算短序列、block 间序列长度保持不变。在 Stable Diffusion v2.1 上这一流水线把延迟降了 31%，又因为信息被结构化地压缩再恢复，实测反而提升了生成质量。

### 损失函数 / 训练策略

MaMe 是免训练方法，可直接套到预训练模型上，也能作为即插即用模块集成进从头训练；整个过程全可微、保持梯度流，所以无需任何额外可学习参数。

## 实验关键数据

### 主实验

| 任务 | 模型 | 指标 | MaMe 结果 | 加速比 |
|------|------|------|-----------|--------|
| 图像分类 | ViT-B | 准确率 | -2% | 2× 吞吐量 |
| 图像分类 | ViT-B (微调最后层) | 准确率 | +1.0% | 1.1× |
| 零样本分类 | SigLIP2-B@512 | 准确率 | 几乎无损 | 1.3× |
| 视频识别 | VideoMAE-L (K400) | 准确率 | -0.84% | +48.5% 速度 |
| 图像生成 | SD v2.1 (MaMe+MaRe) | 质量 | 提升 | -31% 延迟 |

### 消融实验

- 阈值 $\tau$ 控制融合稀疏度：过小导致过度合并，过大则保留过多 token
- 独特 token 保留是关键：移除后性能明显下降
- 自适应权重精修相比简单归一化提升显著

### 关键发现

- MaMe 在某些任务上实现了性能和速度的同时提升
- 在图像生成中 MaMe+MaRe 不仅加速还能提升生成质量
- 方法天然保持因果性，可作为 LLM 中减少 KV cache 大小的候选方案

## 亮点与洞察

- 全矩阵运算设计理念优雅，兼顾理论效率和实际加速
- 可微分 + 无参数 + 即插即用的三重优势使其实用性极强
- 微调最后一层即可反超原模型精度是很有说服力的结果

## 局限与展望

- token 分区策略（交替/随机）的选择对性能影响有待更深入分析
- 在密集预测任务（如目标检测、分割）上的适用性未充分验证
- LLM KV cache 压缩的讨论停留在概念层面

## 相关工作与启发

- 相比 ToMe 的二分图匹配，矩阵运算方案更 GPU 友好
- 可微分设计使得与端到端训练的结合更自然
- MaRe 恢复操作为生成模型的 token 压缩提供了新思路

## 评分

7/10 — 设计优雅、实验全面、实用性强，但核心创新集中在工程实现层面。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[ICML 2025\] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](../../ICML2025/model_compression/bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)

</div>

<!-- RELATED:END -->
