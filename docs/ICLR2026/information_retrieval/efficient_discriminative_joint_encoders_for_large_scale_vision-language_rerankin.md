---
title: >-
  [论文解读] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking
description: >-
  [ICLR 2026][视觉语言检索] 提出EDJE（高效判别式联合编码器），通过将视觉特征提取离线化并用轻量级注意力适配器压缩视觉Token，实现50k图文对/秒的高吞吐推理，同时在Flickr（零样本）和COCO（微调）检索上匹配现有联合编码器的性能，每张图仅需49kB存储。
tags:
  - ICLR 2026
  - 视觉语言检索
  - 联合编码器
  - 重排序
  - Token压缩
  - 高效推理
---

# Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking

**会议**: ICLR 2026  
**arXiv**: [2510.06820](https://arxiv.org/abs/2510.06820)  
**代码**: [GitHub](https://github.com/gitanony04-lab/Simple-Efficient-Fusion)  
**领域**: 多模态VLM  
**关键词**: 视觉语言检索, 联合编码器, 重排序, Token压缩, 高效推理

## 一句话总结
提出EDJE（高效判别式联合编码器），通过将视觉特征提取离线化并用轻量级注意力适配器压缩视觉Token，实现50k图文对/秒的高吞吐推理，同时在Flickr（零样本）和COCO（微调）检索上匹配现有联合编码器的性能，每张图仅需49kB存储。

## 研究背景与动机
大规模多模态检索中，基于嵌入的模型（如CLIP）通过向量相似度实现高效搜索，但独立编码两个模态限制了细粒度跨模态交互。联合编码器（如BLIP、BLIP-2）将两个模态联合处理，检索性能更强，在文本检索领域cross-encoder重排序已是标准范式。

**核心矛盾**：现有联合编码器的视觉特征提取是严重瓶颈——BLIP用ViT-B处理一个batch的64张图需~400ms，用ViT-L需~1400ms，视觉特征提取占总推理时间的83%-93%。相比之下，文本检索中最流行的MiniLM重排器仅22M参数、60ms处理同等batch。这解释了为什么多模态重排器在实际系统中几乎缺席。

**核心idea**: 将视觉特征提取离线化——图像编码一次后缓存到磁盘，推理时仅运行紧凑的联合编码器处理少量视觉Token和文本，同时通过Token压缩适配器大幅降低存储需求。

## 方法详解

### 整体框架
EDJE分为离线和在线两个阶段：
- **离线阶段**: 用ViT编码图像，通过适配器压缩为紧凑Token集并存储到磁盘
- **在线阶段**: 紧凑语言模型（MiniLM）联合处理压缩的视觉Token和文本Token，输出重排序分数

### 关键设计
1. **视觉预计算范式转换**: 由于视觉编码器仅对图像操作，其输出可以缓存复用。一个ViT-B将每个16×16 patch投影为d=384的嵌入，以FP16存储时占用空间与原始8位RGB图像相当。扩大视觉编码器可提升表示质量而不增加在线成本。关键挑战是原始存储量在web-scale数据库中不可行（可达TB级），需要压缩策略。

2. **Token压缩适配器**: 引入 $m$ 个可学习的通用查询Token $\mathbf{Q} = [\mathbf{q}_1, ..., \mathbf{q}_m]$，通过交叉注意力机制聚合 $n$ 个视觉Token $\mathbf{X}$ 的信息：$\mathbf{H} = \text{MultiHeadAttention}(\mathbf{Q}, \mathbf{X}\mathbf{W}_K, \mathbf{X}\mathbf{W}_V)$。输出经残差块和线性投影到语言模型嵌入空间：$\mathbf{Y} = (\mathbf{H} + \text{MLP}(\text{LayerNorm}(\mathbf{H})))\mathbf{W}_{proj}$。这将576个ViT Token压缩到64个Token，存储从442kB降至49kB/图。

3. **紧凑联合编码器**: 采用VLM的范式——将视觉Token投影到语言嵌入空间后与文本Token拼接，用自注意力处理跨模态交互。语言模型使用MiniLM（33M参数），远小于BLIP的139-167M参数。这种设计具有模块化优势：任意ViT视觉编码器可与任意预训练语言模型配对。

### 损失函数 / 训练策略
联合优化三个目标：
- **图文匹配（ITM）**: 正对 vs. batch内硬负样本的二分类
- **掩码语言建模（MLM）**: 50%文本Token掩码预测，增强跨模态依赖
- **文本嵌入恢复**: CLS Token投影后与文本编码器嵌入的余弦距离最小化
- **Local-to-Compressed蒸馏**: 用未压缩的Local模型作为教师，对压缩模型进行logit级蒸馏

使用CC12M/CC3M/SBU/VG/COCO共14M图文对预训练，仅用COCO微调。

## 实验关键数据

### 主实验

**Flickr30k零样本检索（SigLIP2 ViT-L/16, 384²）:**

| 方法 | T2I R@1 | I2T R@1 | 存储/图 | 参数量 | 推理时间 |
|------|---------|---------|---------|--------|----------|
| BLIP ViT-L/16 | 86.7 | 96.7 | 2,359kB | 139M | 101.61ms |
| BLIP-2 ViT-L/16 | 88.6 | 96.9 | 2,359kB | 167M | 98.64ms |
| EDJE Local | 87.8 | 96.5 | 442kB | 33M | 4.14ms |
| EDJE Compressed-64 | 86.9 | 96.4 | 49kB | 33M | 1.91ms |

**EDJE对多种嵌入模型的提升（Flickr30k零样本T2I R@1）:**

| 骨干网络 | 原始 | +EDJE | 提升 |
|----------|------|-------|------|
| CLIP ViT-B/16 | 62.1 | 76.8 | +14.7 |
| CLIP ViT-L/14 | 65.2 | 80.6 | +15.4 |
| SigLIP2 ViT-B/16 | 82.1 | 84.3 | +2.2 |
| SigLIP2 ViT-L/16 | 82.3 | 87.8 | +5.5 |

### 消融实验
- **Token数量**: 测试了{32, 64, 128, 256}个目标Token，64个Token在效率和性能间取得最佳平衡，32 Token出现明显性能下降，256 Token接近Local变体的576 Token性能
- **重排池大小**: 从k=5到k=50，检索性能在R@1/5/10上保持稳定，证明EDJE对噪声候选的鲁棒性
- **训练目标**: ITM单独→+MLM→+文本嵌入恢复，三者逐步叠加均有正向贡献，完整目标最强
- **Local-to-Compressed蒸馏**: 为压缩变体提供进一步的判别能力增益
- **语义分析**: 压缩后的64个Token映射到有意义的物体/场景描述词（如boulders, caves），而未压缩的576 Token中大量映射到无意义的特殊Token（unused80），说明原始ViT Token存在大量冗余
- **量化实验**: 压缩Token量化存储后性能损失极小，可进一步优化存储-性能权衡

### 关键发现
- EDJE作为即插即用重排器，对所有测试的嵌入模型（CLIP/DFN/MetaCLIP/SigLIP2）都有检索提升
- 推理速度比BLIP-2快53×，存储减少48×（49kB vs 2,359kB/图）
- 量化压缩Token后性能下降极小，可进一步优化存储-性能权衡

## 亮点与洞察
- 精准识别视觉特征提取是联合编码器瓶颈（占83%-93%推理时间），提出的离线化+压缩方案简洁优雅
- Token压缩适配器的语义分析很有启发性：大部分ViT Token确实是冗余的，64个压缩Token足以捕获关键语义
- 设计具有高度模块化特性，作为drop-in重排器的实用价值极高
- 论文的写作逻辑极其清晰：从瓶颈分析→范式转换→具体设计→实验验证，层层递进
- 首次系统性地将文本检索中成熟的cross-encoder重排思路引入多模态检索，填补了重要空白

## 局限与展望
- 仅涵盖图文检索，未覆盖多语言多模态检索、音频、视频等模态
- 联合编码器的判别能力仍有提升空间，可探索更大或更强的语言模型替代MiniLM
- 压缩到32 Token时性能下降明显，更极致的压缩方法值得研究
- 对DFN和MetaCLIP的提升不如对CLIP和SigLIP2明显（DFN在Flickr上做过滤网络微调）
- 未探索零样本分类和大规模数据集过滤等联合编码器擅长的下游应用

## 相关工作与启发
- **与BLIP系列的关系**: EDJE可视为BLIP重排能力的高效替代，核心贡献是将视觉特征提取从在线移到离线
- **与ColBERT的联系**: 类似文本检索中ColBERT的token级离线存储思路，但增加了压缩维度
- **与Q-Former的关系**: Token压缩层与BLIP-2的Q-Former有相似性，但更轻量且专注于压缩而非生成
- **与LightningDOT对比**: LightningDOT用Region特征做重排但每个region压缩为单向量，本质更接近嵌入模型而非真正的联合编码器

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心idea（离线视觉+压缩Token）直觉清晰，但各组件均有先驱
- 实验充分度: ⭐⭐⭐⭐⭐ 跨多种骨干网络验证、详细消融、语义可视化、效率分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精准，动机阐述极其清晰，瓶颈分析有数据支撑
- 价值: ⭐⭐⭐⭐⭐ 高实用价值，填补了多模态检索中联合编码器重排器的空白

<!-- RELATED:START -->

## 相关论文

- [Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](../../ACL2025/information_retrieval/llm_reranking_harmful_content.md)
- [Joint Vision-Language Social Bias Removal for CLIP](../../CVPR2025/information_retrieval/joint_vision-language_social_bias_removal_for_clip.md)
- [TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)

<!-- RELATED:END -->
