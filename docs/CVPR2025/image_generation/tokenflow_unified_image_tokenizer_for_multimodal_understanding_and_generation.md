---
title: >-
  [论文解读] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
description: >-
  [CVPR 2025][图像生成][统一图像tokenizer] 提出TokenFlow统一图像tokenizer，通过双码本+共享映射架构解耦语义和像素级特征学习，首次实现离散视觉输入超越LLaVA-1.5 13B（+7.2%），同时在自回归生成中达到GenEval 0.55的SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - 统一图像tokenizer
  - 双码本
  - VQ编码
  - 多模态理解
  - 自回归生成
---

# TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation

**会议**: CVPR 2025  
**arXiv**: [2412.03069](https://arxiv.org/abs/2412.03069)  
**代码**: [GitHub](https://github.com/ByteVisionLab/TokenFlow)  
**领域**: Image Generation  
**关键词**: 统一图像tokenizer, 双码本, VQ编码, 多模态理解, 自回归生成

## 一句话总结

提出TokenFlow统一图像tokenizer，通过双码本+共享映射架构解耦语义和像素级特征学习，首次实现离散视觉输入超越LLaVA-1.5 13B（+7.2%），同时在自回归生成中达到GenEval 0.55的SOTA。

## 研究背景与动机

多模态统一（理解+生成）面临表征困境：
- **语义 vs 像素**：理解任务需要高层语义表征（如CLIP特征），生成任务需要精细像素级信息（如VQGAN token）
- **现有方案的trade-off**：
    - 重建导向的VQ编码器（VQGAN）：生成很好但理解崩溃（MME-P仅756 vs CLIP的1461）
    - 语义导向的VQ编码器（VQKD/CLIP蒸馏）：理解可以但重建严重模糊
    - Janus：双编码器增加复杂度但未根本解决

核心发现：如果tokenizer能把"语义相似且像素相似"的patch映射到同一索引，那么量化特征可同时用于理解和生成。

## 方法详解

### 整体框架

TokenFlow采用双编码器+双码本+共享索引架构：
1. 语义编码器 $\mathcal{E}_{sem}$（CLIP初始化）提取高层特征
2. 像素编码器 $\mathcal{E}_{pix}$ 提取低层特征
3. 双码本共享索引映射：通过最小化加权距离和选择索引
4. 双解码器分别用于语义对齐和图像重建
5. 量化特征拼接后用于下游理解和生成

### 关键设计1：共享映射的双码本量化

- **功能**：在单一索引空间中同时编码语义和像素信息
- **核心思路**：维护两个码本 $\mathbf{Z}_{sem} \in \mathbb{R}^{K \times d_{sem}}$ 和 $\mathbf{Z}_{pix} \in \mathbb{R}^{K \times d_{pix}}$，共享大小 $K$。量化索引通过联合距离最小化选择：$i^* = \arg\min_i (d_{sem,i} + w_{dis} \cdot d_{pix,i})$
- **设计动机**：单码本方法（如VQGAN或VQKD）要么丢失语义要么丢失像素细节。双码本共享索引意味着同一个index既能检索到语义嵌入（用于理解），也能检索到像素嵌入（用于重建），实现"一个索引两种用途"。可视化验证：TokenFlow的聚类同时体现语义和视觉相似性

### 关键设计2：多尺度VQ + 大码本高利用率

- **功能**：提供丰富的码本表征能力和高利用率
- **核心思路**：采用MSVQ（多尺度VQ）结构做next-scale prediction。码本大小扩展到131K条目时仍维持95%+利用率
- **设计动机**：大码本提供更多语义-像素组合可能性，但传统VQ方法在大码本下利用率骤降。共享映射策略天然促进码本条目的充分利用——需要同时在两个距离上接近才被选中，减少了冗余/死码

### 关键设计3：多步采样推理策略

- **功能**：解决next-scale paradigm中单次top-k采样导致的图像崩溃和重复纹理
- **核心思路**：两步采样——先用大的$(k_1, p_1)$采样获得多样性，再用小的$(k_2, p_2)$对同一scale精炼一致性
- **设计动机**：交叉熵训练目标主要与top-1预测建立attention关系，独立top-k采样可能产生互不相关的token。渐进缩小采样空间保持创意多样性同时强化一致性

### 损失函数

$\mathcal{L}_{total} = \mathcal{L}_{sem} + \mathcal{L}_{VQ} + \mathcal{L}_{pix}$，其中像素损失包含$\ell_2$重建+LPIPS感知+GAN对抗。

## 实验关键数据

### 主实验1：多模态理解（LLaVA-1.5框架）

| 方法 | 类型 | MME-P↑ | SEED-B↑ | TextVQA↑ |
|------|------|--------|---------|----------|
| CLIP ViT-B/14 (连续) | Sem. | 1460.9 | 64.1 | 53.4 |
| VQGAN | Pix. | 756.1 | 38.2 | 46.8 |
| VQKD | Sem. | 1252.4 | 57.8 | 48.2 |
| **TokenFlow** | **统一** | **超越LLaVA-1.5 13B** | **+7.2%平均** | - |

首次证明离散视觉输入可以超越连续CLIP在理解任务上的表现。

### 主实验2：图像重建与生成

| 指标 | 数值 |
|------|------|
| rFID@384×384 | **0.63** |
| GenEval@256×256 | **0.55**（自回归SOTA） |
| 码本利用率 | **95%+**（131K条目） |

### 关键发现

- 理解只需最终尺度特征（残差/所有尺度反而引入噪声）
- 码本大小增加持续提升理解和生成性能（unique to TokenFlow）
- 多步采样显著优于单步top-k采样
- 训练仅需8×A100 GPU不到24小时（理解微调部分）

## 亮点与洞察

1. **一个索引两种用途**：共享映射是连接理解与生成的优雅桥梁
2. **大码本可行**：95%+利用率在131K规模下前所未有，说明联合距离选择天然防止码本崩溃
3. **首次离散超连续**：打破了"离散token理解必弱于连续特征"的固有认知

## 局限与展望

- 生成分辨率仅256×256，高分辨率生成能力待验证
- 双编码器+双码本+双解码器的参数量和复杂度较高
- 语义编码器依赖CLIP初始化，对CLIP覆盖不好的领域可能泛化受限
- 未探索视频理解和生成的统一

## 相关工作与启发

- **Chameleon / EMU3**：使用单一VQ tokenizer的统一方法，理解受限
- **Janus**：双编码器但非共享映射，复杂度高且未根本解决表征冲突
- **LlamaGen**：自回归图像生成基线，TokenFlow在GenEval上超越

## 评分

⭐⭐⭐⭐⭐ — 双码本共享映射是精妙的设计，首次实现离散视觉超越LLaVA-1.5 13B具有里程碑意义。理论动机清晰、实验全面、生成和理解双SOTA。分辨率限制是主要提升空间。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](../../NeurIPS2025/image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2025\] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)

<!-- RELATED:END -->
