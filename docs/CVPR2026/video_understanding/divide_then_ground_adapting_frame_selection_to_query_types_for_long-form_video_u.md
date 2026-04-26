---
title: >-
  [论文解读] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding
description: >-
  [CVPR 2026][视频理解][长视频理解] 提出 DIG，一个免训练的帧选择框架，通过将查询分为全局查询和定位查询两类，对全局查询使用均匀采样、对定位查询使用一套专门的内容自适应帧选择+LMM奖励评分+视频精炼流水线，在三个长视频理解基准上持续超越现有方法。
tags:
  - CVPR 2026
  - 视频理解
  - 长视频理解
  - 帧选择
  - 查询分类
  - 内容自适应采样
  - 大多模态模型
---

# DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2512.04000](https://arxiv.org/abs/2512.04000)  
**代码**: [GitHub](https://github.com/Jialuo-Li/DIG)  
**领域**: Video Understanding  
**关键词**: 长视频理解, 帧选择, 查询分类, 内容自适应采样, 大多模态模型

## 一句话总结

提出 DIG，一个免训练的帧选择框架，通过将查询分为全局查询和定位查询两类，对全局查询使用均匀采样、对定位查询使用一套专门的内容自适应帧选择+LMM奖励评分+视频精炼流水线，在三个长视频理解基准上持续超越现有方法。

## 研究背景与动机

大多模态模型（LMM）在视频理解中面临一个核心矛盾：视频token量巨大但LLM上下文长度有限，只能输入采样后的帧子集。均匀采样虽然最大化时间覆盖，但完全与查询无关。

现有工作提出了查询感知的自适应帧选择机制（如AKS、Q-Frame），但计算开销大。作者提出了一个**被频繁忽视的关键问题**：

> 复杂的搜索机制是否对所有查询类型都必要？答案是**否**。

作者的核心发现驱动了整个方法设计：

**全局查询**（如"这段视频的主题是什么？"）：需要全面的视频信息，均匀采样已经足够好且高效

**定位查询**（如"那个人骑的是什么车？"）：针对特定时间段，均匀采样会注入大量无关帧导致性能下降

**更多帧≠更好性能**：实验发现所有LMM的准确率在某个最优帧数后反而下降，且下降主要由定位查询贡献

## 方法详解

### 整体框架

DIG包含两条路径：
- **全局查询路径**：LLM分类→均匀采样→LMM推理
- **定位查询路径**：LLM分类→CAFS内容自适应帧选择→LMM奖励评分→视频精炼→均匀采样→LMM推理

### 关键设计

1. **查询类型识别（§4.1）**：

    - 用LLM（Qwen3-Next-80B-A3B）将查询自动分类为全局或定位
    - 全局查询直接走均匀采样路径，避免不必要的计算开销
    - **设计动机**：实验证明全局查询中复杂帧选择的收益可忽略甚至为负

2. **内容自适应帧选择（CAFS, §4.2）**：

    - 先以2fps采样获得 $M$ 帧，用DINOv2提取特征
    - 计算相邻帧的余弦距离序列：$d_i = 1 - \text{sim}(V_{I_i}, V_{I_{i+1}})$
    - 检测距离序列中的峰值（prominence > 0.1）作为场景分割点
    - 在每个分割段内选取中点帧作为**代表帧（r-frame）**
    - **设计动机**：解决静态采样的两难——低帧率遗漏关键事件，高帧率产生冗余。CAFS自适应视频内容的语义变化频率

3. **奖励评分（§4.3）**：

    - 直接利用LMM（而非CLIPScore等表面特征匹配方法）评估r-frame与查询的相关性
    - **二维评分设计**：(1) 当前帧与查询的直接相关性；(2) 当前帧内容是否暗示相邻帧可能包含补充信息
    - **设计动机**：CLIPScore对复杂推理不可靠，LMM具备更深层的上下文推理能力

4. **视频精炼（§4.4）**：

    - **迭代奖励引导选择**：参数无关的方法，迭代减去均值直到集合稳定——保留奖励始终高于平均的r-frame
    - **段落合并**：对选中的r-frame，不仅使用单帧而是取其所在分割段及窗口 $w_{len}$ 内的相邻段，合并成连续视频段
    - 从精炼后的视频中均匀采样作为最终输入
    - **设计动机**：Top-K需要固定超参数，迭代方法自适应不同视频；段落合并保留连续细粒度信息而非稀疏帧

### 损失函数 / 训练策略

DIG是完全**免训练**的框架，不涉及任何训练过程。所有组件（DINOv2、LLM分类器、LMM评分器）使用现成预训练模型。超参数设置：每帧56 token，$w_{len}=2$。

## 实验关键数据

### 主实验

**Qwen2.5-VL-32B 作为基础LMM：**

| 方法 | #帧 | MLVU | LVB | VideoMME-Med | VideoMME-Long |
|------|-----|------|-----|-------------|--------------|
| UNI | 32 | 61.91 | 57.89 | 57.89 | 53.33 |
| AKS | 32 | 66.42 | 59.31 | 59.89 | 56.00 |
| Q-Frame | 32 | 60.95 | 57.37 | 60.43 | 55.90 |
| **DIG** | 32 | **70.69** | **61.86** | **60.87** | **57.76** |

**Qwen2.5-VL-7B 作为基础LMM：**

| 方法 | #帧 | MLVU | LVB | VideoMME-Med | VideoMME-Long |
|------|-----|------|-----|-------------|--------------|
| UNI | 32 | 59.52 | 56.92 | 59.08 | 52.02 |
| **DIG** | 32 | **67.20** | **60.43** | **61.62** | **53.24** |

**高帧数扩展性（7B模型）：**

| 方法 | #帧 | MLVU | LVB |
|------|-----|------|-----|
| UNI | 256 | 69.15 | 61.48 |
| AKS | 256 | 71.50 | 61.03 |
| **DIG** | 256 | **72.46** | **64.62** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| UNI on GQ vs DIG pipeline on GQ | 性能相当 | 全局查询无需复杂帧选择 |
| UNI on LQ vs DIG pipeline on LQ | DIG显著领先 | 定位查询是DIG的主要收益来源 |
| CLIPScore奖励 vs LMM奖励 | LMM奖励+1~3% | LMM评分在复杂推理中更可靠 |
| 帧选择从8扩展到256 | DIG持续领先 | AKS和Q-Frame在高帧数时可能低于UNI |

### 关键发现

1. **32帧下DIG比均匀采样提升7.68%（MLVU）和4.51%（LVB）**
2. **查询类型分类是关键**：全局查询用均匀采样最优，定位查询才需复杂搜索
3. **帧数增加的性能下降主要来自定位查询**：全局查询对帧数增加保持稳定
4. **高帧数下DIG保持优势**：AKS和Q-Frame在128+帧时可能退化到不如均匀采样
5. **CAFS比固定帧率采样更高效**：用更少的帧达到更高的覆盖度

## 亮点与洞察

- **简洁有力的核心洞察**：不是所有查询都需要复杂帧选择——这个观察简单但被普遍忽视
- **免训练设计**：完全利用现成模型，无需任何额外训练，部署成本极低
- **内容自适应采样**：基于DINOv2特征距离的峰值检测是一种优雅的视频分割方案
- **迭代奖励引导选择**：参数无关的选择算法，避免了Top-K中K值的超参数调优
- **二维LMM评分**：不仅评估当前帧的直接相关性，还考虑相邻帧的潜在补充价值
- **可扩展性强**：从8帧到256帧均保持稳定增益

## 局限与展望

1. **查询分类依赖LLM**：分类准确率直接影响后续流水线，分类错误时可能比均匀采样更差
2. **计算开销**：虽然全局查询路径高效，但定位查询需要DINOv2特征提取+LMM评分+视频精炼，总开销仍不小
3. **二分法可能过于简化**：有些查询介于全局和定位之间（如"视频前半段和后半段的情绪变化"），更细粒度的类型划分值得探索
4. **r-frame选择的峰值prominence阈值（0.1）固定**：可能不适合所有视频类型（如慢镜头vs快剪辑）
5. **仅在VQA任务验证**：视频摘要、视频字幕等其他视频理解任务的适用性待验证

## 相关工作与启发

- 与AKS、Q-Frame等当前SOTA帧选择方法直接对比，展示了策略自适应的优势
- DINOv2作为视觉特征提取器的通用性再次得到验证
- LMM自身作为评分器（而非依赖CLIP）是一个有前景的趋势
- 查询分类+策略路由的设计模式可推广到其他多模态理解任务
- 段落合并的思想在视频检索、视频摘要中也有应用潜力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 核心洞察（查询类型决定最优策略）简洁有力
- 实验充分度: ⭐⭐⭐⭐⭐ — 三基准、两LMM、8~256帧全覆盖，含充分消融
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，实验图表丰富
- 价值: ⭐⭐⭐⭐ — 实用性强，可直接部署到现有LMM视频理解流水线

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](../../ICCV2025/video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)

<!-- RELATED:END -->
