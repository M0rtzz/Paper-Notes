---
title: "DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos"
description: "提出DeCafNet，采用delegate-and-conquer策略，引入高效sidekick encoder生成显著性图筛选关键clip，仅对top-c%的clip使用expert encoder处理，配合DeCaf-Grounder统一多分辨率特征，在LVTG任务上以47%的计算量节省实现SOTA性能。"
tags: ["长视频时序定位", "高效推理", "双编码器", "显著性选择", "多尺度特征"]
---

# DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos

**会议**: CVPR 2025  
**arXiv**: [2505.16376](https://arxiv.org/abs/2505.16376)  
**代码**: https://github.com/ZijiaLewisLu/CVPR2025-DeCafNet (有)  
**领域**: 视频理解  
**关键词**: 长视频时序定位, 高效推理, 双编码器, 显著性选择, 多尺度时序精炼

## 一句话总结

提出DeCafNet，通过**delegate-and-conquer双编码器策略**（轻量sidekick encoder密集提特征+生成显著性图，expert encoder仅处理top-c%关键clip），配合**DeCaf-Grounder**统一不同时序分辨率特征，在长视频时序定位任务上以**减少47% TFLOPs**的代价超越所有先前方法。

## 研究背景与动机

长视频时序定位（LVTG）旨在从数分钟到数小时的长视频中，根据用户文本查询定位对应的时间片段，应用于视频摘要、内容推荐、监控等场景。

**现有痛点**：当前SOTA方法（如RGNet、SnAG）遵循"切片→逐片过expert encoder→grounding"的两阶段范式。短视频时这一方案可行，但长视频中clip数量巨大（可达数百个），逐个用大规模预训练expert encoder处理带来了**计算量爆炸**的问题——TFLOPs高达668、GPU内存224G、推理17秒。

**核心矛盾**：长视频中，查询相关的时刻通常只占整个视频的极小比例（Ego4D-NLQ中平均仅1.7%），大量clip与查询无关，却消耗了同等的计算资源。

**本文切入角度**：受"计算异质性"启发——不是所有时间位置都同等复杂或同等重要——本文提出delegate-and-conquer策略：将大部分计算**委托**给一个高效的sidekick encoder，仅对最显著的clip使用expert encoder进行精细处理，从而**攻克**计算瓶颈。

## 方法详解

### 整体框架

DeCafNet由三部分组成：(1) **Sidekick Encoder** $\Psi_D$：高效提取所有clip的密集特征，并生成显著性图；(2) **Expert Encoder** $\Psi_E$：仅处理显著性图筛选出的top-c%关键clip；(3) **DeCaf-Grounder**：通过query-aware时序聚合和多尺度时序精炼，统一两个编码器的不同分辨率特征，预测时间区间。

### 关键设计

1. **Sidekick Encoder（轻量级副手编码器）**
    - **功能**：以极低计算成本为所有clip提取密集特征，并生成文本查询相关的显著性图
    - **核心思路**：基于ViT架构，在transformer block前插入时空卷积池化层，将时间维度L和空间维度(H,W)各缩减4倍，大幅降低后续block的计算量。同时利用**时序插值**——每隔$\tau=2$个clip采样一次，未采样clip通过FFN从相邻clip特征插值得到
    - **设计动机**：长视频中相邻clip内容高度相似，可从邻近clip推断而无需从头计算，结合卷积池化实现31倍TFLOPs缩减
    - **损失函数**：采用**saliency loss**（对比学习对齐视频-文本特征）和**distillation loss**（L2损失蒸馏expert encoder特征），确保轻量编码器的特征质量

2. **显著性选择机制（Saliency Selection）**
    - **功能**：根据查询文本动态筛选最相关的clip，减少expert encoder的处理量
    - **核心思路**：计算sidekick encoder的密集特征$\mathbf{F}_D$与文本查询CLS token $\mathbf{q}_{cls}$的内积得到显著性分数$\mathbf{S}=\mathbf{F}_D \cdot \mathbf{q}_{cls}$，选取top-c%的clip送入expert encoder
    - **设计动机**：平衡信息保留与计算效率——sidekick encoder因池化不可避免地丢失信息，expert encoder仍需为关键clip提供高质量细粒度特征

3. **DeCaf-Grounder（统一精炼模块）**
    - **功能**：统一来自两个编码器的不同时序分辨率特征，执行精确的时序定位
    - **核心思路**：
     - **Query-aware时序聚合**：对非显著clip在$\mathbf{F}_S$中做zero-padding对齐时序维度，拼接$\mathbf{F}_D$、$\hat{\mathbf{F}}_S$和显著性分数$\mathbf{S}$，再通过视频-文本交叉注意力增强查询相关信息
     - **多尺度时序精炼**：通过temporal transformer构建L=8级特征金字塔（每级时序长度减半），用FFN分类器生成各尺度的置信分数，再通过扩张时序卷积跨尺度同步定位信息
    - **设计动机**：两个编码器的特征存在不同的时序分辨率和语义粒度，标准grounding模块直接使用效果欠佳，需要专门的聚合-精炼架构
    - **损失函数**：使用Focal loss + Distance-IoU loss训练分类和回归头

## 实验关键数据

### 主实验

| 数据集 | 方法 | R1@0.3 | R1@0.5 | R5@0.3 | R5@0.5 | AVG |
|---------|------|--------|--------|--------|--------|-----|
| Ego4D-NLQ | SnAG | 15.87 | 11.26 | 38.26 | 27.16 | 23.14 |
| | RGNet | 18.28 | 12.04 | 34.02 | 22.89 | 21.81 |
| | **DeCafNet-50%** | **18.10** | **12.55** | **38.85** | **28.27** | **24.44** |
| Ego4D-Goalstep | SnAG | 18.34 | 15.12 | 45.95 | 38.55 | 29.49 |
| | **DeCafNet-50%** | **21.29** | **17.46** | **47.27** | **40.40** | **31.61** |
| MAD | SnAG | 10.28 | 8.46 | 24.42 | 20.60 | 13.84 |
| | **DeCafNet** | **13.25** | **10.96** | **27.73** | **23.68** | **16.47** |

### 计算效率对比

| 编码器配置 | TFLOPs | GPU内存(G) | 推理时间(s) |
|-----------|--------|-----------|------------|
| 仅$\Psi_E$（先前方法） | 668.2 | 224.2 | 17.1 |
| DeCafNet-50% | 355.7 (**↓47%**) | 126.2 (**↓44%**) | 8.4 (**↓51%**) |
| DeCafNet-30% | 222.1 (**↓66%**) | 79.9 (**↓65%**) | 5.7 (**↓67%**) |
| 仅$\Psi_D$ | 21.6 | 10.9 | 0.6 |

### 消融实验

- 仅使用$\mathbf{F}_D$：AVG=21.51；仅使用$\mathbf{F}_S$：AVG=22.57；两者结合：AVG=23.91；再加显著性分数$\mathbf{S}$：**AVG=24.41**，三种特征互补
- Sidekick encoder比随机选择和均匀选择在显著性选择上平均高出1.8%+
- 在短视频数据集上DeCaf-Grounder也超越SnAG：Charades-STA +1.37%，TACoS +0.81%

### 关键发现

- 长视频中查询关联片段极稀少（Ego4D-NLQ仅1.7%），支持aggressive clip筛选
- Sidekick encoder实现31倍TFLOPs和22倍GPU内存的缩减
- 仅处理30% clip即可匹配先前SOTA，处理50% clip即全面超越

## 亮点与洞察

- **Delegate-and-conquer思想**极具通用性：将昂贵计算委托给轻量代理，仅在关键位置投入重量级资源，可推广到其他长序列处理场景
- 显著性选择是**query-dependent**的，不同查询会选择不同clip，比固定采样策略更加灵活
- DeCaf-Grounder在仅使用expert encoder特征（MAD数据集）时仍大幅超越先前方法，说明其多尺度精炼架构本身也具有独立价值

## 局限性

- Sidekick encoder需要额外训练（两个loss + 蒸馏），增加了训练复杂度
- 显著性选择比例c%是固定超参数，未能根据视频内容自适应调整
- 目前仅在第一人称视频（Ego4D）和电影（MAD）上验证，更多类型长视频场景有待探索

## 相关工作与启发

- **短视频时序定位**（SVTG）的方法（如Moment-DETR直接预测时间点）无法扩展到长视频场景
- **CONE**的粗到细策略和**SnAG**的late fusion为LVTG提供了基础，但都忽略了特征提取的计算开销
- 本文的delegate-and-conquer思想启发：在高效推理领域，不一定需要压缩单个模型，而可以**设计计算资源的动态分配策略**

## 评分

⭐⭐⭐⭐ — 方法设计简洁有效，在效率-性能权衡上达到了新的pareto最优。双编码器+显著性选择的框架具有很好的通用性和工程实用价值，三个LVTG基准上全面超越SOTA且大幅降低计算量。

<!-- RELATED:START -->

## 相关论文

- [VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](../../ECCV2024/video_understanding/rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](../../ICCV2025/video_understanding/videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)

<!-- RELATED:END -->
