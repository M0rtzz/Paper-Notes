---
title: >-
  [论文解读] SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection
description: >-
  [CVPR 2025][人体理解][开放词汇HOI检测] 提出分层粒度比较网络SGC-Net，通过粒度感知对齐(GSA)模块聚合CLIP多层视觉特征，并利用层级分组比较(HGC)模块借助LLM递归生成区分性描述，解决开放词汇HOI检测中的特征粒度不足和语义混淆问题。
tags:
  - CVPR 2025
  - 人体理解
  - 开放词汇HOI检测
  - 多粒度特征对齐
  - 层级分组比较
  - CLIP适配
  - LLM辅助分类
---

# SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection

**会议**: CVPR 2025  
**arXiv**: [2503.00414](https://arxiv.org/abs/2503.00414)  
**代码**: 无（论文声明将在GitHub发布）  
**领域**: Human Understanding / HOI Detection  
**关键词**: 开放词汇HOI检测, 多粒度特征对齐, 层级分组比较, CLIP适配, LLM辅助分类

## 一句话总结

提出分层粒度比较网络SGC-Net，通过粒度感知对齐(GSA)模块聚合CLIP多层视觉特征，并利用层级分组比较(HGC)模块借助LLM递归生成区分性描述，解决开放词汇HOI检测中的特征粒度不足和语义混淆问题。

## 研究背景与动机

开放词汇HOI检测（OV-HOI）需要在只使用基类交互类别训练的情况下识别新类别交互。现有CLIP-based方法面临两个核心问题：

1. **特征粒度不足**: 现有方法主要依赖CLIP最后一层视觉特征与文本对齐，但最后一层关注高层语义而忽略中间层捕获的局部细节（如手臂姿态、面部表情），而这些细节对HOI检测至关重要。

2. **语义相似混淆**: CLIP在大规模长尾数据上训练导致对某些类别有偏见（如混淆"hug cat"和"hold cat"），而LLM仅基于标签生成的描述难以充分区分语义相似的交互类别（如"hold cat"和"chase cat"都被描述为"手臂伸展"）。

现有方法CMD-SE虽尝试利用中间层特征，但其损失函数需要最小化连续变量与离散变量的差异，难以优化。

## 方法详解

### 整体框架

SGC-Net是端到端的OV-HOI检测网络，无需预训练物体检测器。包含两个核心模块：(1) 粒度感知对齐(GSA)模块——将CLIP视觉编码器分块，用距离感知高斯权重聚合多粒度特征；(2) 层级分组比较(HGC)模块——利用LLM递归构建类别层次，在每个层级比较HOI表示与文本嵌入。

### 关键设计1：粒度感知对齐(GSA)模块

- **功能**: 有效聚合CLIP多层视觉特征的局部细节与全局语义
- **核心思路**: 将CLIP的12层视觉编码器分为$S$个块（如{6-8}, {9-11}, {12}），每个块内用距离感知高斯权重（可训练的$\sigma$）融合层特征：$\alpha_l^s = \exp(-\frac{(d-l)^2}{2\sigma^2})$。块间同样加权聚合，最后一层作为独立块赋予较大权重以保留预训练的视觉-语言对齐。同时使用视觉提示调优引入可学习token促进中间层与文本的对齐
- **设计动机**: 直接聚合浅层和深层特征会破坏CLIP预训练的视觉-语言对齐。分块策略确保块内特征差异小从而安全融合，而高斯权重允许自适应学习层级信息。相比CMD-SE更易优化且保留了预训练对齐

### 关键设计2：层级分组比较(HGC)模块

- **功能**: 递归生成区分性文本描述，解决语义相似类别的混淆
- **核心思路**: 三步流程：(a) **分组**——用K-means聚类LLM生成的初始描述的CLIP文本特征；(b) **比较**——对大组用LLM总结组特征后生成对比描述，对小组直接查询LLM进行类别间比较；(c) **层级分类**——从顶到底遍历类别层次，在每层比较HOI特征与文本嵌入，用迭代评估器$u_i^k = \mathbb{I}(p_i^{k+1} > p_i^k + \tau)$过滤不可靠的低层描述
- **设计动机**: 类别数量多导致全对比较的描述矩阵呈二次增长，分组策略在保持区分力的同时控制了复杂度。递归比较确保了从粗到细的判别边界逐步细化

### 关键设计3：迭代评估器与融合策略

- **功能**: 自适应选择层级描述中最有信息量的部分
- **核心思路**: 通过监测分数单调递增序列，计算running average $r(\boldsymbol{x}, i)$，最终融合为$s(\boldsymbol{x}, i) = (1-\lambda)(p_i^1 + \boldsymbol{t} \cdot \boldsymbol{x}^T) + \lambda \cdot r(\boldsymbol{x}, i)$
- **设计动机**: 不可靠的低层描述会引入误差和冗余，自动评估和过滤机制确保只有真正有鉴别力的描述被使用

### 损失函数

$\mathcal{L} = \lambda_b \sum_{i \in \{h,o\}} \mathcal{L}_b^i + \lambda_{iou} \sum_{i \in \{h,o\}} \mathcal{L}_{iou}^i + \lambda_{cls} \mathcal{L}_{cls}$，包括人/物体的边界框回归损失、IoU损失和交互分类损失，使用匈牙利算法进行标签匹配。$\lambda_b=5, \lambda_{cls}=2, \lambda_{iou}=5$。

## 实验关键数据

### 主实验：HICO-DET数据集（不使用预训练检测器）

| 方法 | 预训练检测器 | Unseen | Seen | Full |
|------|:----------:|--------|------|------|
| THID | ✗ | 15.53 | 24.32 | 22.38 |
| CMD-SE | ✗ | 16.70 | 23.95 | 22.35 |
| **SGC-Net** | **✗** | **23.27** | **28.34** | **27.22** |
| HOICLIP | ✓ | 23.48 | 34.47 | 32.26 |

### 主实验：SWIG-HOI数据集

| 方法 | Non-rare | Rare | Unseen | Full |
|------|----------|------|--------|------|
| CMD-SE | 21.46 | 14.64 | 10.70 | 15.26 |
| **SGC-Net** | **23.67** | **16.55** | **12.46** | **17.20** |

### 消融实验

| 配置 | Non-rare | Rare | Unseen | Full |
|------|----------|------|--------|------|
| Base | 15.69 | 11.53 | 7.32 | 11.45 |
| + GSA | 22.74 | 16.00 | 11.64 | 16.49 |
| + HGC | 21.18 | 14.19 | 10.69 | 14.81 |
| **SGC-Net** | **23.67** | **16.55** | **12.46** | **17.20** |

### 关键发现

- GSA模块贡献最大（+5.04 Full mAP），说明多粒度特征聚合对OV-HOI至关重要
- 不使用预训练检测器的SGC-Net在Unseen类别上接近甚至匹配使用预训练检测器的方法
- 最优分块策略为{6-8}, {9-11}, {12}，最后一层单独成块保留CLIP预训练对齐
- 使用3个块比1个或2个块效果显著更好（Full: 17.20 vs 14.81/14.78）

## 亮点与洞察

1. **多粒度与对齐的优雅平衡**: 分块+高斯权重策略既利用了中间层细节又保留了CLIP对齐，比CMD-SE的方案更简洁且更有效
2. **LLM的递归比较策略**: 通过分组-比较-层级化三步，将$O(n^2)$的描述生成复杂度降到可控水平
3. **迭代评估器的自适应过滤**: 自动识别并仅使用有效的层级描述，避免了噪声传播

## 局限与展望

- SWIG-HOI上的绝对性能仍然较低（Full仅17.20），说明大词汇量OV-HOI仍是挑战
- LLM生成的描述质量受限于提示工程，不同LLM可能产生不同效果
- 层级分类的递归深度受限，过深可能引入噪声

## 相关工作与启发

- 多粒度特征聚合的思路可推广到其他需要CLIP适配的视觉任务
- LLM辅助的类别比较策略对细粒度分类任务有参考价值

## 评分

⭐⭐⭐⭐ — 两个模块设计都有清晰的问题驱动和优雅的解决方案，在不使用预训练检测器的情况下达到了有竞争力的性能。消融实验充分验证了各组件的贡献。

<!-- RELATED:START -->

## 相关论文

- [Anomize: Better Open Vocabulary Video Anomaly Detection](anomize_better_open_vocabulary_video_anomaly_detection.md)
- [MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network](../../AAAI2026/human_understanding/mvgd-net_a_novel_motion-aware_video_glass_surface_detection_network.md)
- [Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection](forensics_adapter_adapting_clip_for_generalizable_face_forgery_detection.md)
- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](../../NeurIPS2025/human_understanding/hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)

<!-- RELATED:END -->
