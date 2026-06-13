---
title: >-
  [论文解读] Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks
description: >-
  [CVPR 2025][视频理解][视频基准去偏] 提出 UTD 方法，利用 VLM+LLM 生成视频帧文本描述来系统分析视频基准中的对象/时序/常识偏差，并构建去偏测试集，使视频理解评估更加鲁棒公正。 1. 领域现状：视频理解依赖大量基准数据集（UCF101、Kinetics等）来评估模型性能，但已有研究指出这些基准可能存…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频基准去偏"
  - "表征偏差"
  - "文本描述"
  - "VLM/LLM"
  - "数据集分析"
---

# Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks

**会议**: CVPR 2025  
**arXiv**: [2503.18637](https://arxiv.org/abs/2503.18637)  
**代码**: [https://utd-project.github.io/](https://utd-project.github.io/) (项目页)  
**领域**: 视频理解  
**关键词**: 视频基准去偏、表征偏差、文本描述、VLM/LLM、数据集分析

## 一句话总结

提出 UTD 方法，利用 VLM+LLM 生成视频帧文本描述来系统分析视频基准中的对象/时序/常识偏差，并构建去偏测试集，使视频理解评估更加鲁棒公正。

## 研究背景与动机

1. **领域现状**：视频理解依赖大量基准数据集（UCF101、Kinetics等）来评估模型性能，但已有研究指出这些基准可能存在表征偏差——仅识别物体或只看单帧即可正确预测。
2. **现有痛点**：部分去偏方案（如人物分割+黑背景替换、动作迁移到无关场景）会引入视觉伪影或域偏移，导致去偏数据集很少被主流采用。而关注时序偏差的工作主要在构建新benchmark，未系统分析已有基准。
3. **核心矛盾**：物体表征偏差让有强目标识别能力的模型占便宜，掩盖了模型在时序推理和动作理解上的真实能力，影响评估的公平性。
4. **本文目标**：（1）设计一套可扩展的自动化方法来衡量和分析视频基准中的各类表征偏差；（2）为现有基准构建去偏测试集；（3）系统评估30个SOTA模型。
5. **切入角度**：既然直接改视频会产生伪影，不如用文本描述作为中间表征——通过VLM生成帧描述，再用LLM提取特定概念（仅物体/仅动作/仅动词），从而精确控制表征内容。
6. **核心 idea**：用文本描述作为代理表征来测量偏差，并通过排除高偏差样本构建去偏split，无需修改任何视频数据。

## 方法详解

### 整体框架

输入视频帧序列 → VLM（LLaVA-1.6-Mistral-7B）生成逐帧文本描述 → LLM 提取不同概念类别（objects/activities/verbs） → 组合为不同时序配置（单帧/均值/序列） → 使用文本嵌入模型计算偏差指标 → 排除高偏差样本构建去偏split。

### 关键设计

1. **基于文本描述的表征偏差分析框架**

    - 功能：精确测量三个独立维度的表征偏差——概念偏差（仅物体是否足够预测）、时序偏差（单帧是否足够）、常识vs数据集偏差
    - 核心思路：用 VLM 对每帧生成详细描述 $d_{n,i} = d(f_{n,i})$，再用 LLM 从描述中按 prompt 分别提取物体列表 $o_{n,i}$、活动 $a_{n,i}$、纯动词 $\nu_{n,i}$。这些文本描述可以组合成不同时序配置：中间帧、最大分数帧、帧平均、帧序列（"Frame 1: ... Frame 2: ..."）。然后用预训练文本嵌入模型（如 E5）测零样本分类/检索性能作为偏差度量 $M(D, \phi)$。
    - 设计动机：相比直接裁剪/修补视频中的特定内容，文本描述可以更干净地隔离特定概念（物体/动作），避免信息泄漏和伪影；且文本形式天然适配 LLM 的零样本推理能力。

2. **常识偏差 vs 数据集偏差分离**

    - 功能：区分"看到钢琴就猜弹钢琴"这种合理常识推理，和"化妆视频中总出现镜子和花"这种数据集虚假相关
    - 核心思路：常识偏差用零样本文本嵌入模型衡量（不接触训练集），数据集偏差则在训练集文本嵌入上训练线性分类器衡量。两者之差即纯数据集偏差。实验发现训练后性能大幅提升（如 UCF101 objects 从 63.3% 到 80.3%），说明数据集中存在大量虚假物体-标签关联。
    - 设计动机：常识偏差本身可能合理（日常生活中object和action确实有关联），但数据集偏差是有害的虚假相关，需要区分对待。

3. **基于一致性投票的自动去偏策略**

    - 功能：自动识别并排除高对象偏差的测试样本，构建去偏评测集
    - 核心思路：使用3种不同 prompt 的文本嵌入模型 × 3个bootstrapped训练集 = 9个模型。对每个样本，只有当9个模型一致判断"仅凭物体即可正确分类/检索"时才将其排除。排除比例由偏差严重程度自动决定。还额外构建了类别平衡版本的去偏split。
    - 设计动机：多模型投票确保去偏结果的稳定性，避免因单一模型的随机波动而错误排除正常样本。

### 损失函数 / 训练策略

本文不涉及端到端训练新模型——而是一个分析框架。线性分类器用标准交叉熵训练，文本嵌入模型使用预训练权重（E5-large-v2）。关键参数是投票阈值（9/9全部一致才排除）。

## 实验关键数据

### 主实验

分析覆盖12个视频数据集，包括6个分类（UCF101, SSv2, K400/600/700, MiT）和6个检索（MSRVTT, DiDeMo, ActivityNet, LSMDC, YouCook2, S-MiT）。

| 数据集 | 物体偏差(seq) | 活动偏差(seq) | 动词偏差(seq) | 去偏后保留% |
|--------|-------------|-------------|-------------|-----------|
| UCF101 | 63.3% | 67.4% | 50.8% | 27% |
| SSv2 | 5.3% | 6.4% | 5.8% | 93% |
| K400 | 45.9% | 45.2% | 24.8% | 45% |
| K700 | 37.0% | 36.7% | 17.6% | 52% |
| MiT | 21.0% | 21.0% | 16.2% | 76% |

### 消融实验

| 配置 | UCF101 原始 | UCF101 UTD-split | 性能下降 |
|------|-----------|-----------------|---------|
| VideoMAE-B-K400 | 89.2 | 81.3 | -7.9 |
| VideoMAE-B-UH | 88.4 | 79.2 | -9.2 |
| VideoMAE2-B | 89.7 | 82.5 | -7.2 |
| InternVid-B | 92.1 | 82.2 | -9.9 |

### 关键发现

- UCF101 偏差最严重（73% 样本有物体偏差），SSv2 最干净（仅 7%），符合 SSv2 focuses on manipulative actions 的设计意图
- 仅看物体即可在 UCF101 上达到 63.3% 准确率（SSv2 仅5.3%），说明很多基准的"视频理解能力"其实是在考物体识别
- 数据集偏差远超常识偏差（UCF101 objects: 63.3→80.3, +17），说明训练集中存在大量虚假物体-标签关联
- 去偏后所有30个模型的性能均下降，InternVid 等大模型降幅更大（-9.9），说明大模型可能更善于利用物体偏差走捷径

## 亮点与洞察

- **文本作为中间表征的思路非常巧妙**：用文本描述精确隔离物体/动作/动词等概念，比像素级操作（裁剪、修补）更干净。这个idea可以通用化——值得尝试用文本描述来分析其他模态（如图像基准/多模态基准）中的偏差。
- **9模型一致性投票的鲁棒去偏策略**：避免了人工设定阈值，让去偏比例由数据本身决定。
- **SSv2 的特殊性被量化证实**：从数据角度解释了为什么 SSv2 上各模型表现差异和其他数据集pattern不同。

## 局限与展望

- VLM 描述质量直接影响偏差分析准确性，目前只用了 LLaVA-1.6-Mistral-7B，更强的 VLM 可能发现更细粒度的偏差
- 目前只做了物体去偏，未做时序或活动去偏——可以进一步构建针对不同偏差维度的去偏split
- 分析是offline的静态方式，尚未探索如何在训练阶段利用偏差信息（如对偏差样本做 re-weighting）
- 文本描述可能丢失视觉细节（如物体纹理、运动速度），导致偏差被低估

## 相关工作与启发

- **vs HAT/Diving48**：它们通过修改视频（分割人物）或设计特殊数据集来去偏，UTD 的优势是不修改任何视频、适用于任何已有基准
- **vs TempAct/TemporalBench**：这些工作构建新基准专门评时序理解，UTD 是分析已有基准的时序偏差程度，两者互补
- 这篇论文提供了一套系统化的视频基准分析工具，可作为评估新数据集质量的标准pipeline

## 评分

- 新颖性: ⭐⭐⭐⭐ 文本描述去偏的idea新颖但不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 12 数据集+30 模型+多维度分析极其详尽
- 写作质量: ⭐⭐⭐⭐ 框架化定义清晰，但公式较多读起来偏重
- 价值: ⭐⭐⭐⭐ 提供了有用的去偏split和分析工具, 对社区有实际贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](../../NeurIPS2025/video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)

</div>

<!-- RELATED:END -->
