---
title: >-
  [论文解读] Online Temporal Action Localization with Memory-Augmented Transformer
description: >-
  [ECCV 2024][在线时序动作定位] 本文提出 MATR（Memory-Augmented Transformer），通过记忆队列选择性地保存历史片段特征来建模长期上下文，并采用双 Transformer 解码器分别定位动作的结束和起始时间，在 THUMOS14 和 MUSES 两个在线时序动作定位基准上刷新了 SOTA，甚至可与部分离线方法媲美。
tags:
  - ECCV 2024
  - 在线时序动作定位
  - Transformer
  - 长期上下文建模
  - 端到端检测
  - 滑动窗口
---

# Online Temporal Action Localization with Memory-Augmented Transformer

**会议**: ECCV 2024  
**arXiv**: [2408.02957](https://arxiv.org/abs/2408.02957)  
**代码**: https://cvlab.postech.ac.kr/research/MATR/ (有)  
**领域**: Others (视频理解)  
**关键词**: 在线时序动作定位, 记忆增强Transformer, 长期上下文建模, 端到端检测, 滑动窗口

## 一句话总结
本文提出 MATR（Memory-Augmented Transformer），通过记忆队列选择性地保存历史片段特征来建模长期上下文，并采用双 Transformer 解码器分别定位动作的结束和起始时间，在 THUMOS14 和 MUSES 两个在线时序动作定位基准上刷新了 SOTA，甚至可与部分离线方法媲美。

## 研究背景与动机
时序动作定位（TAL）旨在未修剪视频中检测每个动作实例的起止时间和类别。在线 TAL（On-TAL）要求仅使用当前时刻及之前的帧进行推理，且已输出的预测不可修改，应用场景包括视频监控、体育分析和视频摘要。早期 On-TAL 方法基于在线动作检测（OAD），先对每帧分类再聚合为实例，但这种帧级监督不够优化。后续方法 OAT 引入滑动窗口和锚框机制利用实例级监督，但仍有两个关键缺陷：（1）每次迭代仅处理固定大小的视频片段，无法建模超出窗口的长期动作；（2）性能对输入片段大小高度敏感，每个数据集都需要仔细调参。核心痛点在于如何在线设定下有效利用长期上下文来定位跨越多个片段的动作实例。本文的核心 idea：用 FIFO 记忆队列选择性地存储过去的片段特征，并通过双解码器——先检测动作结束再回溯记忆找起始——实现精确定位。

## 方法详解

### 整体框架
MATR 包含四部分：（1）特征提取器：使用预训练骨干网络（TSN/I3D）提取当前输入片段的帧级特征；（2）记忆增强视频编码器：用 Transformer 编码器编码片段内时序上下文，并通过 flag token 机制选择性地将片段特征存入记忆队列；（3）实例解码模块：由 end decoder 和 start decoder 两个 Transformer 解码器组成，分别利用当前片段特征定位动作结束、利用记忆队列定位动作起始；（4）预测头：分别预测结束偏移、起始区域+偏移、和动作类别。模型以滑动窗口方式逐帧推理，端到端训练。

### 关键设计
1. **记忆队列与 Flag Token 机制**:
    - 功能：选择性存储过去的片段特征，为模型提供长期上下文
    - 核心思路：记忆队列以 FIFO 方式管理，队列满时淘汰最旧的。关键创新是引入一个可学习的 flag token，与片段特征一起送入编码器后由 flag 预测头判断当前片段是否与动作实例相关。训练时用 ground truth 的 FLAG 标签，推理时用 $\text{sigmoid}(\hat{g}) > \theta$ 判断。仅当 FLAG=1 时才将片段存入记忆，有效过滤背景帧，提高记忆利用效率
    - 设计动机：与 OAD 中的记忆模块不同，TAL 需要保留时序位置信息以精确预测时间，直接压缩记忆会丢失关键的时序位置。选择性存储比全部存储更高效，避免背景帧的干扰

2. **End-Start 双解码器定位**:
    - 功能：分别定位动作的结束和起始时间
    - 核心思路：End Decoder 利用编码后的当前片段特征，通过交叉注意力定位当前时间附近的动作结束点。Start Decoder 接收 End Decoder 的输出嵌入，并利用记忆队列拼接当前片段特征作为长期上下文，通过交叉注意力回溯找到动作起始点。两个解码器共享架构但使用不同的信息源。采用 2D 时序位置编码（相对片段位置 + 相对帧位置）来支持不可预测长度的流式视频
    - 设计动机：动作的结束通常在当前片段附近（可以从短期特征判断），而起始可能在很久之前（需要长期记忆），两者的信息需求不同所以分别建模更合理。实验证明双解码器比单解码器提升 6.8 mAP

3. **Class-Boundary 分离查询**:
    - 功能：解耦动作分类与边界定位两个子任务
    - 核心思路：为每个实例设置一对查询：class query $Q_\text{class}$ 负责动作分类，boundary query $Q_\text{bound}$ 负责边界定位。两者共享相同的位置编码 $E_\text{pos}$ 以关联同一实例。分类头将 End Decoder 和 Start Decoder 的 class 嵌入拼接后输出类别概率。起始预测头采用区域分类+偏移回归的分层策略，将时间范围划分为 $L_m + 2$ 个区域
    - 设计动机：受目标检测（DETR 系列）中分离分类和定位的经验启发，让不同的查询专注不同子任务可以减少任务间的干扰

### 损失函数 / 训练策略
端到端训练，使用匈牙利算法匹配预测与 GT。总损失 $L = L_\text{class} + L_\text{start} + L_\text{end} + L_\text{diou} + L_\text{flag}$，其中分类用 Focal Loss，起始区域用交叉熵，起始和结束偏移用 L1 损失，实例级监督用 DIoU Loss，flag token 用 BCE Loss。所有损失系数均为 1，无需额外平衡。推理时在每个时间步应用 NMS，并去除预测结束时间超过当前时间的实例。

## 实验关键数据

### 主实验
THUMOS14 和 MUSES 数据集上的 mAP(%) 对比：

| 方法 | 类型 | THUMOS14 Avg mAP | MUSES Avg mAP |
|------|------|-----------------|--------------|
| SimOn | OAD-based Online | 34.4 | - |
| CAG-QIL | OAD-based Online | 29.7 | 4.8 |
| OAT-OSN | Instance Online | 44.6 | 13.7 |
| **MATR** | **Instance Online** | **49.5** | **14.4** |
| G-TAD | Offline | 39.9 | 11.4 |
| MUSES | Offline | 53.4 | 18.6 |
| ActionFormer | Offline | 66.8 | - |

MATR 在 THUMOS14 上超越前 SOTA OAT-OSN 4.9 个点，在 MUSES 上超越 0.7 个点，甚至超过部分离线方法（G-TAD、P-GCN）。

### 消融实验
| 配置 | Avg mAP | 说明 |
|------|---------|------|
| Full model (MATR) | 49.5 | 完整模型 |
| w/o flag token | 47.4 | 去掉选择性存储，-2.1 |
| w/o segment encoder | 46.6 | 去掉片段编码器，-2.9 |
| Single decoder | 42.7 | 单解码器同时预测起止，-6.8 |
| w/o splitting query | 47.9 | 不分离 class/boundary 查询，-1.6 |
| w/o sampling | 47.2 | 不对记忆做采样，-2.3 |
| w/o DIoU loss | 41.4 | 去掉 DIoU 损失，-8.1 |
| w/o memory | 46.0 | 完全不用记忆队列，-3.5 |
| memory size=7 (best) | 49.5 | THUMOS14 最佳记忆大小 |
| memory size=15 (best) | 14.4 | MUSES 最佳记忆大小 |

### 关键发现
- 双解码器设计是最关键的组件，单解码器性能骤降 6.8 mAP
- DIoU 实例级监督对在线 TAL 至关重要（-8.1 mAP），说明帧级监督不足以学好边界定位
- 记忆队列的大小只要覆盖训练集中 99% 实例的时长即可（THUMOS14 约 7 个片段），更大并不一定更好
- MATR 对片段大小不敏感：从 64 降到 8，性能仅下降 9.1%，而 OAT-OSN 从 44.6 降到 25.8
- 与 OAD 记忆模块对比，MATR 仅需 24M 参数和 167ms 推理时间，而 MAT 需 40M/192ms、E2E-LOAD 需 53M/196ms
- 起始预测使用区域分类+偏移回归（49.5）优于纯偏移回归（46.7）

## 亮点与洞察
- "先找结束再回溯记忆找起始"的检测范式非常直觉化且合理——人类判断动作结束是即时的，而回忆起始需要访问长期记忆
- Flag token 的选择性存储既简单又有效，是一种优雅的记忆管理方案，避免了复杂的记忆压缩和注意力选择
- 分离 class query 和 boundary query 的设计成功借鉴了目标检测领域的经验
- 2D 时序位置编码（段位置+帧位置）巧妙解决了流式视频时长不确定的位置编码问题
- 端到端训练且所有损失系数均为 1，简洁不需调参

## 局限与展望
- 当记忆队列中存在多个动作实例时，可能出现起始点匹配错误
- 存储片段时仅考虑是否与动作相关，未利用已存储记忆中的上下文关系
- 在 MUSES 这种多镜头切换的数据集上性能提升有限，可能需要更强的跨镜头建模能力
- 推理时仍需 NMS 后处理，不是完全端到端的
- 模型参数量（192.8M）显著大于 OAT-OSN（128.7M），主要因为双解码器设计

## 相关工作与启发
- **DETR/ActionFormer**: Transformer 端到端检测和时序定位的范式，本文在在线场景中引入类似思路
- **OAT**: 首个使用实例级监督的 On-TAL 方法，但受限于固定窗口无法建模长期动作
- **Stream Buffer / MAT**: OAD 中的记忆模块，在 TAL 场景下因压缩丢失时序位置信息而不适用
- 启发：在线视频理解中记忆管理是核心问题，选择性存储（何时存/何时丢弃）比简单压缩更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 端到端在线 TAL + 记忆队列 + 双解码器的组合有新意，但各组件并不全新
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、详尽的消融（模块/记忆大小/片段大小/预测头/记忆压缩/推理时间）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，方法阐述到位
- 价值: ⭐⭐⭐⭐ 对在线 TAL 任务有实际推动，填补了长期上下文建模的空缺

<!-- RELATED:START -->

## 相关论文

- [Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)
- [Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing](wavelength-embedding-guided_filter-array_transformer_for_spectral_demosaicing.md)
- [Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/others/learning-augmented_online_bipartite_fractional_matching.md)
- [Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)

<!-- RELATED:END -->
