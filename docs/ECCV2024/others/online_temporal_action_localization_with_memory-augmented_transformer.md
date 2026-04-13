---
title: >-
  [论文解读] Online Temporal Action Localization with Memory-Augmented Transformer
description: >-
  [ECCV 2024][在线时序动作定位] 提出 MATR（Memory-Augmented Transformer），通过记忆队列存储过去片段的特征来利用长期上下文，并采用分离的 Start/End Transformer 解码器进行动作实例定位，在在线时序动作定位（On-TAL）任务上取得 SOTA，甚至可比肩部分离线方法。
tags:
  - ECCV 2024
  - 在线时序动作定位
  - 记忆增强
  - Transformer
  - 视频理解
  - 动作检测
---

# Online Temporal Action Localization with Memory-Augmented Transformer

**会议**: ECCV 2024  
**arXiv**: [2408.02957](https://arxiv.org/abs/2408.02957)  
**代码**: [https://cvlab.postech.ac.kr/research/MATR/](https://cvlab.postech.ac.kr/research/MATR/)  
**领域**: 其他  
**关键词**: 在线时序动作定位, 记忆增强, Transformer, 视频理解, 动作检测

## 一句话总结

提出 MATR（Memory-Augmented Transformer），通过记忆队列存储过去片段的特征来利用长期上下文，并采用分离的 Start/End Transformer 解码器进行动作实例定位，在在线时序动作定位（On-TAL）任务上取得 SOTA，甚至可比肩部分离线方法。

## 研究背景与动机

在线时序动作定位（On-TAL）的目标是在流式视频中实时检测动作实例（起止时间+类别），不能使用未来帧且不能修改已有预测。现有方法存在以下局限：

1. **OAD-based 方法**（TeSTra、SimOn 等）：基于逐帧动作检测再聚合，仅使用帧级监督，无法充分利用实例级信息
2. **OAT（Online Anchor Transformer）**：使用固定大小的输入片段，**无法考虑长期上下文**，对超出片段大小的长动作检测能力有限
3. **片段大小超参数敏感**：OAT 在片段大小从64降到8时，mAP 从 44.6 骤降到 25.8

MATR 的核心思路：使用**记忆队列**选择性存储过去片段特征，突破固定输入窗口的限制；使用**分离的 Start/End 解码器**分别定位动作的起止时间。

## 方法详解

### 整体框架

MATR 由四部分组成：
1. **特征提取器**：冻结的预训练 backbone（TSN/I3D）+ 线性投影
2. **记忆增强视频编码器**：片段编码器 + 记忆更新模块
3. **实例解码模块**：End Decoder + Start Decoder
4. **预测头**：End/Start/Classification 三个预测头

采用滑动窗口方案，以片段为单位逐帧滑动处理流式视频。

### 关键设计

1. **记忆队列与 Flag Token 选择性存储**：采用 FIFO 记忆队列存储过去片段特征。关键创新是 **Flag Token** 机制——在片段编码器中拼接一个可学习的 flag token，训练其预测当前片段是否与动作实例相关（$[\text{FLAG}]=1$ 当片段与动作实例有时间重叠）。只有 flag 预测为正的片段才存入记忆，避免存储无关背景信息。推理时使用阈值 $\theta=0.5$ 判断。

2. **分离的 End/Start Decoder**：两个 Transformer 解码器共享架构但使用不同的信息源：
   - **End Decoder**：从当前编码片段特征中定位动作结束时间（短期信息）
   - **Start Decoder**：从记忆队列+当前片段的拼接中定位动作起始时间（长期信息）

   这种分离设计使得 start 和 end 的预测各自关注最相关的时间范围。消融实验显示单解码器方案导致 mAP 从 49.5 大幅下降到 42.7。

3. **Class Query 与 Boundary Query 的分离**：每个动作实例使用 $2N$ 个查询向量，其中 $N$ 个用于分类（class query），$N$ 个用于边界定位（boundary query）。同一实例的 class/boundary query 共享位置编码 $E_{pos}$ 以保持实例对应关系。动作分类使用两个解码器的 class embeddings 拼接输入分类头。

4. **Start 预测的区域分类+偏移回归**：将时间轴划分为 $L_m+2$ 个区域（记忆覆盖的 $L_m$ 段 + 记忆前 + 当前片段），先分类 start 所在区域，再在区域内做偏移回归，缩小回归范围：

$$\hat{s}_i = t - (\hat{o}_i + \hat{v}_i(\hat{o}_i)) \times L_s$$
$$\hat{e}_i = t + \hat{u}_i \times L_s$$

5. **2D 时间位置编码**：为适应不可预知长度的流式视频，将时间位置编码分为相对片段位置和相对帧位置两个维度，扩展位置编码的覆盖范围。

6. **记忆特征 50% 均匀采样**：相邻帧信息相似，对记忆特征进行50%采样以提高效率。

### 损失函数 / 训练策略

总损失由五个部分组成，所有系数均为1，无需损失平衡：

$$L = L_{class} + L_{start} + L_{end} + L_{diou} + L_{flag}$$

- $L_{class}$：Focal Loss 用于动作分类
- $L_{start}$：Cross Entropy（区域分类）+ $\ell_1$（偏移回归）
- $L_{end}$：$\ell_1$ 偏移回归损失
- $L_{diou}$：DIoU Loss 提供实例级监督，连接 start 和 end 预测
- $L_{flag}$：BCE Loss 训练 flag 预测头

使用 Hungarian 算法进行预测-GT 匹配。训练时，模型检测结束时间在 $[t-T_d+1, t+T_a]$ 范围内的动作实例。

## 实验关键数据

### 主实验

| 方法 | 类型 | THUMOS14 Avg mAP | MUSES Avg mAP |
|------|------|-----------------|---------------|
| **MATR** | **Online** | **49.5** | **14.4** |
| OAT-OSN | Online | 44.6 | 13.7 |
| SimOn* | Online (OAD) | 36.9 | - |
| CAG-QIL | Online (OAD) | 29.7 | 4.8 |
| ActionFormer | Offline | 66.8 | - |
| TriDet | Offline | 69.3 | - |
| MUSES | Offline | 53.4 | 18.6 |
| P-GCN | Offline | - | 13.0 |

MATR 在 THUMOS14 上超越之前最佳在线方法 OAT-OSN **4.9%p**，在 MUSES 上超越 **0.7%p**，甚至超过了 G-TAD 和 P-GCN 等离线方法。

### 消融实验

| 配置 | THUMOS14 Avg mAP | 说明 |
|------|-----------------|------|
| Full MATR | **49.5** | 完整模型 |
| w/o flag token | 47.4 | -2.1，选择性存储重要 |
| w/o segment encoder | 46.6 | -2.9，片段内时序上下文重要 |
| single decoder | 42.7 | **-6.8**，分离解码器至关重要 |
| w/o splitting query | 47.9 | -1.6，分离 class/boundary query 有效 |
| w/o memory | 46.0 | -3.5，记忆队列贡献显著 |
| w/o DIoU loss | 41.4 | **-8.1**，实例级监督最重要 |

### 记忆大小分析

| 记忆大小 | THUMOS14 mAP | MUSES mAP |
|---------|-------------|-----------|
| 0 (w/o) | 46.0 | 13.3 |
| 1 | 47.7 | 13.6 |
| 3 | 48.3 | 13.9 |
| **7** | **49.5** | 13.8 |
| **15** | 48.0 | **14.4** |
| 19 | 49.1 | 14.1 |

### 关键发现

- **分离 Start/End 解码器是最关键设计**：消融显示 6.8%p 的性能下降
- **DIoU 实例级监督不可或缺**：移除后 mAP 下降 8.1%p
- **对片段大小不敏感**：片段从 64 降到 8 时，MATR 仅下降 9.1%p，而 OAT-OSN 下降 18.8%p
- **推理效率优于 OAD 记忆方法**：167.1ms/iter（6.0fps），参数量 24.0M（vs MAT 的 40.1M），因为记忆模块仅用片段编码器+flag头，设计更轻量
- 减小片段到16时可达 18.6fps 同时 mAP 仍优于 OAT-OSN

## 亮点与洞察

1. **"先检测 End，再从记忆中找 Start" 的定位策略**非常符合在线场景的直觉——动作结束可从当前片段判断，起始则需回溯历史
2. **Flag Token 的选择性记忆机制**巧妙且高效，既避免存储无关背景，又几乎不增加计算量
3. **Class/Boundary Query 分离**借鉴了目标检测的解耦策略（类似 DINO/DAB-DETR），在时序任务中同样有效
4. **区域分类+偏移回归的两阶段 start 预测**有效缩小了长时间跨度下的回归范围

## 局限性 / 可改进方向

1. **记忆中多实例混淆**：当记忆队列包含多个动作实例时，可能将检测到的 end 与错误的 start 匹配
2. **记忆存储未利用历史上下文**：flag token 仅考虑当前片段是否与动作相关，未利用记忆中已有的信息来指导新片段的存储决策
3. 与离线 SOTA（ActionFormer 66.8 vs MATR 49.5）仍有较大差距
4. MUSES 数据集上性能提升有限（仅 0.7%p），多镜头动作仍是挑战
5. 使用冻结的 backbone（TSN/I3D），端到端微调 backbone 可能进一步提升

## 相关工作与启发

- **OAT**：最直接的前序工作，MATR 在其滑动窗口方案基础上增加了记忆队列
- **ActionFormer / TriDet**：离线 TAL 的 SOTA，MATR 的 DIoU Loss 借鉴了 ActionFormer
- **DETR 系列**：Query-based 检测范式的灵感来源，分离 class/boundary query 借鉴了 DINO
- **Stream Buffer / MAT**：OAD 中的记忆模块，但无法保留时间位置信息不适合 TAL
- 启发：记忆增强的在线推理范式可推广到在线目标检测、在线分割等流式视频任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Start/End 分离解码+ Flag Token 记忆机制组合新颖，但各模块单独看并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融极其全面（模块消融、记忆大小、片段大小、预测头设计、压缩因子、推理时间），双数据集验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示直观，方法描述详细
- **价值**: ⭐⭐⭐⭐ — 在线 TAL 是重要的实际应用方向，方法通用性强
