---
title: >-
  [论文解读] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos
description: >-
  [CVPR 2026][目标检测][长时间referring] 利用固定视角视频中背景结构的时不变性，构建离线 Anchor Bank + 在线 Anchor Map 作为语言-场景持久记忆，配合锚点引导的重入先验和 ReID-Gating 身份验证机制，实现目标遮挡/离场后的鲁棒重捕获，RCR 提升 10.3%、RCL 降低 24.2%。
tags:
  - CVPR 2026
  - 目标检测
  - 长时间referring
  - 固定视角视频
  - 背景锚点
  - 重入检测
  - 身份重识别
---

# AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos

**会议**: CVPR 2026  
**arXiv**: [2603.07758](https://arxiv.org/abs/2603.07758)  
**代码**: 待确认  
**领域**: 目标检测 / 视频理解 / 语言引导目标定位  
**关键词**: 长时间referring, 固定视角视频, 背景锚点, 重入检测, 身份重识别

## 一句话总结

利用固定视角视频中背景结构的时不变性，构建离线 Anchor Bank + 在线 Anchor Map 作为语言-场景持久记忆，配合锚点引导的重入先验和 ReID-Gating 身份验证机制，实现目标遮挡/离场后的鲁棒重捕获，RCR 提升 10.3%、RCL 降低 24.2%。

## 研究背景与动机

**领域现状**：语言引导的视频目标定位（referring）已成为监控、行为分析等场景的核心技术。现有方法（MTTR、ReferFormer、OnlineRefer 等）主要面向短时序场景，假设目标在大部分帧中可见，通过帧间外观传播维持身份一致性。

**现有痛点**：在长时间固定视角视频中（如监控摄像头，平均 >120s），目标会被遮挡、离开视野再重新进入。现有方法面临三大问题：
   - **语义记忆丢失**：目标不可见时，framewise pipeline 的语义记忆中断，无法在目标重入时恢复关联
   - **外观漂移**：长时间跨度下光照变化、姿态变换导致外观特征不可靠，基于 ReID 的纯外观匹配容易漂移
   - **近语义干扰**：相似外观的干扰目标（如穿相似衣服的行人）在目标缺失期间被误识别

**核心矛盾**：现有方法的语义对齐完全依赖目标本身的外观特征，一旦目标不可见，"文本-目标"的语义链条断裂。但固定视角视频的背景结构是稳定的——这个信息完全被忽略了。

**本文切入角度**：固定摄像头 → 背景布局不变 → 可以从背景中蒸馏出一组空间锚点 → 将文本 query 与锚点对齐 → 即使目标消失，"文本-场景"的空间记忆仍然持续有效 → 目标重入时利用空间先验快速重捕获。

**核心 idea**：**用背景结构的时不变性弥补目标外观的时变性**——将 referring 从"找目标"升级为"在场景坐标系中定位 query 对应的空间区域"。

## 方法详解

### 整体框架

输入为固定视角视频帧序列 $\{I_t\}_{t=1}^{T}$ 和自然语言 query $q$，输出逐帧 bounding box $\{y_t\}$。Pipeline 分两阶段：

- **离线阶段**：从前 $T_0$ 帧提取静态背景结构，蒸馏为 Anchor Bank
- **在线阶段**：query 与 Anchor Bank 对齐生成 Anchor Map → 锚点引导的候选过滤 + 融合评分 → 搜索模式下维护重入先验 → ReID-Gating 验证身份

关键设计是**不假设目标在首帧可见**，系统需要从头开始"找到"目标。

### 关键设计

#### 1. Anchor Bank（离线背景结构蒸馏）

- **功能**：从固定视角视频的前 $T_0$ 帧中提取 $K$ 个静态背景区域锚点 $\mathcal{B} = \{(M_k, p_k, c_k)\}_{k=1}^{K}$
- **核心思路**：选择中位亮度帧 $t^\star$，用分割模型（SAM）提取持久区域 mask $M_k$，然后在视觉编码器特征图上做 mask-aware 均值池化得到原型向量：
  $$p_k = \text{Norm}\left(\frac{1}{|M_k|}\sum_x M_k(x) F_{t^\star}(x)\right)$$
  质心为 $c_k = \frac{1}{|M_k|}\sum_x M_k(x) \cdot x$
- **设计动机**：固定视角下背景结构不变，锚点一次提取终身可用。这些锚点提供了一个**场景坐标系**——后续所有操作都在这个坐标系中进行，天然具备空间不变性。默认 $K=64$，$T_0 \in [30, 120]$。

#### 2. Anchor Map（在线语言-场景对齐）

- **功能**：将文本 query 映射到场景空间，生成一个 query-conditioned 的空间热力图
- **核心思路**：通过轻量对齐头 $\phi_l, \phi_v$ 将文本嵌入 $e_q$ 和锚点原型 $p_k$ 映射到共同子空间，计算余弦相似度和 softmax 权重：
  $$s_k = \cos(\phi_l(e_q), \phi_v(p_k)), \quad \omega_k = \frac{\exp(\tau \cdot s_k)}{\sum_j \exp(\tau \cdot s_j)}$$
  加权融合各锚点 mask 得到 Anchor Map：
  $$A(x) = \sum_{k=1}^{K} \omega_k M_k(x) \in [0,1]$$
- **设计动机**：Anchor Map 对给定 query 是**固定的**——$\{M_k\}$ 和 $\{\omega_k\}$ 在推理期间不变。即使目标消失数百帧，系统仍然"记得"query 描述的目标最可能出现在场景的哪个区域。**这是全文最核心的设计**：将短暂的"目标外观记忆"替换为持久的"场景空间记忆"。

#### 3. 锚点引导的候选过滤与融合评分

- **空间过滤**：开放词汇检测器 $\mathcal{D}$（GroundingDINO）生成候选区域 $\mathcal{R}_t$，只保留 Anchor Map 响应超过阈值 $\eta$ 的候选：
  $$\tilde{\mathcal{R}}_t = \{r \in \mathcal{R}_t \mid \bar{A}_{bb}(r) \geq \eta\}$$
- **融合评分**：对过滤后的候选做 mask-aware 特征池化 $g_v(r)$，融合文本-视觉相似度和锚点证据：
  $$\text{Score}(r) = \lambda \cos(g_v(r), g_l(q)) + (1-\lambda) \bar{A}_m(r)$$
  当最高分低于阈值 $\theta$ 时系统进入搜索模式（目标不可见），否则进入 ReID-Gating 验证。

#### 4. Anchor-based Re-entry Prior（重入先验）

- **功能**：在目标不可见期间维护一个空间概率分布，预测目标最可能从哪里重新出现
- **核心思路**：重入先验 $P_t^{re}$ 初始化为 Anchor Map $A$，通过 EMA + 高斯平滑 + $\ell_1$ 归一化迭代更新：
  $$\tilde{P}_t^{re} = \beta(G_\sigma * \tilde{P}_{t-1}^{re}) + (1-\beta) A$$
  候选框获得乘性权重 $W(r) \propto A(x) \cdot P_t^{re}(x)$，使评分偏向高概率重入区域。当目标被确认出现在锚点 $k^\star$ 时，先验重定向：
  $$\tilde{P}_{t+1}^{re} = \rho \cdot G_\sigma(\cdot - c_{k^\star}) + (1-\rho) A$$
- **设计动机**：目标重入不是随机的——在固定视角下，行人倾向于从入口/通道等特定区域重新出现。重入先验编码了这个"空间习惯"，加速重捕获。

#### 5. ReID-Gating（身份验证门控）

- **功能**：验证候选目标的身份是否与之前追踪的目标一致，防止身份漂移
- **核心思路**：综合三路信号做门控决策——外观 ReID 相似度、锚点一致性、锚点坐标系中的位移：
  $$G(r) = \sigma(\alpha_1 \cdot \text{sim}_{\text{ReID}}(r) + \alpha_2 \cdot \bar{A}_m(r) - \alpha_3 \cdot \hat{\Delta}(r) + b)$$
  其中 $\text{sim}_{\text{ReID}}(r)$ 通过动量队列 $\mathcal{Q}$ 稳定外观嵌入，$\hat{\Delta}(r)$ 是候选与上次确认锚点的归一化位移。$G(r) \geq \gamma$ 则接受候选。
- **设计动机**：纯外观 ReID 在长时间跨度下不可靠（光照/姿态变化），加入锚点证据和位移约束后相当于在"场景坐标系"中做身份验证。位移惩罚防止远处的相似外观目标被误匹配。

### 实现细节

系统完全 zero-shot 运行，使用冻结编码器：提案用 GroundingDINO，跨模态消歧用 RexSeek 风格的 refiner，mask 用 SAM，身份嵌入用 CLIP 族编码器，query 预处理用 spaCy。关键超参：$K=64$, $\tau=10$, $\lambda=0.6$, $\theta=0.4$, $\beta=0.8$, $\gamma=0.5$。

## 实验关键数据

### AR²-4FV-Bench（新基准）

首个面向固定视角长时间 referring + ReID 的专用基准：

| 维度 | 规模 |
|------|------|
| 视频数量 | 1,684 |
| 平均时长 | >120 秒 |
| 场景类型 | 校门/大堂/社区路口/室内走廊等 |
| 标注内容 | 逐帧可见性 + bbox + 重入时间戳 |
| 难度分层 | 消失时长(短/中/长) × 重入次数(单/多) |
| query 类型 | 锚点参照型 + 属性消歧型 + 同义改写 |

### 主实验（重入性能）

| 方法 | 会议 | IDF1↑ | RCR↑ | RCL↓ |
|------|------|-------|------|------|
| MTTR | CVPR'22 | 56.3 | 0.60 | 33.8 |
| ReferFormer | CVPR'22 | 57.9 | 0.63 | 31.2 |
| OnlineRefer | ICCV'23 | 58.6 | 0.64 | 29.9 |
| SOC | NeurIPS'23 | 58.7 | 0.64 | 30.3 |
| DsHmp | CVPR'24 | 60.4 | 0.66 | 28.6 |
| SSA | CVPR'25 | 61.5 | 0.68 | 26.5 |
| DUTrack | CVPR'25 | 62.3 | 0.69 | 25.8 |
| **AR²-4FV** | **-** | **64.8** | **0.75** | **20.1** |

AR²-4FV vs 最优 baseline（DUTrack）：RCR +8.7%（0.69→0.75），RCL -22.1%（25.8→20.1 帧）。

### 定位性能

| 方法 | mAP↑ | mIoU↑ |
|------|------|-------|
| OnlineRefer | 46.1 | 64.2 |
| DUTrack | 46.5 | 63.7 |
| SSA | 45.2 | 64.0 |
| **AR²-4FV** | **49.2** | **66.9** |

mAP +6.7%，mIoU +4.2%，在高 IoU 阈值（P@0.8, P@0.9）下优势更明显。

### 消融实验

| Anchor Bank | Anchor Map | Re-entry Prior | ReID-Gating | mIoU | mAP | IDF1 | RCR | RCL |
|:-:|:-:|:-:|:-:|------|------|------|-----|-----|
| ✓ | — | — | — | 63.2 | 45.2 | 61.2 | 0.67 | 27.1 |
| ✓ | ✓ | ✓ | — | 64.7 | 46.3 | 62.2 | 0.70 | 26.9 |
| ✓ | ✓ | — | ✓ | 63.8 | 45.5 | 61.3 | 0.68 | 21.3 |
| ✓ | ✓ | ✓ | ✓ | **66.9** | **49.2** | **64.8** | **0.75** | **20.1** |

### 关键发现

- **Anchor Map 是基础**：提供空间记忆，是后续模块的前提
- **Re-entry Prior 主攻 RCR**：加入后 RCR 从 0.67 提升到 0.70，帮助更快找到重入目标
- **ReID-Gating 主攻 RCL**：加入后 RCL 从 27.1 大幅降到 21.3，减少误判带来的延迟
- **三模块组合互补性强**：全配置 mIoU 66.9 >> 单 Anchor Bank 63.2，各模块解决不同维度的问题

## 亮点与洞察

- **背景作为"空间身份证"**的思路非常巧妙：固定视角下，"门口的人"比"穿红衣服的人"更可靠——前者时不变，后者随光照/姿态变化。这将 referring 从外观空间转移到场景空间
- **Zero-shot 运行**：所有编码器冻结，Anchor Bank 只需一次性提取，推理无需训练。这使得部署到真实监控场景的门槛极低
- **重入先验 $P^{re}$ 的动态更新机制**可迁移到其他需要预测"再次出现位置"的任务（如机器人导航中的障碍物重现预测）
- **ReID-Gating 的三路信号融合**（外观 + 空间 + 位移）比纯外观 ReID 更鲁棒，思路可用于行人重识别

## 局限与展望

1. **强依赖固定视角假设**：摄像头有轻微抖动或 PTZ 时，Anchor Bank 失效。可考虑引入背景配准模块适应准固定视角
2. **锚点数量 $K=64$ 固定**：不同场景复杂度差异大，自适应锚点数量选择可能更优
3. **线性时间复杂度但常数较大**：每帧需要 GroundingDINO + SAM + CLIP 多次前向，实时性存疑
4. **不处理跨场景外观变化**：作者明确排除了换衣等跨场景 ReID 场景，适用范围有限
5. **重入先验假设空间规律性**：对于行为高度随机的场景（如动物行为分析），基于背景结构的重入假设可能不成立

## 相关工作与启发

- **vs ReferFormer/MTTR**：这些方法假设目标持续可见，通过帧间 Transformer 传播语义；AR²-4FV 不做此假设，用场景结构代替帧间传播
- **vs OVTrack**：OVTrack 用文本检索做开放词汇跟踪，但仍依赖外观连续性；AR²-4FV 在外观不可靠时用空间先验接管
- **vs ByteTrack/BoT-SORT**：这些 MOT 方法面向短时场景，没有语言引导和长时间重入处理
- **vs 背景建模方法（MOG/ViBe）**：AR²-4FV 借鉴了传统背景建模的"前景-背景分离"思想，但将其升级为语义级的"锚点-query 对齐"

## 评分

- 新颖性: ⭐⭐⭐⭐ 用背景结构做语言引导 referring 的思路新颖，但各子模块是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ 新 benchmark + 完整消融，但缺少跨数据集泛化和推理速度分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法伪代码和公式完整，但部分表格排版混乱
- 价值: ⭐⭐⭐⭐ 固定视角监控场景有实际应用价值，zero-shot 设计降低部署门槛

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)
- [\[CVPR 2026\] GeoBridge: A Semantic-Anchored Multi-View Foundation Model for Geo-Localization](geobridge_semantic-anchored_multi-view_foundation_model_for_geo-localization.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/object_detection/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[CVPR 2026\] Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)
- [\[AAAI 2026\] Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](../../AAAI2026/object_detection/robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)

<!-- RELATED:END -->
