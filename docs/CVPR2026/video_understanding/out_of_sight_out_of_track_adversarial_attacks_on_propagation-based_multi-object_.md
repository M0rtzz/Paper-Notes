---
title: >-
  [论文解读] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation
description: >-
  [CVPR 2026][视频理解][多目标跟踪] 首次系统分析 Tracking-by-Query-Propagation（TBP）跟踪器的对抗脆弱性，提出 FADE 攻击框架，通过时序查询洪泛（TQF）耗尽固定查询预算和时序记忆腐蚀（TMC）破坏隐状态传播两种策略，在 MOT17/MOT20 上对 MOTR/MOTRv2/MeMOTR/Samba/CO-MOT 造成最高约 30 点 HOTA 下降和 10 倍以上身份切换。
tags:
  - CVPR 2026
  - 视频理解
  - 多目标跟踪
  - 对抗攻击
  - 查询传播
  - 时序记忆腐蚀
  - 物理攻击
---

# Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation

**会议**: CVPR 2026  
**arXiv**: [2604.00452](https://arxiv.org/abs/2604.00452)  
**代码**: 无  
**领域**: 视频理解 / 多目标跟踪 / 对抗攻击  
**关键词**: 多目标跟踪, 对抗攻击, 查询传播, 时序记忆腐蚀, 物理攻击

## 一句话总结

首次系统分析 Tracking-by-Query-Propagation（TBP）跟踪器的对抗脆弱性，提出 FADE 攻击框架，通过时序查询洪泛（TQF）耗尽固定查询预算和时序记忆腐蚀（TMC）破坏隐状态传播两种策略，在 MOT17/MOT20 上对 MOTR/MOTRv2/MeMOTR/Samba/CO-MOT 造成最高约 30 点 HOTA 下降和 10 倍以上身份切换。

## 研究背景与动机

1. **领域现状**：多目标跟踪（MOT）从传统的 Tracking-by-Detection（TBD）范式发展到更先进的 Tracking-by-Propagation（TBP）范式。TBP 跟踪器（如 MOTR、MOTRv2、MeMOTR、Samba、CO-MOT）通过自回归传播 track query 实现端到端检测与关联，避免了 Kalman 滤波等启发式关联。

2. **现有痛点**：现有的 MOT 对抗攻击（Daedalus、Hijacking、F&F、BankTweak）均针对 TBD 架构设计——攻击 NMS 阈值、Kalman 滤波预测、或独立的外观特征库——这些组件在端到端 TBP 跟踪器中不存在。

3. **核心矛盾**：TBP 跟踪器引入了全新的结构性脆弱点：(1) **固定查询预算**形成零和博弈——分配给虚假轨迹的 query 必然减少合法轨迹的容量；(2) **循环隐状态传播**创建时序依赖链——状态腐蚀会跨帧传播；(3) **内置时序记忆**放大攻击持久性。

4. **本文目标** 针对 TBP 特有的查询预算和时序记忆机制设计专用攻击方法。

5. **切入角度**：从 TBP 的核心机制（query 预算分配 + query 更新器的记忆传播）出发，设计两种互补攻击：一种"渗透"（制造假轨迹占预算），一种"遗忘"（破坏真轨迹的记忆）。

6. **核心 idea**：通过洪泛虚假查询耗尽预算或腐蚀时序记忆，从 TBP 的架构内部瓦解跟踪。

## 方法详解

### 整体框架

FADE 是一个统一的对抗攻击流水线：输入帧通过数字扰动或物理传感器欺骗（声学/电磁）生成对抗样本，送入 TBP 跟踪器，利用跟踪器的输出预测和隐状态计算 FADE 损失（$\mathcal{L}_{TQF}$ 或 $\mathcal{L}_{TMC}$），反向传播梯度到 PGD 优化循环更新扰动参数。

### 关键设计

1. **时序查询洪泛（Temporal Query Flooding, TQF）**:

    - 功能：通过制造大量高置信度、时序持久的虚假轨迹来耗尽固定查询预算
    - 核心思路：三个子损失协同工作。(1) **Query Flooding** $\mathcal{L}_{Flood}$：最大化未匹配查询的检测置信度；(2) **Cost Mimicry** $\mathcal{L}_{Cost}$：让对抗查询对真实目标呈现低匹配代价，欺骗二部图匹配算法把真实目标的身份误分配给虚假查询；(3) **Identity Siphoning** $\mathcal{L}_{Siphon}$：让当前对抗查询的隐状态与上一帧合法轨迹的隐状态相似，"窃取"合法轨迹的身份和历史记录使虚假轨迹持久化
    - 设计动机：仅制造单帧假阳性不够，必须让假轨迹融入 query 状态成为持久轨迹才能真正耗尽预算

2. **时序记忆腐蚀（Temporal Memory Corruption, TMC）**:

    - 功能：直接破坏现有合法轨迹的时序记忆和隐状态
    - 核心思路：两个子损失。(1) **Temporal Decorrelation** $\mathcal{L}_{Decorr}$：最小化当前帧隐状态 $\mathcal{H}^t$ 与上一帧 $\mathcal{H}^{t-1}$ 的余弦相似度，切断时序关联使 query 更新器无法重新关联轨迹；(2) **Track Erasure** $\mathcal{L}_{Erase}$：最小化已匹配查询隐状态的 L2 范数，迫使其坍缩至零向量，擦除轨迹的特征身份
    - 设计动机：与 TQF 的"渗透"策略互补，TMC 采用"遗忘"策略，不创建新轨迹而是直接摧毁现有轨迹

3. **从数字到物理的可微分攻击管线**:

    - 功能：将数字攻击扩展到可模拟物理世界的传感器欺骗攻击
    - 核心思路：建模两种传感器欺骗的可微分模拟器——声学注入（AAI，模拟相机防抖器共振引起的运动模糊）和电磁干扰（EAI，模拟ADC转换被干扰产生的色条伪影），在统一 PGD 框架下优化物理参数 $\theta_{AAI}$ 或 $\theta_{EAI}$
    - 设计动机：传统贴片式攻击仅适用于单目标跟踪，传感器级攻击可同时影响所有目标

### 损失函数 / 训练策略

- TQF: $\mathcal{L}_{TQF} = \lambda_{Flood}\mathcal{L}_{Flood} + \lambda_c\mathcal{L}_{Cost} + \lambda_s\mathcal{L}_{Siphon}$
- TMC: $\mathcal{L}_{TMC} = \lambda_{Decorr}\mathcal{L}_{Decorr} + \lambda_{Erase}\mathcal{L}_{Erase}$
- 数字攻击：ε=8/255，α=1/255，50 次 PGD 迭代，单帧施加
- 物理攻击：α=8/255，100 次迭代，连续 3 帧施加

## 实验关键数据

### 主实验

MOT17 数字攻击结果（部分关键跟踪器）：

| 跟踪器 | 攻击 | HOTA ↓ | AssA ↓ | IDSW ↑ |
|---------|------|--------|--------|--------|
| MeMOTR | Clean | 67.35 | 79.60 | 0.81 |
| MeMOTR | Daedalus | 42.41 | 51.94 | 4.09 |
| MeMOTR | FADE_TMC | 41.56 | 49.18 | **4.63** |
| MeMOTR | FADE_TQF | 41.41 | 50.03 | 4.31 |
| CO-MOT | Clean | 58.16 | 74.87 | 1.83 |
| CO-MOT | FADE_TMC | 41.73 | 55.89 | **10.94** |
| CO-MOT | FADE_TQF | 37.26 | 51.93 | 9.50 |

MOT20 高密度场景（约150目标/帧）：

| 跟踪器 | 攻击 | HOTA ↓ | IDSW ↑ |
|---------|------|--------|--------|
| MeMOTR | Clean | 69.61 | 0.46 |
| MeMOTR | FADE_TMC | 37.70 | 4.90 |
| MeMOTR | FADE_TQF | 57.67 | 1.51 |
| MOTRv2 | Clean | 59.56 | 0.73 |
| MOTRv2 | FADE_TQF | 29.64 | 5.10 |

### 消融实验

各攻击策略对比（MOT17 CO-MOT, Clean HOTA=58.16）：

| 攻击方法 | HOTA | 相对下降 | 说明 |
|---------|------|---------|------|
| Daedalus (检测逃避) | 40.01 | -18.15 | TBD攻击直接应用 |
| F&F (关联扰动) | 52.78 | -5.38 | TBD攻击效果有限 |
| FADE_TMC | 41.73 | -16.43 | 记忆腐蚀有效 |
| FADE_TQF | **37.26** | **-20.90** | 查询洪泛最强 |

### 关键发现

- **TQF 对查询预算紧张的模型最有效**：CO-MOT 上 HOTA从 58.16 降至 37.26（-20.90），因为其标签分配策略更容易被查询洪泛利用
- **TMC 擅长制造身份切换**：CO-MOT 上 IDSW 从 1.83 飙升至 10.94（约 6 倍），直接腐蚀记忆导致频繁重标id
- **高密度场景放大攻击效果**：MOT20 上 MeMOTR 被 TMC 攻击从 69.61 HOTA 降至 37.70（-31.91），超过 MOT17 的降幅
- **现有 TBD 攻击对 TBP 部分失效**：Hijacking 和 F&F 对 MOTRv2 几乎无效（HOTA仅下降约 1-4 点），证实了 TBP 需要专用攻击

## 亮点与洞察

- **首次揭示 TBP 三大脆弱点**：固定查询预算的零和博弈、循环隐状态传播、内置时序记忆的持久性放大。这些分析对 TBP 架构的安全性设计有指导意义
- **TQF 的"身份窃取"设计巧妙**：不是简单制造假目标，而是让假轨迹与真轨迹在隐状态空间中对齐来窃取身份，利用了 TBP 自身的匹配和传播机制
- **可微分物理攻击管线**：将声学/电磁传感器欺骗建模为可微分函数纳入 PGD 优化，是将数字攻击转化为物理攻击的通用范式

## 局限与展望

- 物理攻击仍停留在仿真层面，未在真实传感器上验证
- 攻击优化需要白盒访问跟踪器权重，对黑盒场景的可迁移性未探讨
- 未提出针对这些攻击的防御方法
- 对于有外部检测器增强的 MOTRv2，TQF 效果不如其他跟踪器，说明攻击对架构变体的鲁棒性有待提升

## 相关工作与启发

- **vs Daedalus/Hijacking/F&F**: 这些是 TBD 专用攻击，依赖 NMS/Kalman/外观库，在 TBP 上效果有限或完全失效。FADE 直接攻击 query 传播和时序记忆
- **vs BankTweak**: BankTweak 假设可以直接访问推理管线注入噪声到外观特征，不实际。FADE 通过图像级扰动攻击
- 这项工作为 TBP 跟踪器的鲁棒性研究开辟了新方向，可激发防御方法如查询预算动态调整、隐状态安全校验等

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次针对 TBP 跟踪器的专用对抗攻击，TQF/TMC 设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 5 种 SOTA 跟踪器 × 2 个数据集 × 数字+物理攻击，对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但公式符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐ 对安全关键应用（自动驾驶、监控）中 TBP 跟踪器的脆弱性提供了重要警示

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[AAAI 2026\] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](../../AAAI2026/video_understanding/rethinking_progression_of_memory_state_in_robotic_manipulation_an_object-centric.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2025\] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](../../CVPR2025/video_understanding/dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)

<!-- RELATED:END -->
