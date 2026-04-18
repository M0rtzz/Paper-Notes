---
title: >-
  [论文解读] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception
description: >-
  [ICCV 2025][自动驾驶][协同感知] 提出 CoopTrack，首个完全实例级端到端协同 3D 多目标跟踪框架，通过可学习的图注意力关联模块和多维特征提取实现跨Agent实例匹配与融合，在 V2X-Seq 上达到 SOTA。
tags:
  - ICCV 2025
  - 自动驾驶
  - 协同感知
  - 3D多目标跟踪
  - 端到端学习
  - 车路协同
  - 实例级融合
---

# CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception

**会议**: ICCV 2025  
**arXiv**: [2507.19239](https://arxiv.org/abs/2507.19239)  
**代码**: [GitHub](https://github.com/zhongjiaru/CoopTrack)  
**领域**: autonomous_driving  
**关键词**: 协同感知, 3D多目标跟踪, 端到端学习, 车路协同, 实例级融合

## 一句话总结

提出 CoopTrack，首个完全实例级端到端协同 3D 多目标跟踪框架，通过可学习的图注意力关联模块和多维特征提取实现跨Agent实例匹配与融合，在 V2X-Seq 上达到 SOTA。

## 研究背景与动机

单车感知受限于单一视角的局限性——遮挡、感知范围有限等。V2X（车联万物）通信使得多Agent协同感知成为可能。然而，目前协同感知研究主要集中在**单帧任务**（3D 检测），更具挑战性的**协同序列感知任务**（如协同 3D 多目标跟踪 MOT）仍然探索不足。

**现有方案的问题**：

**Tracking-by-Cooperative-Detection (TBCD)**：先协同检测，再用传统跟踪器（如 AB3DMOT）做后处理。问题是跟踪无法利用融合信息，检测和跟踪解耦优化导致次优。

**已有端到端方案 (UniV2X)**：虽然开创性地提出了端到端协同跟踪框架，但存在两个设计问题：
   - 使用**基于规则的关联**（如欧氏距离匹配），信息不充分且受位姿噪声影响
   - 采用 **fusion-before-decoding** 流程——先融合两个 Agent 的 query，再用 ego 特征解码，导致歧义和冲突

**核心 idea**：设计 **fusion-after-decoding** 管线——先各自解码得到实例级特征，再进行跨Agent关联和融合。将关联从规则驱动变为**基于图注意力的可学习关联**，利用语义+运动的多维特征提供更丰富的匹配信息。

## 方法详解

### 整体框架

CoopTrack 包含车端和路端两个子系统。每个子系统独立完成：图像特征提取 → Transformer 解码 → 多维特征提取（MDFE）。路端的实例级特征通过 V2X 通信传输到车端（传输量极低）。车端接收后依次经过：跨Agent对齐（CAA）→ 基于图的关联（GBA）→ 特征聚合 → FFN 解码最终输出。

### 关键设计

1. **多维特征提取 (MDFE)**：

    - **解耦语义与运动特征**：现有 query-based 方法将两者隐式耦合，导致解码歧义
    - 语义特征：通过 MLP 从 query 特征中提取
    - 运动特征：从粗糙 3D 检测框中提取 8 个角点的相对坐标，通过 PointNet（4 层 MLP + max pooling）编码为运动特征
    - **时序增强**：引入专用 temporal transformer block（2 层 decoder），使用正弦位置编码捕获时间依赖关系，短序列用零填充+二进制掩码处理
    - 历史特征按 FIFO 方式在滑动窗口中更新

2. **跨Agent对齐 (CAA)**：

    - 车端和路端传感器、视角、空间位置的差异导致特征域间隙
    - 核心思路：域间隙可以近似为线性变换（类似于空间坐标的刚体变换）
    - 显式空间变换：$\tilde{\mathcal{P}}^I = \mathcal{P}^I \cdot \mathbf{R}^\top + \mathbf{t}$
    - 隐式特征变换：$\tilde{\mathcal{M}}^I = \mathcal{M}^I \cdot \hat{\mathbf{R}}^\top + \hat{\mathbf{t}}$
    - 潜在旋转矩阵 $\hat{\mathbf{R}} \in \mathbb{R}^{d \times d}$ 和平移 $\hat{\mathbf{t}} \in \mathbb{R}^{1 \times d}$ 由两个 MLP 从显式位姿参数预测
    - 使用 6D 连续旋转表示和分段映射减少参数量

3. **基于图的关联 (GBA)**：

    - 构建车-路实例间的全连接关联图 $\mathcal{G} = \{\mathcal{N}, \mathcal{E}\}$
    - 节点特征：由运动+语义特征拼接后通过 MLP 提取
    - 边特征：由参考点间的欧氏距离差异通过 MLP 编码
    - 图注意力计算亲和度矩阵：
      $\hat{A} = \frac{(\mathcal{N}^V W^V)(\mathcal{N}^I W^I)^T}{\sqrt{d}} + \mathcal{E} W^E$
    - FNN + sigmoid 生成最终亲和度矩阵 $A$
    - 使用匈牙利算法从 $1 - A$ 获得匹配对

4. **特征聚合与跟踪传播**：

    - 匹配的实例对融合多维特征为一组（消除重复检测）
    - 未匹配的实例直接保留（扩展观测范围）
    - 活跃实例的语义特征作为 query feature 传播到下一帧
    - 参考点使用恒速假设预测下一帧位置

### 损失函数 / 训练策略

**两阶段训练**：

- Stage 1：分别训练车端/路端端到端跟踪模型
    - $\mathcal{L}_{\text{stage1}} = 0.25 \cdot \mathcal{L}_{\text{bbx}} + 2.0 \cdot \mathcal{L}_{\text{cls}}$
    - 分类用 Focal Loss（$\alpha=0.25, \gamma=2.0$），回归用 L1 Loss

- Stage 2：端到端协同跟踪+关联训练
    - $\mathcal{L}_{\text{stage2}} = 0.25 \cdot \mathcal{L}_{\text{bbx}} + 2.0 \cdot \mathcal{L}_{\text{cls}} + 10.0 \cdot \mathcal{L}_{\text{asso}}$
    - 关联标签通过匈牙利算法匹配预测与 GT 自动生成
    - 关联损失为 Focal Loss（$\alpha=0.5, \gamma=1.0$）

## 实验关键数据

### 主实验

V2X-Seq 数据集上与协同感知 SOTA 对比（ResNet101 backbone）：

| 方法 | 范式 | mAP↑ | AMOTA↑ | 传输量↓ |
|------|------|------|--------|--------|
| V2X-ViT | TBCD | 0.268 | 0.287 | 2.56×10⁶ |
| Where2comm | TBCD | 0.162 | 0.106 | 5.40×10⁵ |
| Late Fusion | TBCD | 0.196 | 0.263 | 6.60×10² |
| UniV2X | E2EC | 0.295 | 0.239 | 6.96×10⁴ |
| **CoopTrack** | **E2EC** | **0.390** | **0.328** | **5.64×10⁴** |

比 UniV2X 提升 +9.5% mAP 和 +8.9% AMOTA，传输量更低。

### 消融实验

各模块的增量效果（ResNet50 backbone）：

| Pipeline | MDFE | CAA | GBA | mAP↑ | AMOTA↑ |
|----------|------|-----|-----|------|--------|
| ✗ | ✗ | ✗ | ✗ | 0.310 | 0.266 |
| ✓ | ✗ | ✗ | ✗ | 0.337 | 0.277 |
| ✓ | ✓ | ✗ | ✗ | 0.345 | 0.283 |
| ✓ | ✗ | ✓ | ✗ | 0.354 | 0.304 |
| ✓ | ✓ | ✓ | ✗ | 0.355 | 0.332 |
| ✓ | ✓ | ✓ | ✓ | **0.356** | **0.346** |

历史帧数量影响：0帧→4帧，AMOTA 从 0.100 提升至 0.346，验证了时序建模的价值。

### 关键发现

- fusion-after-decoding 管线比 fusion-before-decoding 本身带来 2.7% mAP 和 1.1% AMOTA 提升
- 跨Agent对齐模块学到了隐式信息：仅对对齐模块加旋转噪声时性能下降轻微，但全局加噪声时下降显著
- 更高帧率有利于跟踪：10Hz 下 CoopTrack AMOTA 比 2Hz 高 10.7%
- Griffin 数据集（空地协同）上同样有效，验证了方法的通用性

## 亮点与洞察

- 实例级特征传输极低带宽开销（5.64×10⁴ bytes/s），仅为 BEV 特征融合的千分之一
- 可学习关联比规则匹配更鲁棒：即使参考点位置不准确，利用语义+运动特征仍能正确关联
- 多维特征解耦（语义 vs 运动）解决了 query-based 方法中隐式耦合导致的解码歧义
- 关联标签的自动生成方案巧妙利用了第一阶段模型的预测能力

## 局限与展望

- 两阶段训练流程较复杂，未来可探索端到端一阶段训练
- 仅在车-路（V2I）场景验证，V2V 多车场景待探索
- 通信延迟补偿模块虽然有效但不够精细，长延迟下性能仍有下降
- 位姿噪声对系统影响较大（全局噪声），鲁棒位姿估计很重要
- 未探索 LiDAR 输入，仅使用相机图像

## 相关工作与启发

- PF-Track 的时序 query 传播和预测思路在本文得到扩展
- ADA-Track 的可微分关联模块与本文的 GBA 思路类似但应用在不同层面
- QUEST 的实例级特征融合思路与本文的实例级传输一脉相承
- UniV2X 的开创性工作为本文提供了基准和改进方向

## 评分

- 新颖性：⭐⭐⭐⭐ 首个完全端到端可学习关联的协同跟踪框架
- 技术深度：⭐⭐⭐⭐ 多模块设计合理，理论基础扎实
- 实验充分度：⭐⭐⭐⭐⭐ 两个数据集+全面消融+定性分析
- 实用价值：⭐⭐⭐⭐ 低带宽+高性能，实用前景好
- 总体推荐：⭐⭐⭐⭐
