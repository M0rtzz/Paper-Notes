---
title: >-
  [论文解读] Open-World Skill Discovery from Unsegmented Demonstration Videos
description: >-
  [ICCV 2025][图像分割][技能发现] 受人类认知事件分割理论（EST）启发，提出 Skill Boundary Detection (SBD) 算法，利用预训练无条件动作预测模型的**预测误差跳变**来自动识别未分割演示视频中的技能边界，在 Minecraft 中显著提升条件策略和层级智能体的表现。
tags:
  - ICCV 2025
  - 图像分割
  - 技能发现
  - 时序视频分割
  - 行为克隆
  - 开放世界
  - Minecraft
---

# Open-World Skill Discovery from Unsegmented Demonstration Videos

**会议**: ICCV 2025  
**arXiv**: [2503.10684](https://arxiv.org/abs/2503.10684)  
**代码**: [craftjarvis.github.io/SkillDiscovery](https://craftjarvis.github.io/SkillDiscovery/)  
**领域**: 图像分割  
**关键词**: 技能发现, 时序视频分割, 行为克隆, 开放世界, Minecraft

## 一句话总结

受人类认知事件分割理论（EST）启发，提出 Skill Boundary Detection (SBD) 算法，利用预训练无条件动作预测模型的**预测误差跳变**来自动识别未分割演示视频中的技能边界，在 Minecraft 中显著提升条件策略和层级智能体的表现。

## 研究背景与动机

构建开放世界智能体的关键挑战之一是**从长视频中学习原子技能**。层级智能体通常采用"规划器+控制器"架构：规划器将高层指令拆解为原子技能，控制器执行单个技能。训练这种架构需要将长轨迹分割为独立技能片段，但现实中的演示视频**通常很长且未分割**。

现有分割方法的缺陷：

**随机分割**（固定长度）：不保证每段包含独立完整技能，且预设长度可能与实际技能时长不匹配

**奖励驱动**：无法捕获无奖励关联的技能，且重复获得奖励时会错误分割

**自顶向下**（人工预定义技能集）：昂贵且技能多样性有限

**自底向上**（聚类/BPE）：在视觉部分可观测环境中，仅基于动作序列效果差

所有方法都依赖**人为设计规则**，需要一种**基于学习**的自适应方法。

核心洞察（来自人类认知科学 EST 理论）：人类在知觉预期的**预测误差升高时**会自然将连续经验分割为离散事件。类比到智能体——当无条件策略的预测误差突然增大时，表明正在发生技能转换。

## 方法详解

### 整体框架

四阶段流水线：

**Stage I**: 在未分割数据集上预训练 Transformer-XL 无条件策略 $\pi_{unconditional}$，进行行为克隆（动作标签由逆动力学模型生成）

**Stage II**: 使用 SBD 算法将长视频分割为原子技能片段

**Stage III**: 在分割后的数据集上训练条件策略（视频条件 GROOT / 文本条件 STEVE-1）

**Stage IV**: 将条件策略与视觉语言模型结合，构建层级智能体

### 关键设计

1. **Skill Boundary Detection (SBD) 算法**: 在每个时间步 $t$，用无条件模型预测动作并计算与真值的损失。当损失超过历史平均值一个阈值 GAP 时，标记为技能边界。滑动窗口模拟模型的记忆，边界处清空记忆。

    - 核心判据：$\text{loss} - \text{mean}(\text{loss\_history}) > \text{GAP}$
    - 超参数 GAP 设为 18，综合考虑平均轨迹长度和语义。

2. **理论保证 — 预测概率的边界定理**: 基于三个假设建立了技能切换检测的理论基础：

    - **技能一致性**：$P(\pi_{t+1} \neq \pi_t | o_{1:t+1}) < 1/K$（技能不频繁切换）
    - **技能置信度**：$P(\pi_t(a_t|o_{1:t}) > c) > 1 - \delta$（智能体对动作有高置信度）
    - **技能转换时的动作偏离**：切换技能时会执行在旧技能下概率极低的动作
   
   **定理 3.4** 证明：未切换时相对预测概率有高下界，切换时有低上界。当 $c > m$ 且 $(K-4)c^2 > 2$ 时，两个边界不重叠，保证可区分。

3. **外部信息辅助**: 可选组件——利用游戏内日志（如 crafting 事件）标记难以通过损失检测的边界。仅在检测失败时作为补充，对纯视觉数据也有效。

### 损失函数 / 训练策略

**无条件策略训练**：标准行为克隆
$$\min_\theta \sum_{t \in [1...T]} -\log \pi_{unconditional}(a_t | o_{1:t})$$

**SBD 使用的预测损失**：负对数似然 $-\log P(a_t | o_{1:t})$

**条件策略训练**：
- GROOT：C-VAE 编码 128 帧视频指令 → 行为克隆
- STEVE-1：VPT 模型适配 MineCLIP 潜空间 → 文本/视频指令跟随

## 实验关键数据

### 主实验：原子技能基准

| 策略 | 指令类型 | 原始版 avg | SBD版 avg | **相对提升** |
|------|---------|-----------|----------|------------|
| GROOT | 视频条件 | 9.5 | 25.4 | **+63.7%** |
| STEVE-1 | 图像+文本 | 46.9 | 71.9 | **+52.1%** |

典型技能提升：

| 技能 | GROOT 原始 | GROOT SBD | 提升 |
|------|-----------|----------|------|
| hunt sheep | 26% | 54% | +107.7% |
| use bow | 30% | 80% | +166.7% |
| collect wood (find+collect) | 14.5 | 19.7 | +36.1% |

### 长程任务：层级智能体

| 方法 | Wood | Food | Stone | Iron | 平均相对提升 |
|------|------|------|-------|------|------------|
| OmniJARVIS (原始) | 95% | 44% | 82% | 32% | - |
| OmniJARVIS (SBD) | 96% | 55% | 90% | 35% | **+11.3%** |

| 方法 | Diamond | Armor | Food | 平均相对提升 |
|------|---------|-------|------|------------|
| JARVIS-1 (原始) | 8% | 12% | 39% | - |
| JARVIS-1 (SBD) | 10% | 19% | 62% | **+20.8%** |

### 消融实验

| 配置 | 平均成功率 | 说明 |
|------|----------|------|
| 随机分割 (128帧固定) | 基线 | GROOT 原版默认 |
| SBD (仅损失) | 提升较大 | 纯基于预测误差检测 |
| SBD (损失+外部信息) | **最优** | 结合游戏事件日志 |

### 关键发现

- SBD 产生的分割**长度分布更接近真实技能时长**，而随机分割偏向固定长度
- **损失跳变与技能边界高度相关**，验证了 EST 理论在智能体场景的适用性
- 在**无外部信息**的数据集上 SBD 仍有效，证明核心机制是损失检测，外部信息是可选增强
- SBD 可利用 YouTube 视频训练指令跟随智能体，降低数据标注成本

## 亮点与洞察

- **认知科学启发的清晰动机**：EST 理论 → 预测误差检测技能边界，理论和直觉高度一致
- **理论保证**：定理 3.4 给出了技能切换/不切换时预测概率的可区分边界，不是纯经验方法
- **通用性强**：SBD 仅需一个预训练的无条件策略，无需额外标注、奖励信号或预定义技能集
- **即插即用**：可直接替换现有方法的分割步骤（GROOT、STEVE-1、OmniJARVIS），均获提升

## 局限与展望

- GAP 超参数需要手动调优，对不同环境/数据集可能需要不同值
- 对于动作变化不明显的技能转换（如 Minecraft crafting）效果有限，依赖外部信息
- 仅在 Minecraft 环境验证，需扩展到机器人操作、自动驾驶等领域
- 未讨论如何自适应确定技能数量（当前依赖后处理修剪长度）

## 相关工作与启发

- EST 理论（Zacks et al.）将人类事件分割与预测误差关联，为计算方法提供认知科学基础
- 与 Option Framework (Sutton 1999) 在层级强化学习中的技能发现思路互补
- Transformer-XL 的长序列建模能力是 SBD 有效性的基础

## 评分

- 新颖性：⭐⭐⭐⭐ — 认知科学理论 + 学习方法 + 理论证明的结合
- 实验充分度：⭐⭐⭐⭐ — 多策略 + 多智能体 + 短程/长程任务
- 实用性：⭐⭐⭐⭐ — 即插即用，可利用 YouTube 数据
- 总体：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting](../../CVPR2025/segmentation/rocket-1_mastering_open-world_interaction_with_visual-temporal_context_prompting.md)
- [ReferEverything: Towards Segmenting Everything We Can Speak of in Videos](refereverything_towards_segmenting_everything_we_can_speak_of_in_videos.md)
- [V-CLR: View-Consistent Learning for Open-World Instance Segmentation](../../CVPR2025/segmentation/v-clr_view-consistent_learning_for_open-world_instance_segmentation.md)
- [Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)
- [2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)

<!-- RELATED:END -->
