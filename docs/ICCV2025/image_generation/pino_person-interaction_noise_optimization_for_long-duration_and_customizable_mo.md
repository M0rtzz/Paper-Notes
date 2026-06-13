---
title: >-
  [论文解读] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups
description: >-
  [ICCV 2025][图像生成][多人交互生成] 提出Person-Interaction Noise Optimization（PINO），一种无需训练的框架，将复杂的多人群体交互分解为语义明确的两人交互对，利用预训练的两人交互扩散模型通过噪声优化和物理惩罚项顺序合成任意规模的群体交互运动，支持精细化用户控制和长时序运动生成。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "多人交互生成"
  - "噪声优化"
  - "training-free"
  - "运动扩散模型"
  - "物理约束"
---

# PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups

**会议**: ICCV 2025  
**arXiv**: [2507.19292](https://arxiv.org/abs/2507.19292)  
**代码**: [GitHub](https://sinc865.github.io/pino/)  
**领域**: 运动生成/扩散模型  
**关键词**: 多人交互生成, 噪声优化, training-free, 运动扩散模型, 物理约束

## 一句话总结

提出Person-Interaction Noise Optimization（PINO），一种无需训练的框架，将复杂的多人群体交互分解为语义明确的两人交互对，利用预训练的两人交互扩散模型通过噪声优化和物理惩罚项顺序合成任意规模的群体交互运动，支持精细化用户控制和长时序运动生成。

## 研究背景与动机

生成逼真的多人群体交互运动在动画、游戏和机器人领域有重要需求，但随着人数增加，交互复杂度呈指数增长，使得这一问题极具挑战性。现有解决方案存在以下关键不足：

**训练数据瓶颈**：Shan et al.等方法需要在专门的多人数据集上训练，但注释成本随人数急剧增加，且模型被限制在固定群组大小

**缺乏灵活控制**：InterControl和FreeMotion等基于ControlNet的方法使用单一共享prompt描述整个群体，无法为每个角色指定不同的交互关系，导致生成的群体动作过于均匀和简化（如"三人手拉手走路"）

**物理真实性缺失**：现有方法缺乏对重叠和穿透等物理伪影的约束，随着角色增多问题愈发严重。FreeMotion虽然条件化生成，但不施加生成后的关系约束

**扩展性受限**：要实现精细控制（如碰撞避免、朝向控制、区域限制）通常需要为每种控制信号重新训练ControlNet

PINO的核心洞察是：**群体交互本质上由更小的、互联的两人交互组成**。例如合影不是一个整体交互，而是摄影师与每个被拍者之间、以及被拍者之间的多对两人交互的组合。共享角色（如摄影师）起到连接不同交互的枢纽作用。

## 方法详解

### 整体框架

PINO的输入包含两部分：（1）有序的参与者列表，指定每组两人交互的目标/参考角色；（2）描述每对交互的独立文本prompt。框架按以下流程工作：

1. 使用预训练两人扩散模型生成第一对交互
2. 对第一对交互进行噪声优化以消除物理伪影
3. 逐步引入新角色，每个新角色以一个已存在角色为枢纽（reference），使用独立prompt生成
4. 对每个新角色的初始噪声进行优化，约束与所有已有角色的物理关系

### 关键设计

1. **基于掩码的条件扩散生成**：使用修改版的两人扩散模型 $G_\theta^{mask}$，在去噪过程中将参考角色的运动序列 $\hat{\mathbf{x}}_0^{cond}$ 的噪声版本替换条件序列 $\mathbf{x}_t^{cond}$，仅对目标角色的噪声 $\mathbf{x}_T^{tgt}$ 进行去噪。这确保新生成的运动与参考角色的已有运动保持一致。

   对于第 $p$ 个目标角色，其噪声优化为：

    $\hat{\mathbf{x}}_T^p \leftarrow \arg\min_{\mathbf{x}_T^p} \mathcal{L}\left(G_\theta^{mask}(\mathbf{x}_T^p, \hat{\mathbf{x}}_0^{k_p}, c_{k_p, p}), \{\mathbf{x}_0^{i \in \mathcal{I}}\}\right)$

   其中 $\mathcal{I}$ 是预定义的参与优化的已有角色子集。

2. **物理惩罚项设计**：

   **重叠避免损失**：当两人的根节点位置距离小于阈值 $\delta$ 时施加惩罚：

    $\mathcal{L}_{overlap} = \sum_i \sum_n \max\left(0, \delta - \|\mathbf{p}_{root}^p(n) - \hat{\mathbf{p}}_{root}^i(n)\|_2\right)$

   **时空运动控制损失** $\mathcal{L}_{control}$ 包含四种可微分惩罚：
    - 根节点位置惩罚：约束角色在特定时刻到达指定位置
    - 运动区域惩罚：限制角色在定义区域内移动
    - 朝向惩罚：控制特定帧的面朝方向
    - 相对位置惩罚：维持角色间的期望距离或朝向关系

   总优化损失为：$\mathcal{L} = \mathcal{L}_{overlap} + \mathcal{L}_{control}$

3. **长时序运动生成**：通过运动修复（inpainting）技术扩展交互时长。从已有序列的最后 $n$ 帧作为上下文参考，在每个去噪步骤中通过二值掩码替换初始帧：

    $\mathbf{x}_t^i \leftarrow \mathbf{m} \odot \hat{\mathbf{x}}^i + (1 - \mathbf{m}) \odot \mathbf{x}_t^i$

   同时添加边界惩罚以最小化关节加速度，确保新旧段之间的运动过渡自然。结合prompt切换，可在长序列中实现角色交替（如三人轮流握手）。

### 损失函数 / 训练策略

本方法完全training-free，依赖预训练的InterGen模型（在InterHuman数据集上训练的两人交互扩散模型）。
- 使用50步DDIM采样器
- 噪声优化学习率0.003，100步优化迭代
- 通过反向传播穿过扩散过程来优化初始噪声

## 实验关键数据

### 主实验

两人交互生成（InterHuman测试集300样本）：

| 方法 | Overlap ↓ | PenVol.(cm³) ↓ | FID ↓ | R-Prec. ↑ | Diversity |
|------|-----------|----------------|-------|-----------|-----------|
| GT | 0.029 | 471.75 | 0.983 | 0.715 | 7.921 |
| InterGen | 0.119 | 3112.72 | 13.278 | 0.674 | 7.793 |
| **PINO-InterGen** | **0.000** | **275.65** | **13.163** | **0.675** | **7.904** |

多人交互生成（5人，逐步添加，以第1人为枢纽）：

| 方法 | 对 | FID ↓ | Overlap ↓ |
|------|-----|-------|-----------|
| FreeMotion | (1,5) | 25.671 | 0.991 |
| InterGen | (1,5) | 19.501 | 0.977 |
| **PINO-InterGen** | (1,5) | **16.911** | **0.069** |

### 消融实验

惩罚项逐步添加的效果（24个序列生成）：

| 配置 | 位置误差 ↓ | 重叠 ↓ | 区域违规 ↓ | 朝向误差 ↓ |
|------|-----------|--------|-----------|-----------|
| InterGen（基线） | 1.0 | 0.292 | 0.500 | 1.0 |
| + $\mathcal{L}_{root}$ | **0.0** | 0.333 | 0.917 | 1.0 |
| + $\mathcal{L}_{overlap}$ | 0.0 | **0.0** | 0.958 | 1.0 |
| + $\mathcal{L}_{region}$ | 0.083 | 0.0 | **0.043** | 1.0 |
| + $\mathcal{L}_{orientation}$ | 0.083 | 0.043 | 0.083 | **0.208** |

每个惩罚项有效降低其对应的误差。完整惩罚组合在所有指标上均达到最低违规率。

### 关键发现

- PINO将Overlap从0.119降至0.000（两人）和从0.991降至0.069（五人），几乎完全消除了角色重叠
- 穿透体积从3112.72 cm³降至275.65 cm³（甚至优于GT的471.75 cm³）
- 在改善物理真实性的同时，语义质量（FID、R-Precision）也有提升
- 运动扩展实验中，PINO的脚滑（Foot Skate）从0.070降至0.045，运动更自然

## 亮点与洞察

- "群体交互 = 互联的两人交互"的分解思路简洁而有效，将指数复杂度问题转化为线性
- 完全training-free：无需多人数据集，无需重新训练，仅通过噪声优化实现精细控制
- 支持为每对交互指定独立prompt，远比共享prompt更灵活
- 物理惩罚项设计为可微分形式，可无缝嵌入扩散模型的噪声优化循环
- 运动扩展+prompt切换组合可实现丰富的时序交互叙事

## 局限与展望

- 性能依赖基础两人模型（InterGen）的质量，最终会受其能力上限约束
- 聚焦于两人交互的分解方式无法建模高阶协调行为（如一人推另一人撞到第三人的连锁反应）
- 基于关节位置的惩罚无法完全避免手部区域的穿透（InterHuman数据集缺少手部关节数据）
- 噪声优化需要100步迭代，计算开销随角色数线性增长
- 未与基于专用多人训练的方法（如Shan et al.）进行直接比较

## 相关工作与启发

- 噪声优化思路源自图像生成领域的Attend-and-Excite和InitNo，以及运动生成的ProgMoGen和DNO
- InterGen提供了两人交互的强大先验，PINO通过后处理优化将其扩展到任意人数
- 与FreeMotion（基于ControlNet的条件生成）和InterControl（关节位置引导）互补——PINO不需要额外训练

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将两人交互扩展到任意人数的training-free方案极具创新性
- **实验充分度**: ⭐⭐⭐⭐ 涵盖两人/多人/扩展/消融，物理指标全面
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，伪代码和可视化丰富
- **价值**: ⭐⭐⭐⭐ 对动画、游戏等应用具有直接价值，framework设计优雅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)

</div>

<!-- RELATED:END -->
