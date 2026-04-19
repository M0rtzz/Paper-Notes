---
title: >-
  [论文解读] Language-Grounded Decoupled Action Representation for Robotic Manipulation
description: >-
  [CVPR 2026][机器人] 提出 LaDA 框架，将连续 7-DoF 机器人动作解耦为语言描述的可解释运动基元（平移、旋转、夹爪），通过语义引导的软标签对比学习统一视觉-语言-动作表示空间，实现跨任务泛化。
tags:
  - CVPR 2026
  - 机器人
---

# Language-Grounded Decoupled Action Representation for Robotic Manipulation

**会议**: CVPR 2026  
**arXiv**: [2603.12967](https://arxiv.org/abs/2603.12967)  
**代码**: 无  
**领域**: 机器人

## 一句话总结

提出 LaDA 框架，将连续 7-DoF 机器人动作解耦为语言描述的可解释运动基元（平移、旋转、夹爪），通过语义引导的软标签对比学习统一视觉-语言-动作表示空间，实现跨任务泛化。

## 背景与动机

1. **高层语义与低层控制的异质性鸿沟**：当前 VLA 模型在视觉-语言理解和精细动作控制之间缺乏有效桥梁，高层语义指令（如"倒水"）难以直接映射为精确的运动参数。
2. **共享运动基元未被利用**：语义不同的任务（如"倒水"和"放瓶子"）往往共享底层运动基元（伸手、抓取、旋转），现有模型无法复用这些共享结构，导致冗余学习和跨任务泛化差。
3. **现有范式的固有缺陷**：
    - 端到端 VLA：感知与控制耦合，缺乏可解释性和运动结构复用
    - 隐式动作学习：潜空间由视觉差异定义，缺乏显式语义，跨任务迁移受限
    - 语言条件策略：依赖粗粒度离散基元（如"向前移动"），缺少精细运动参数（平移量、旋转角度）
4. **缺乏语义接地层**：根本原因在于符号意图与连续执行之间缺少语义基础层，语言天然适合承担这一角色。

## 方法详解

### 3.1 总体架构

LaDA 使用语言作为语义桥梁，将视觉、语言和动作统一在共享嵌入空间中。核心流程为：动作解耦 → 语义对比学习 → 自适应加权 → 微调推理。

### 3.2 语言接地的动作解耦 (Language-Grounded Action Decomposition)

将每个 7-DoF 末端执行器动作 $\mathbf{a}_t$ 投影为三类可解释运动基元 $\Pi: \mathbf{a}_t \mapsto \mathbf{p}_t$：

| 基元类型 | 符号 | 语言模板示例 |
|----------|------|-------------|
| 平移基元 | $\Delta T$ | "Move [dist] meters along [dir]" |
| 旋转基元 | $\Delta R$ | "Rotate [mag] degrees around [axis]" |
| 夹爪基元 | $G$ | "Open" / "Close" |

每个基元被离散化为语言对齐的类别，将连续控制轨迹转化为可解释的语义类别。这种解耦在低层运动学与高层语义之间建立桥梁，支持跨任务对齐与组合式泛化。

### 3.3 语义引导的对比学习

#### 软标签相似度构建

构建软标签相似度矩阵 $S \in [0,1]^{N \times N}$，编码基元级别的语义亲和度：

$$S = \frac{w_t M_t + w_r M_r + w_g M_g}{w_t + w_r + w_g}$$

其中 $M_t$、$M_r$、$M_g$ 分别为平移、旋转、夹爪的二值匹配矩阵，$(w_t, w_r, w_g)$ 为超参数。每个 $S_{ij}$ 表示动作 $i$ 与 $j$ 之间细粒度的基元级语义相似度。

#### 双路径软标签对比学习

使用预训练 CLIP 编码器提取视觉 token $v_i = f_v(V_i)$ 和语言 token $l_i = f_l(L_i)$，通过 FiLM 融合并用 MLP 投影：$A_i = \text{MLP}(\text{FiLM}(v_i, l_i))$。

**路径一：动作-动作对齐**，使共享基元属性的动作在嵌入空间中更近：

$$\mathcal{L}_a = -\sum_{i=1}^N \sum_{j=1}^N S_{ij} \log \frac{\exp(\text{sim}(A_i, A_j) / \tau)}{\sum_{k=1}^N \exp(\text{sim}(A_i, A_k) / \tau)}$$

**路径二：动作-基元对齐**，将每个动作锚定到其基元语言描述 $P_j = f_l(\mathcal{D}(p_j))$：

$$\mathcal{L}_m = -\sum_{i=1}^N \sum_{j=1}^N S_{ij} \log \frac{\exp(\text{sim}(A_i, P_j) / \tau)}{\sum_{k=1}^N \exp(\text{sim}(A_i, P_k) / \tau)}$$

总对比损失：$\mathcal{L}_{\text{CL}} = \mathcal{L}_a + \lambda \mathcal{L}_m$

### 3.4 自适应损失加权

模仿损失 $\mathcal{L}_{\text{IL}}$（预测离散化基元类别）与对比损失 $\mathcal{L}_{\text{CL}}$ 具有不同收敛特性。使用滑动平均自适应加权：

$$w_{\text{IL}} = \frac{\text{MA}(\mathcal{L}_{\text{IL}})}{\text{MA}(\mathcal{L}_{\text{IL}}) + \text{MA}(\mathcal{L}_{\text{CL}})}, \quad w_{\text{CL}} = \frac{\text{MA}(\mathcal{L}_{\text{CL}})}{\text{MA}(\mathcal{L}_{\text{IL}}) + \text{MA}(\mathcal{L}_{\text{CL}})}$$

最终目标：$\mathcal{L}_{total} = w_{\text{CL}} \mathcal{L}_{\text{CL}} + w_{\text{IL}} \mathcal{L}_{\text{IL}}$

### 3.5 微调与推理

预训练后使用轻量 MLP 动作头进行 7-DoF 动作预测的微调（$\mathcal{L}_1$ 轨迹回归损失）。推理时直接从 $(V_t, L_t)$ 输出连续动作，无需显式基元标签。

## 实验结果

### 预训练数据

使用 Open X-Embodiment (OXE) 数据集，约 2250 万视觉帧，涵盖 22 种机器人形态，每个动作为 7-DoF 控制向量。

### LIBERO 基准测试

| 模型 | 参数量 | Spatial | Object | Goal | Long | 平均 |
|------|--------|---------|--------|------|------|------|
| UniACT | 0.5B | 65.0 | 78.0 | 68.0 | 47.0 | 64.5 |
| OpenVLA | 7.5B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π-FAST | 2B | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| FlowVLA | 8.5B | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| CLIP-RT | 1.3B | 95.2 | 99.2 | 94.2 | 83.8 | 93.1 |
| **LaDA** | **0.6B** | **95.2** | **99.2** | **93.6** | **86.4** | **93.6** |

LaDA 仅用 0.6B 参数即取得 93.6% 平均成功率（SOTA），在 LIBERO-Long 上尤为突出（86.4%），超过参数量大数倍的模型。

### MimicGen 基准测试

| 模型 | C_D0 | C_D1 | S_D0 | S_D1 | ST_D0 | ST_D1 | T_D0 | TPA_D0 | TPA_D1 | 平均 |
|------|------|------|------|------|-------|-------|------|--------|--------|------|
| OpenVLA | 42% | 18% | 84% | 86% | 36% | 20% | 20% | 28% | 8% | 38% |
| Phoenix | 94% | 48% | 96% | 86% | 50% | 20% | 68% | 52% | 6% | 58% |
| CLIP-RT* | 77% | 34% | 93% | 87% | 68% | 52% | 32% | 11% | 4% | 51% |
| **LaDA** | **94%** | **46%** | **96%** | **95%** | **76%** | **71%** | **48%** | **50%** | **25%** | **67%** |

LaDA 平均成功率 67%，比 Phoenix 高 9%，比 CLIP-RT* 高 16%，在多步长任务（StackThree_D1: 71%）上优势显著。

### 消融实验

| 方法 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| w/o SCL | 79.2 | 82.8 | 76.6 | 63.4 | 75.5 |
| w/o AW | 93.6 | 94.4 | 87.2 | 74.4 | 87.4 |
| **LaDA** | **95.2** | **99.2** | **93.6** | **86.4** | **93.6** |

去除软标签对比学习（SCL）导致性能骤降 18.1%，验证了细粒度语义对齐的关键作用；去除自适应加权（AW）也降低 6.2%。

### 泛化能力

在跨任务泛化测试中，CLIP-RT* 对未见过的"push"指令成功率为 0%，而 LaDA 达到 12.3%。在多任务训练场景中，LaDA 从多任务联合训练中获得显著收益，而 CLIP-RT 仅有边际提升。

## 亮点

- **语言作为语义桥梁的创新视角**：将语言从任务指令提升为连接感知与控制的通用接口，实现动作的语义接地
- **细粒度可解释基元**：不同于粗粒度的"向前移动"，LaDA 的基元包含精确运动参数（距离、角度），实现真正的语义-控制对齐
- **参数效率极高**：0.6B 参数超越 7B+ 的大模型（OpenVLA、CoT-VLA），性能/参数比极优
- **软标签对比学习**：突破硬正负样本对的局限，通过连续亲和度权重捕捉细粒度运动对应关系
- **多基准全面验证**：覆盖 LIBERO（语言条件多任务）、MimicGen（接触密集操控）及真机部署

## 局限性

- **依赖预定义基元离散化**：平移/旋转的离散化粒度是超参数，可能难以覆盖所有精细运动需求
- **真机实验规模有限**：仅在单个 pick-and-place 任务上验证，未展示更复杂的真机操控场景
- **预训练数据依赖**：依赖大规模 OXE 数据集进行预训练，数据获取成本较高
- **泛化上限受限**：跨任务泛化虽优于基线但绝对成功率仍较低（12.3%），说明零样本泛化仍是开放问题

## 评分

- ⭐⭐⭐⭐ 新颖性：语言接地的动作解耦思路新颖，软标签对比学习在机器人动作表示中属首创
- ⭐⭐⭐⭐ 实用性：0.6B 参数即达 SOTA、框架通用性强，对实际部署友好
- ⭐⭐⭐⭐ 实验充分度：两个仿真基准 + 真机 + 全面消融 + 泛化测试，覆盖面广
- ⭐⭐⭐ 写作质量：结构清晰，但部分符号和公式可进一步统一简化

<!-- RELATED:START -->

## 相关论文

- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2025/robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)

<!-- RELATED:END -->
