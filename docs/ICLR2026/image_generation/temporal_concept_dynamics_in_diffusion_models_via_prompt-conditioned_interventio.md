---
title: >-
  [论文解读] Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions
description: >-
  [ICLR 2026][图像生成][概念时间动力学] 提出 PCI（Prompt-Conditioned Intervention）框架，通过在去噪轨迹不同时间步切换文本提示，量化概念何时在扩散模型中锁定，并将此发现应用于时间感知的图像编辑。
tags:
  - ICLR 2026
  - 图像生成
  - 概念时间动力学
  - 提示条件干预
  - 概念插入成功率
  - 扩散可解释性
  - 训练免费编辑
---

# Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions

**会议**: ICLR 2026  
**arXiv**: [2512.08486](https://arxiv.org/abs/2512.08486)  
**代码**: [PCI Framework](https://github.com/agoerguen/PCI)  
**领域**: 扩散模型 / 可解释性 / 图像编辑  
**关键词**: 概念时间动力学, 提示条件干预, 概念插入成功率, 扩散可解释性, 训练免费编辑

## 一句话总结

提出 PCI（Prompt-Conditioned Intervention）框架，通过在去噪轨迹不同时间步切换文本提示，量化概念何时在扩散模型中锁定，并将此发现应用于时间感知的图像编辑。

## 研究背景与动机

扩散模型通常仅通过最终输出评估，但生成过程是沿轨迹展开的动态过程：

**时间动态被忽视**：现有可解释性方法大多关注"哪里"（归因图）或"什么"（概念瓶颈），而非"何时"

**静态分析的不足**：
   - 归因图定位概念但不回答概念何时出现
   - 概念瓶颈模型需额外训练且不忠实于原始模型
   - 稀疏自编码器在单一时间步评估

**编辑缺乏时间感知**：现有编辑方法不知道何时干预最有效

**核心问题**：噪声何时变成特定概念（如年龄、天气），并在去噪轨迹中锁定？

## 方法详解

### 1. 提示条件干预（PCI）

**基本流程**：
1. 使用基础提示 $P_b$ 开始去噪
2. 在时间步 $t_s$ 切换为概念提示 $P_c$（基础提示+目标概念）
3. 继续去噪直至生成最终图像
4. 使用 VQA 模型（Qwen-VL-3B）检测概念是否存在

$$\mathbf{x}_{t_s} = \text{Denoise}(\mathbf{x}_T, P_b)$$
$$\mathbf{x}_0(P_b \xrightarrow{t_s} P_c) = \text{Denoise}(\mathbf{x}_{t_s}, P_c)$$

**特点**：训练免费、模型无关、无需访问模型内部

### 2. 概念插入成功率（CIS）

定义为：在时间步 $t_s$ 插入概念后、概念出现在最终图像中的概率。

- 通过多种随机种子和基础提示平均
- 单调非递减，定义良好的水平穿越时间 $\tau_q$
- CIS 曲线揭示概念的时间行为

**关键指标**：
- $\tau_{50}$, $\tau_{70}$：CIS 达到 50%/70% 的穿越时间步
- $W_{70 \to 50} = |\tau_{70} - \tau_{50}|$：过渡窗口宽度

### 3. 概念分类体系

覆盖约800个细粒度概念描述：
- 人口统计（性别、种族、年龄组）
- 物体（动物、人造物品、自然元素）
- 人类属性（衣着、配饰、体貌特征）
- 动作、属性、环境因素、风格

每个概念在8种不同上下文中评估。

## 实验

### 评估模型
SD 2.1, SDXL, SD 3.5, PixArt-alpha, FLUX.1-dev

### 核心发现

#### 跨类别时间层级

| 概念类型 | 锁定时间 | 特点 |
|---------|---------|------|
| 全局因素（风格、时间、天气、季节、颜色） | **早期** | 过渡窗口窄 |
| 人类属性（年龄、性别） | **中期** | 中等窗口 |
| 细节属性（配饰） | **中后期** | 较宽窗口 |
| 非分布概念（客厅里的马） | **异常早期** | 窗口窄且脆弱 |

#### 跨模型差异

| 模型类型 | 特点 |
|---------|------|
| 扩散模型（SD 2.1, SDXL） | 保持更多后期灵活性 |
| 整流流模型（SD 3.5, FLUX） | 概念锁定更早，过渡更陡 |
| PixArt-alpha (DiT) | 介于两者之间 |

#### 上下文依赖性

- 同一概念在不同上下文中插入时间显著不同
- 例：婴儿在"游乐场"比"公交站"锁定更晚（更自然的上下文）
- 例：穿手术服在"医院"比"街道"锁定更晚
- **OOD概念锁定更早**：不常见的概念-上下文组合导致更早锁定

### 图像编辑应用

| 方法 | CLIP_img↑ | CLIP_txt↑ | CLIP_dir↑ |
|------|-----------|-----------|-----------|
| NTI+P2P | 0.867 | 0.222 | 0.098 |
| Stable Flow | 0.832 | 0.215 | 0.063 |
| PCI-$\tau_{50}$ | 0.889 | **0.224** | **0.139** |
| PCI-$\tau_{60}$ | 0.863 | **0.229** | **0.153** |
| PCI-$\tau_{70}$ | 0.835 | **0.234** | **0.168** |

CIS 引导的编辑窗口 $[\tau_{50}, \tau_{70}]$ 在所有指标上实现最佳的编辑-保持平衡。

### 消融实验

| 设置 | 效果 |
|------|------|
| 不同 VQA 模型 | 结果一致 |
| 提示措辞变化 | 鲁棒 |
| 种子数量 | 平均后种子噪声被压制 |

## 亮点

1. **开创性的时间维度分析工具**：将扩散时间变为可解释的分析轴
2. **发现丰富的时间行为模式**：全局→人类→细节的锁定层级
3. **跨模型对比揭示架构影响**：整流流 vs 扩散模型的时间差异
4. **实用的编辑应用**：CIS引导的编辑在所有指标上超越SOTA
5. **零训练、零成本**：整个框架无需任何训练

## 局限性

1. CIS 依赖 VQA 模型（Qwen-VL-3B），可能引入评估偏差
2. 概念的二值判定（是/否）可能过于粗糙
3. 分析主要针对文本到图像模型，视频扩散的时间动态未探索
4. 多概念交互分析仍较初步
5. CIS引导编辑的自动化（自动选择最优 $\tau$）需要先运行完整CIS曲线

## 相关工作

- **静态可解释性**：归因图 (Tang 2022)、概念瓶颈 (Ismail 2024)
- **动态可解释性**：P2P (Hertz 2023)、稀疏自编码器 (Tinaz 2025)
- **扩散编辑**：NTI+P2P、Stable Flow、SDEdit

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 全新的时间维度分析范式
- **实用性**: ⭐⭐⭐⭐ — 编辑应用实用，分析洞察有价值
- **实验**: ⭐⭐⭐⭐⭐ — 800+概念描述，5个模型，分析极其全面
- **写作**: ⭐⭐⭐⭐⭐ — 结构清晰，发现有趣且表达精准

<!-- RELATED:START -->

## 相关论文

- [Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)
- [Intention-Conditioned Flow Occupancy Models](intention-conditioned_flow_occupancy_models.md)
- [SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)

<!-- RELATED:END -->
