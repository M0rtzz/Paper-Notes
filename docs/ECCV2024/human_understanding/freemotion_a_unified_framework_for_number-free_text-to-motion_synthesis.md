---
title: >-
  [论文解读] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis
description: >-
  [ECCV 2024][人体理解][文本到动作生成] 提出FreeMotion框架，通过条件概率分解将多人运动联合分布递归拆解为单人条件运动生成，首次实现任意人数的文本驱动运动合成，并支持多人空间控制。
tags:
  - ECCV 2024
  - 人体理解
  - 文本到动作生成
  - 多人运动合成
  - 扩散模型
  - 条件运动建模
  - 空间控制
---

# FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis

**会议**: ECCV 2024  
**arXiv**: [2405.15763](https://arxiv.org/abs/2405.15763)  
**代码**: 有 (https://VankouF.github.io/FreeMotion)  
**领域**: 人体理解  
**关键词**: 文本到动作生成, 多人运动合成, 扩散模型, 条件运动建模, 空间控制

## 一句话总结

提出FreeMotion框架，通过条件概率分解将多人运动联合分布递归拆解为单人条件运动生成，首次实现任意人数的文本驱动运动合成，并支持多人空间控制。

## 研究背景与动机

文本驱动的人体运动合成（Text-to-Motion, T2M）是计算机视觉中的重要任务，在机器人控制、电影制作、动画生产等领域有广泛应用。现有方法存在三个关键局限：

**模型专用性强**：现有方法要么为单人运动设计（如MDM、MLD），要么为双人运动设计（如InterGen），无法同时支持单人和多人推理。单人方法拟合的是边缘分布 $p(\mathbf{x}^1)$，双人方法拟合的是联合分布 $p(\mathbf{x}^1, \mathbf{x}^2)$，两者网络结构不兼容。

**人数不可扩展**：由于缺少多人文本标注的动作数据集（现有最大规模的InterHuman也仅包含双人数据），以及网络设计的非可扩展性（如InterGen使用固定的共享权重双流网络+交叉注意力），现有方法无法生成超过两人的运动。

**多人空间控制困难**：虽然OmniControl等方法已实现单人运动的精确轨迹控制，但将空间控制信号引入多人运动生成仍是非平凡的挑战，因为需要同时协调多个个体的空间位置。

**核心洞察**：作者观察到，多人运动的联合分布可以通过条件概率公式进行分解：

$$p(\mathbf{x}^1, \ldots, \mathbf{x}^n) = p(\mathbf{x}^1) \prod_{i=1}^{n-1} p(\mathbf{x}^{i+1} | \mathbf{x}^i, \ldots, \mathbf{x}^1)$$

这意味着：如果我们能建模**条件运动分布**，就可以通过递归过程生成任意人数的运动。第一步生成第一个人的运动（边缘分布），第二步在已生成运动的条件下生成第二个人的运动，以此类推。这种方式将多人运动合成问题本质上**归约为一系列条件单人运动生成问题**。

## 方法详解

### 整体框架

FreeMotion框架包含两个解耦模块：**生成模块（Generation Module）**和**交互模块（Interaction Module）**。前者负责生成多样化的单人运动，后者负责将条件信号（其他人的运动）注入到当前运动的生成过程中。

运动表示采用InterGen提出的non-canonical表示，保留全局坐标信息以维持多人之间的空间关系：

$$x^p(i) = [\mathbf{j}_{pg}, \mathbf{j}_{gv}, \mathbf{j}_r, \mathbf{c}_f]$$

包含全局关节位置 $\mathbf{j}_{pg} \in \mathbb{R}^{3J}$、全局速度 $\mathbf{j}_{gv} \in \mathbb{R}^{3J}$、局部旋转 $\mathbf{j}_r \in \mathbb{R}^{6J}$ 和足部接触特征 $\mathbf{c}_f \in \mathbb{R}^4$。

### 关键设计

1. **生成模块（Generation Module）**：以Transformer为基础的扩散去噪网络，负责根据文本生成单人运动。训练时对运动 $\mathbf{x}$ 加噪得到 $\mathbf{x}_t$，模块学习将其去噪为 $\mathbf{x}_{t-1}$。文本特征由预训练CLIP-ViT-L-14提取，通过自适应层归一化（Adaptive LayerNorm）注入所有注意力层。为获取单人描述，利用LLM（ChatGPT）将多人交互描述拆分为每个人的独立描述。

2. **交互模块（Interaction Module）**：受ControlNet启发设计，用于建模条件运动之间的交互。核心组件是**交互块（Interactive Block）**：

    - 将待生成的噪声运动 $\mathbf{x}_t^1$ 和 $N-1$ 个条件运动通过共享线性层编码为隐状态
    - 交互块包含两个顺序的自注意力（SA）模块和一个掩码模块
    - 第一个SA处理噪声运动隐状态，第二个SA将第一个SA的输出与（随机掩码后的）条件运动隐状态拼接后做全局自注意力
    - 计算公式：$\mathbf{h}_t^{1,k}, \mathbf{h}^{2,k}, \ldots, \mathbf{h}^{N,k} = SA(SA(\mathbf{h}_t^{1,k-1}), Mask(\mathbf{h}^{2,k-1}, \ldots, \mathbf{h}^{N,k-1}))$
    - **设计动机**：使用全局自注意力而非交叉注意力，因为自注意力对条件运动数量无限制（长度无关特性），天然支持可变数量的运动条件。随机掩码训练使模型能适应任意数量的条件。

3. **空间控制模块**：结合显式和隐式空间引导实现多人运动的精确位置控制。

    - **显式引导**：给定目标位置 $\mathbf{s}$，通过分类器引导在每步去噪时修正偏差：$\mathbf{x}_t = \mathbf{x}_t - \eta \nabla_{\mathbf{x}_t} \|\mathbf{s}_{nj} - \mathbf{x}_{nj}\|_2$
    - **隐式引导**：将空间信号通过独立线性层编码后加到噪声运动隐状态上，训练时随机选择部分帧和关节

### 损失函数 / 训练策略

训练分两阶段：

**阶段一**（单人运动生成）：训练生成模块
$$\mathcal{L}_1 = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{foot} + \lambda_2 \mathcal{L}_{vel} + \lambda_3 \mathcal{L}_{bl}$$

**阶段二**（条件运动生成）：冻结生成模块，用其参数初始化交互模块，加入DM距离损失
$$\mathcal{L}_2 = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{foot} + \lambda_2 \mathcal{L}_{vel} + \lambda_3 \mathcal{L}_{bl} + \lambda_4 \mathcal{L}_{dm}$$

其中 $\mathcal{L}_{foot}$ 为接触损失，$\mathcal{L}_{vel}$ 为关节速度损失，$\mathcal{L}_{bl}$ 为骨骼长度损失，$\mathcal{L}_{dm}$ 为遮掩关节距离图损失。

## 实验关键数据

### 主实验

在InterHuman测试集上的双人运动生成结果：

| 方法 | R-Prec Top1↑ | FID↓ | MM Dist↓ | Diversity→ |
|------|-------------|------|----------|-----------|
| Real | 0.452 | 0.273 | 3.755 | 7.748 |
| TEMOS | 0.224 | 17.375 | 5.342 | 6.939 |
| MDM | 0.153 | 9.167 | 6.125 | 7.602 |
| ComMDM | 0.223 | 7.069 | 5.212 | 7.244 |
| InterGen* | 0.264 | 13.404 | 3.882 | 7.770 |
| **FreeMotion** | **0.326** | **6.740** | **3.848** | **7.828** |

单人运动生成（使用LLM重标注文本）：

| 方法 | R-Prec Top1↑ | FID↓ | MM Dist↓ | Diversity→ |
|------|-------------|------|----------|-----------|
| InterGen* | 0.206 | 23.415 | 3.925 | 7.514 |
| **FreeMotion** | **0.264** | **12.975** | **3.885** | **7.702** |

FreeMotion在单人和双人生成上全面超越InterGen*，FID分别降低44.6%和49.7%。

### 消融实验

| 配置 | InterDes | R-Prec1↑ | FID↓ | MM Dist↓ | Diversity→ |
|------|----------|----------|------|----------|-----------|
| GM* (仅生成模块) | ✗ | 0.300 | 8.842 | 3.863 | 7.761 |
| GM (仅生成模块) | ✓ | 0.259 | 10.749 | 3.883 | 7.645 |
| FreeMotion* | ✗ | 0.300 | 8.792 | 3.865 | 7.750 |
| **FreeMotion** | **✓** | **0.326** | **6.740** | **3.848** | **7.828** |

### 关键发现

- **解耦设计的必要性**：仅用GM时引入InterDes反而性能下降（FID: 8.842→10.749），因为GM需同时适应单人和交互文本，参数更新困难。而FreeMotion通过解耦设计有效利用了两种文本。
- **三人运动能力**：尽管仅在双人数据上训练，FreeMotion可直接推理三人运动，得益于全局自注意力的长度无关特性。
- **空间控制无损性**：空间控制模块仅在单人上训练，挂载交互模块后对多人空间控制无明显损伤。

## 亮点与洞察

- 条件概率分解的思想简洁而有效，将不可扩展的多人联合分布建模问题转化为可递归扩展的条件单人生成
- 交互模块使用ControlNet式的零初始化线性层连接，保证训练初期不破坏已训练好的生成模块
- 利用LLM自动将多人描述拆分为单人描述，解决了数据标注不足的问题

## 局限与展望

- LLM拆分文本可能导致单人描述与运动不完全匹配
- 仅在双人数据上训练，生成大量个体时可能出现人体穿模
- 递归生成方式导致推理时间随人数线性增长
- 未探索非自回归的并行多人生成方式

## 相关工作与启发

- ControlNet的"复制参数+零初始化"范式在运动生成领域同样适用
- 条件概率分解思想可推广到其他多实体生成任务（如多物体3D生成）
- 与InterGen的双流共享权重设计相比，解耦方案更具可扩展性

## 评分

- **新颖性**: ⭐⭐⭐⭐ 条件概率分解实现number-free生成的思路很新颖
- **实验充分度**: ⭐⭐⭐⭐ 涵盖单人/双人/三人定量定性实验及消融
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，动机阐述充分
- **价值**: ⭐⭐⭐⭐ 首次实现任意人数T2M生成，具有明确的实用价值

<!-- RELATED:START -->

## 相关论文

- [Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](../../CVPR2026/human_understanding/party_part-guidance_for_expressive_text-to-motion_synthesis.md)
- [Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](human_motion_forecasting_in_dynamic_domain_shifts_a_homeostatic_continual_test-t.md)
- [SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](../../CVPR2025/human_understanding/simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](../../CVPR2025/human_understanding/mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)

<!-- RELATED:END -->
