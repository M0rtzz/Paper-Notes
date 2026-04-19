---
title: >-
  [论文解读] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery
description: >-
  [CVPR2026][3D视觉][Human Mesh Recovery] 提出基于VLM的双记忆自反思评判代理（Critique Agent）为扩散式人体网格恢复生成组级偏好信号，再通过组偏好对齐（Group Preference Alignment）微调扩散模型，无需3D标注即可大幅提升野外场景下的HMR精度。
tags:
  - CVPR2026
  - 3D视觉
  - Human Mesh Recovery
  - 扩散模型
  - VLM
  - GRPO
  - Preference Alignment
  - Critique Agent
---

# VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery

**会议**: CVPR2026  
**arXiv**: [2602.19180](https://arxiv.org/abs/2602.19180)  
**机构**: Nanyang Technological University, HKUST(GZ), SenseTime Research, A*STAR
**代码**: 待确认  
**领域**: 3d_vision  
**关键词**: Human Mesh Recovery, diffusion model, VLM, GRPO, Preference Alignment, Critique Agent

## 一句话总结

提出基于VLM的双记忆自反思评判代理（Critique Agent）为扩散式人体网格恢复生成组级偏好信号，再通过组偏好对齐（Group Preference Alignment）微调扩散模型，无需3D标注即可大幅提升野外场景下的HMR精度。

## 背景与动机

单目人体网格恢复（HMR）是一个本质不适定问题：同一张2D图像可对应多种3D姿态。现有方法分为三类：

- **优化方法**（如SMPLify）：迭代优化但易陷入局部最优
- **回归方法**（如HMR、HybrIK）：直接预测单一结果，无法处理深度/遮挡模糊
- **概率方法**（如ScoreHypo、ADHMR）：生成多假设但常牺牲精度

扩散式HMR方法虽能生成多样假设，但存在两个关键缺陷：

1. **预测与输入不一致**：生成的3D网格常与2D图像证据偏离，尤其在遮挡和复杂场景下
2. **DPO指导不可靠**：ADHMR使用的HMR-Scorer仅基于2D关节特征打分，容易被轮廓匹配但物理不合理的姿态欺骗；且DPO仅做成对比较，忽略了组内多个预测之间的质量关系

## 核心问题

如何为扩散式HMR提供高质量的偏好监督信号，使模型在无3D真值的野外数据上也能学习生成物理合理且与图像一致的人体网格？

## 方法详解

### 整体框架

框架包含三个核心组件：

1. **VLM引导的HMR评判代理**（Sec 3.3）：为预测网格生成语义感知的质量分数
2. **HMR组偏好数据集构建**（Sec 3.4）：利用评判代理自动标注组级偏好
3. **组偏好对齐训练**（Sec 3.5）：将GRPO思想引入扩散模型微调

### 3.3 VLM引导的HMR评判代理

#### 核心思路

不同于传统回归打分器（从2D关节数据预测分数），本文的评判代理直接从渲染的叠加图像出发，模拟人类专家判断。给定RGB图像 $I$ 和 $n$ 个网格预测叠加图 $\{\hat{I}_j\}_{j=1}^n$，代理输出每个叠加图的分数 $s_j \in [0, 100]$ 和一句话评语 $c_j$。

使用 **Qwen3-VL-32B** 作为VLM骨干。

#### 3.3.1 双记忆机制

设计两种互补的记忆存储：

| 记忆类型 | 存储内容 | 数据结构 | 作用 |
|---------|---------|---------|------|
| **规则记忆** $\mathcal{M}_R$ | 评估规则文本 | $(t_i, T_i, N_i^u, N_i^s)$：规则文本、语义标签、使用次数、成功次数 | 提供通用评判准则 |
| **原型记忆** $\mathcal{M}_P$ | 已评判的典型案例 | $(v_i, r_i, T_i)$：CLIP视觉嵌入、评判理由含分数、语义标签 | 提供相似案例参照 |

**双记忆增强打分流程**（三步）：

**Step 1 — 原型检索**：用查询图像的CLIP嵌入 $v_q$ 从 $\mathcal{M}_P$ 中检索余弦相似度最高的 top-$K$ 个历史案例作为参照。

**Step 2 — 规则检索**：通过混合得分 $\Psi_i$ 选择最有效的评判规则：

$$\Psi_i = \mathrm{R}(T_q, T_i) + \mathrm{U}_i$$

其中语义相关性 $\mathrm{R}(T_q, T_i) = |T_q \cap T_i|$ 奖励与查询标签匹配的规则。UCB探索得分为：

$$\mathrm{U}_i = \rho_i + C\sqrt{\frac{\log N_{\text{total}}}{N_i^u + 1}}$$

$\rho_i = N_i^s / N_i^u$ 是历史成功率，$C$ 为探索常数。该设计平衡了高成功率规则的利用与低频规则的探索。

**Step 3 — 上下文化打分**：将检索到的规则和原型理由动态组装为提示词，送入VLM生成最终分数和评语。

#### 3.3.2 反思式知识构建

直接提示VLM打分会产生不稳定、不一致的结果。因此引入**探索阶段**让代理自主构建领域知识：

1. **双记忆增强打分**：对一批数据执行打分，增加已使用规则的 $N_i^u$
2. **原型回写**：将典型案例存入 $\mathcal{M}_P$
3. **规则更新**：将代理的分数排名与GT指标通过Spearman秩相关进行比较。若相关性 > 阈值 $\tau$，则相应规则的 $N_i^s$ 递增
4. **新规则挖掘（核心）**：指示VLM检查自身输出与GT指标的差异，提出1-2条新的可测试规则，加入 $\mathcal{M}_R$

**评估阶段**：冻结记忆和学习循环，仅执行双记忆增强打分，保证一致性。

### 3.4 HMR组偏好数据集构建

**数据集构建**分两步：

1. **组生成**：对每张训练图像 $I$，用冻结的预训练扩散参考模型 $\epsilon_{\text{ref}}$ 以不同初始噪声采样 $G$ 次，得到一组多样化的人体网格预测 $\{\mathbf{m}^i\}_{i=1}^G$
2. **组级打分**：将图像 $I$ 和所有 $G$ 个预测（渲染为2D叠加图）同时送入评判代理，获得组内一致的相对质量分数：

$$\{s^1, \ldots, s^G\} = \mathcal{C}_{\text{VLM}}(I, \mathbf{m}^1, \ldots, \mathbf{m}^G)$$

最终数据集 $\mathcal{G}_{\text{HMR}} = \{(I, (\mathbf{m}^1, s^1), \ldots, (\mathbf{m}^G, s^G))\}$。该过程完全自动化，无需人工标注。

### 3.5 组偏好对齐训练

#### 从GRPO到扩散模型

GRPO原本用于LLM随机解码对齐，但扩散模型通常使用确定性ODE采样器（如DDIM）。直接用SDE采样引入随机性需沿整条扩散轨迹训练，计算成本高且输出质量下降。

本文的关键创新：**保持ODE采样效率，只提取GRPO的组级偏好信号**。

#### 训练目标推导

**Step 1 — 计算组内优势**：对预偏好数据集中的分数 $\{s^i\}_{i=1}^G$ 计算标准化优势：

$$A_i = \frac{s_i - \text{mean}(\{s_i\}_{i=1}^G)}{\text{std}(\{s_i\}_{i=1}^G)}$$

**Step 2 — 优势加权对数似然比**：将扩散采样器视为条件策略 $p_\theta(\mathbf{m} | \mathbf{c})$，优化目标为：

$$\mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{c}, \{\mathbf{m}^i\}} \left[\sum_{i=1}^G A(\mathbf{m}^i) \log \frac{p_\theta(\mathbf{m}^i | \mathbf{c})}{p_{\text{ref}}(\mathbf{m}^i | \mathbf{c})}\right]$$

**Step 3 — 扩散代理损失**：利用Diffusion-DPO的重参数化，将对数似然比转化为噪声预测损失之差：

$$\log \frac{p_\theta(\mathbf{m}^i | \mathbf{c})}{p_{\text{ref}}(\mathbf{m}^i | \mathbf{c})} \approx T\lambda_t \mathbb{E}_{t, \epsilon}[L_{\text{DM}}^{\text{ref}}(\mathbf{x}_t^i, \epsilon) - L_{\text{DM}}^\theta(\mathbf{x}_t^i, \epsilon)]$$

**最终训练损失**：

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{m} \sim \mathcal{G}_{\text{HMR}}, t, \epsilon} \; \beta T \lambda_t \sum_{i=1}^G \left[A(\mathbf{m}^i)(L_{\text{DM}}^\theta(\mathbf{x}_t^i, \epsilon) - L_{\text{DM}}^{\text{ref}}(\mathbf{x}_t^i, \epsilon))\right]$$

**直觉解释**：高分网格（正优势）被鼓励获得比参考模型更小的去噪损失；低分网格（负优势）则被推向相反方向。整个过程不需要3D真值标注。

## 实验关键数据

### 主实验结果（Tab.1 节选）

| 方法 | 类型 | M | 3DPW MPJPE↓ | 3DPW PA-MPJPE↓ | H36M MPJPE↓ | H36M PA-MPJPE↓ |
|------|------|---|------------|----------------|------------|----------------|
| ScoreHypo | 概率 | 100 | 63.0 | 37.6 | 38.4 | 26.0 |
| ADHMR | 概率 | 100 | 57.2 | 33.5 | 36.9 | 24.8 |
| **Ours** | 概率 | 100 | **52.5** | **31.5** | **35.0** | **23.9** |
| **Ours†** | 概率 | 100 | **49.9** | **31.9** | **34.3** | **23.5** |

- Ours vs ADHMR（M=100）：3DPW MPJPE降低 8.2%（57.2→52.5）
- Ours†额外使用InstaVariety野外数据（仅用偏好信号，无3D标签），3DPW MPJPE进一步降至49.9

### 消融实验（Tab.2）

| 配置 | 3DPW PVE↓ | MPJPE↓ | PA-MPJPE↓ |
|-----|-----------|--------|-----------|
| Base扩散模型 | 73.4 | 63.0 | 37.6 |
| + 监督微调 | 70.2 | 61.3 | 36.5 |
| DPO + Critique Agent | 63.9 | 53.1 | 33.4 |
| Ours w/o Critique Agent（HMR-Scorer） | 65.4 | 54.9 | 34.7 |
| **Ours（完整）** | **59.5** | **49.9** | **31.9** |

- 组偏好对齐 vs DPO：MPJPE降低6.0%（53.1→49.9），说明组级信号优于成对比较
- 去掉Critique Agent用HMR-Scorer：性能明显下降，验证高质量偏好信号的重要性
- 监督微调在噪声伪标签上改善有限

### 评判代理评估

去掉自反思机制（w/o self-reflection）导致所有指标最大幅度下降，证明自反思知识构建是代理排名稳定性的关键。

## 亮点

1. **首个VLM评判代理用于HMR**：双记忆（规则+原型）+ 自反思机制，比传统2D关节打分器有更强的3D感知能力，能识别自穿透、深度关系错误等
2. **GRPO到扩散模型的优雅迁移**：不需要SDE采样引入随机性，保持ODE效率的同时提取组级偏好信号，损失函数推导简洁直观
3. **无需3D真值的野外微调**：仅靠评判代理的相对偏好信号即可在InstaVariety等野外数据上有效微调，突破了HMR依赖高质量3D标注的瓶颈
4. **UCB探索策略**：规则检索借鉴多臂老虎机的UCB策略，自动平衡已验证规则的利用与新规则的探索

## 局限与展望

1. **VLM推理成本**：使用Qwen3-VL-32B作为评判代理，构建偏好数据集时推理成本较高，限制了大规模应用
2. **评判代理的探索阶段依赖GT**：规则学习和验证仍需合成/实验室数据的3D真值，评判能力可能受探索数据分布影响
3. **组大小的影响**：训练时G=20，更大的组是否带来更好的偏好信号未充分探讨
4. **仅支持单人**：框架基于SMPL单人模型，未涉及多人场景的扩展

## 与相关工作的对比

- **vs ADHMR**：ADHMR用DPO+HMR-Scorer做成对偏好学习，打分器基于2D关节特征易受遮挡误导；本文用VLM评判代理提供更可靠的3D感知分数，组偏好对齐优于成对DPO
- **vs ScoreHypo**：ScoreHypo用辅助选择网络挑最优假设，但不改善生成分布；本文直接优化扩散模型的采样策略
- **vs GRPO扩散方法**（DAPO、D-GRPO）：它们通过SDE采样引入随机性，需沿整条轨迹训练；本文采用离线GRPO+ODE采样，更高效

## 启发与关联

- 双记忆+自反思的VLM评判代理是一个通用范式，可迁移到其他需要自动质量评估的3D任务（如手部重建、场景重建）
- 组偏好对齐框架不依赖具体打分器，理论上可与任何质量评估方法结合
- 利用VLM的3D语义先验做评判，是LLM-as-a-Judge在视觉3D任务中的首次成功应用

## 评分

- 新颖性: ⭐⭐⭐⭐ — VLM评判代理+组偏好对齐双创新，GRPO到扩散的离线迁移设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — 多基准对比+详细消融+定性分析+评判代理独立评估
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ — 无3D标注微调的能力对实际应用有重要意义

<!-- RELATED:START -->

## 相关论文

- [OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)
- [Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)
- [MEGA: Masked Generative Autoencoder for Human Mesh Recovery](../../CVPR2025/3d_vision/mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](../../CVPR2025/3d_vision/heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)

<!-- RELATED:END -->
