---
title: >-
  [论文解读] GenMol: A Drug Discovery Generalist with Discrete Diffusion
description: >-
  [ICML2025][医学图像][离散扩散模型] 提出 GenMol，一个基于掩码离散扩散（Masked Discrete Diffusion）的通用分子生成框架，通过非自回归双向并行解码生成 SAFE 序列，并引入片段重掩码（fragment remasking）和分子上下文引导（MCG），用**单一模型**覆盖从头生成、片段约束生成、目标导向 hit 生成和先导化合物优化四大药物发现场景，全面超越此前最优方法。
tags:
  - ICML2025
  - 医学图像
  - 离散扩散模型
  - SAFE分子表示
  - 片段重掩码
  - 分子优化
  - 药物设计
---

# GenMol: A Drug Discovery Generalist with Discrete Diffusion

**会议**: ICML2025  
**arXiv**: [2501.06158](https://arxiv.org/abs/2501.06158)  
**代码**: [NVIDIA-Digital-Bio/genmol](https://github.com/NVIDIA-Digital-Bio/genmol)  
**领域**: 药物发现 / 分子生成  
**关键词**: 离散扩散模型, SAFE分子表示, 片段重掩码, 分子优化, 药物设计

## 一句话总结
提出 GenMol，一个基于掩码离散扩散（Masked Discrete Diffusion）的通用分子生成框架，通过非自回归双向并行解码生成 SAFE 序列，并引入片段重掩码（fragment remasking）和分子上下文引导（MCG），用**单一模型**覆盖从头生成、片段约束生成、目标导向 hit 生成和先导化合物优化四大药物发现场景，全面超越此前最优方法。

## 研究背景与动机
药物发现涉及多个阶段：从头（de novo）分子生成、片段约束生成（linker design、scaffold morphing 等）、目标导向 hit 生成以及先导化合物（lead）优化。现有分子生成模型通常只覆盖其中一两个场景，无法作为通用工具贯穿整个流程。

此前最具代表性的通用方法是 **SAFE-GPT**，它将分子表示为 SAFE（Sequential Attachment-based Fragment Embedding）序列，利用 GPT 自回归解码完成多种任务。然而 SAFE-GPT 存在三个主要缺陷：

**token 顺序依赖**：SAFE 本身是片段顺序无关的，但 GPT 的自左向右解码与此矛盾

**效率低**：自回归逐 token 生成，无法并行解码

**引导困难**：自回归模型难以在生成过程中引入全局引导；目标导向生成需要额外的强化学习微调

GenMol 的核心动机是：**用离散扩散替代自回归**，在保持 SAFE 表示优势的同时解决上述三个痛点。

## 方法详解

### 整体架构
GenMol 采用 **BERT 架构**作为去噪网络，训练框架基于 **MDLM**（Masked Discrete Language Model）。输入是 SAFE 分子序列，前向过程逐步将 token 替换为 [MASK]，反向过程通过双向注意力并行预测被掩码的 token。

### 前向掩码过程
对序列中每个 token $\boldsymbol{x}^l$ 独立插值：

$$q(\boldsymbol{z}_t^l | \boldsymbol{x}^l) = \text{Cat}(\boldsymbol{z}_t^l;\; \alpha_t \boldsymbol{x}^l + (1-\alpha_t)\mathbf{m})$$

其中 $\alpha_t$ 是单调递减的掩码率调度函数，$t=0$ 时全部未掩码，$t=1$ 时全部掩码。

### 反向解码过程
未被掩码的 token 保持不变；对被掩码位置，模型预测去噪分布：

$$p_\theta(\boldsymbol{z}_s^l | \boldsymbol{z}_t^l = \mathbf{m}) = \text{Cat}\!\left(\boldsymbol{z}_s^l;\; \frac{(1-\alpha_s)\mathbf{m} + (\alpha_s - \alpha_t)\boldsymbol{x}_\theta^l(\boldsymbol{z}_t, t)}{1-\alpha_t}\right)$$

### 训练损失
NELBO 损失，本质是不同时间步上 MLM（交叉熵）损失的加权平均：

$$\mathcal{L}_{\text{NELBO}} = \mathbb{E}_q \int_0^1 \frac{\alpha_t'}{1-\alpha_t} \sum_l \log \langle \boldsymbol{x}_\theta^l(\boldsymbol{z}_t, t),\; \boldsymbol{x}^l \rangle \, dt$$

### 置信度采样
在每步解码中，模型对所有被掩码位置并行预测，取 top-$N$ 最高置信度的 token 揭示，通过 softmax 温度 $\tau$ 和随机度 $r$ 控制质量-多样性权衡。

### 片段重掩码（Fragment Remasking）
这是 GenMol 进行**目标导向分子优化**的核心策略，三步循环：

1. **片段评分**：将分子集合分解为片段词表，每个片段的分数为包含该片段的分子的目标属性均值：$y(\boldsymbol{f}_k) = \frac{1}{|\mathcal{S}(\boldsymbol{f}_k)|} \sum_{\boldsymbol{x} \in \mathcal{S}(\boldsymbol{f}_k)} y(\boldsymbol{x})$
2. **片段拼接**：从词表中随机选两个高分片段拼接成初始分子
3. **片段重掩码**：随机选初始分子的一个片段，替换为 [MASK] 序列，由 GenMol 重新生成新片段

这一过程可解释为**片段级吉布斯采样**——在给定分子的邻域中随机游走，动态更新片段词表，实现超越初始词表的化学空间探索。

### 分子上下文引导（MCG）
受 autoguidance 启发，MCG 在 logit 空间插值"好输入"与"差输入"的预测：

$$\log \boldsymbol{x}_{\theta,i}^{(w),l} := w \log \boldsymbol{x}_{\theta,i}^l(\boldsymbol{z}_t, t) + (1-w) \log \boldsymbol{x}_{\theta,i}^l(\tilde{\boldsymbol{z}}_t, t)$$

其中 $\tilde{\boldsymbol{z}}_t$ 是对 $\boldsymbol{z}_t$ 额外掩码 $\gamma \cdot 100\%$ token 后得到的退化输入，$w>1$ 为引导强度。这使 GenMol 在片段约束生成和目标导向生成中更好地利用分子上下文信息。

## 实验关键数据

### De Novo 生成

| 方法 | Validity(%) | Uniqueness(%) | Quality(%) | Diversity |
|------|------------|--------------|-----------|-----------|
| SAFE-GPT | 94.0 | 100.0 | 54.7 | 0.879 |
| GenMol (N=1, τ=0.5, r=0.5) | **100.0** | 99.7 | **84.6** | 0.818 |
| GenMol (N=3, τ=0.5, r=0.5) | 95.6 | 99.0 | 67.1 | 0.861 |

**关键发现**：GenMol Quality 从 SAFE-GPT 的 54.7% 提升到 84.6%（+30pp），同时 Validity 达到 100%。N=3 时采样速度比 SAFE-GPT 快 2.5×。

### 片段约束生成（平均 Quality）

| 方法 | Linker | Scaffold Morphing | Motif Extension | Scaffold Decoration | Superstructure |
|------|--------|-------------------|-----------------|--------------------|----|
| SAFE-GPT | 21.7 | 16.7 | 18.6 | 10.0 | 14.3 |
| GenMol | **21.9** | — | **30.1** | **31.8** | **34.8** |

GenMol 在 5 个子任务上全面超越 SAFE-GPT。

### 目标导向 Hit 生成（PMO benchmark, 23 个任务）

| 方法 | Sum AUC Top-10 |
|------|---------------|
| **GenMol** | **18.362** |
| f-RAG | 16.928 |
| Genetic GFN | 16.213 |
| Mol GA | 14.708 |
| REINVENT | 14.196 |

GenMol 在 23 个任务中的 **19 个取得最佳成绩**，总分 18.362 大幅领先第二名 f-RAG（+1.434）。

### 先导化合物优化
在 5 个靶蛋白×3 个种子分子×2 个相似度阈值 = 30 个任务中，GenMol 在 **26/30** 个任务成功优化（baseline 在 δ=0.6 时大量失败），验证了片段重掩码策略在化学空间探索中的有效性。

## 亮点与洞察

1. **统一框架**：单一模型、单一 checkpoint 覆盖四大药物发现场景，无需针对不同任务微调
2. **片段重掩码 = 片段级吉布斯采样**：将离散扩散的 remasking 与化学直觉（片段是功能单元）结合，优于 token 级 remasking
3. **MCG 引导无需额外训练**：直接通过退化输入对比即可引导生成，不需要条件训练或 RL 微调
4. **质量-多样性帕累托前沿**：通过 (τ, r) 参数连续调节生成策略，用户可根据需求灵活平衡
5. **非自回归并行解码**：天然适配 SAFE 的片段顺序无关性，同时带来采样加速

## 局限与展望

1. **仅限 2D 分子图**：GenMol 生成 SAFE 字符串（2D），不直接生成 3D 构象，对需要 3D 结构的对接任务需后处理
2. **对接分数作为 oracle**：lead optimization 使用 docking score 评估结合亲和力，实际场景中可能需要更精确的评估
3. **片段分解依赖 BRICS 规则**：预定义的分解规则可能遗漏某些化学有意义的子结构
4. **缺乏蛋白-配体联合建模**：目前不考虑靶蛋白的 3D 口袋信息，限制了结构导向的药物设计
5. **MCG 引导的超参数**：引导强度 w 和额外掩码比例 γ 的选取需要任务相关的调参

## 相关工作与启发

- **SAFE-GPT** (Noutahi et al., 2024)：同样基于 SAFE 表示的自回归模型，GenMol 的直接前身，GenMol 用离散扩散替代 GPT
- **MDLM** (Sahoo et al., 2024)：掩码离散扩散的训练框架，GenMol 直接采用其损失函数
- **f-RAG** (Lee et al., 2024a)：片段级检索增强生成，GenMol 的片段评分公式沿用自此工作
- **Mol GA** (Tripp & Hernández-Lobato, 2023)：基于遗传算法的分子优化，片段重掩码可视为其片段级突变的扩散版本
- **Autoguidance** (Karras et al., 2024)：MCG 的理论基础，GenMol 将其从连续扩散推广到掩码离散扩散

## 评分
- 新颖性: ⭐⭐⭐⭐ — 离散扩散+SAFE+片段重掩码的组合新颖，MCG 是首次在掩码离散扩散中引入 autoguidance
- 实验充分度: ⭐⭐⭐⭐⭐ — 四大任务、23+30 个子任务、多个 baseline 对比，消融全面
- 写作质量: ⭐⭐⭐⭐ — 框架清晰，图示直观，数学推导完整
- 价值: ⭐⭐⭐⭐⭐ — 统一框架在所有任务上 SOTA，具有很强的实用价值和工业部署潜力
