---
title: >-
  [论文解读] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression
description: >-
  [CVPR2026][大视觉语言模型] 提出 CIPHER，一种无需训练的测试时幻觉抑制方法：离线阶段用扩散模型生成反事实图像构建 OHC-25K 数据集，通过 SVD 提取视觉幻觉子空间；推理阶段将隐状态投影到该子空间的正交补空间，在不修改模型参数、不增加推理开销的前提下显著降低 LVLM 的视觉幻觉。
tags:
  - CVPR2026
  - 大视觉语言模型
  - 幻觉抑制
  - 反事实推理
  - 扩散模型
  - 特征投影
  - 训练免调
---

# Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression

## 基本信息

**会议**: CVPR2026  
**arXiv**: [2603.10470](https://arxiv.org/abs/2603.10470)  
**代码**: [项目主页](https://hamidreza-dastmalchi.github.io/cipher-cvpr2026/)  
**领域**: 因果推理 / 多模态幻觉抑制  
**关键词**: 大视觉语言模型, 幻觉抑制, 反事实推理, 扩散模型, 特征投影, 训练免调  

## 一句话总结

提出 CIPHER，一种无需训练的测试时幻觉抑制方法：离线阶段用扩散模型生成反事实图像构建 OHC-25K 数据集，通过 SVD 提取视觉幻觉子空间；推理阶段将隐状态投影到该子空间的正交补空间，在不修改模型参数、不增加推理开销的前提下显著降低 LVLM 的视觉幻觉。

## 研究背景与动机

大视觉语言模型（LVLM）如 LLaVA、MiniGPT-4、mPLUG-Owl2 在多模态任务上表现优异，但频繁产生**幻觉**——生成与视觉输入不一致的描述（不存在的物体、错误的属性等）。幻觉来源可分为两类：

**文本诱导幻觉**：源自 LLM 的自回归生成偏好和语言先验

**视觉诱导幻觉**：源自弱视觉 grounding、模态对齐不足

现有测试时方法（如 Nullu）主要通过扰动文本端来提取幻觉方向，**忽略了视觉模态引发的幻觉**。作者通过线性探测实验发现，文本扰动的幻觉信号在隐表征空间中较弱且不稳定（准确率 0.73–0.80），而扩散模型生成的视觉扰动信号则具有**高度可分性和跨层稳定性**（准确率 0.86–0.89）。这一发现促使作者构建视觉反事实来更精确地定位幻觉方向。

## 方法详解

### 整体框架

CIPHER（Counterfactual Image Perturbations for Hallucination Extraction and Removal）分为两个阶段：

- **离线阶段（Offline Phase）**：构建反事实数据集 OHC-25K → 提取幻觉方向 → 估计幻觉子空间
- **推理阶段（Inference Phase）**：对生成过程中的隐状态做正交投影，抑制幻觉分量

### 关键设计一：反事实数据集生成（OHC-25K）

从 MSCOCO 训练集随机选取 $M=5000$ 对图像-标注 $\{(\boldsymbol{I}_i, \mathcal{C}_i)\}_{i=1}^M$，按以下流水线生成反事实图像：

**Step 1: 标注扰动。** 使用 GPT-3.5 对每条真实标注 $\mathcal{C}_i$ 生成幻觉标注 $\tilde{\mathcal{C}}_i$，注入看似合理但实际不存在的物体描述（如在餐桌场景中添加"一串葡萄"）。

**Step 2: 潜空间编码与正向扩散加噪。** 用 Stable Diffusion v1.5 的 VAE 编码器将图像编码到潜空间，然后施加 $t_h$ 步正向扩散过程引入高斯噪声：

$$\boldsymbol{z}_0 = \mathcal{E}(\boldsymbol{I}_i)$$

$$\tilde{\boldsymbol{z}}_{t_h} = \sqrt{\bar{\alpha}_{t_h}} \boldsymbol{z}_0 + \sqrt{1 - \bar{\alpha}_{t_h}} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_{t_h}$ 是噪声调度系数的累积乘积。选择 $t_h = 0.5T$ 保证图像全局结构保留但语义内容可被修改。

**Step 3: 条件反向去噪。** 以幻觉标注 $\tilde{\mathcal{C}}_i$ 为条件执行反向去噪，将噪声潜变量引导回与幻觉标注对齐的图像：

$$\tilde{\boldsymbol{z}}_{t-1} = f_\theta(\tilde{\boldsymbol{z}}_t, t, \tilde{\mathcal{C}}_i), \quad t = t_h, \dots, 1$$

**Step 4: 解码生成反事实图像。** 解码最终潜向量得到反事实图像 $\tilde{\boldsymbol{I}}_{i,j} = \mathcal{D}(\tilde{\boldsymbol{z}}_0)$，其中 $j=1,\dots,B=5$ 索引不同高斯噪声种子的变体。将反事实图像与**原始真实标注**配对形成语义冲突：

$$\textbf{OHC-25K} = \{(\tilde{\boldsymbol{I}}_{i,j}, \mathcal{C}_i) \mid i=1,\dots,M,\; j=1,\dots,B\}$$

### 关键设计二：幻觉子空间估计

对每对原始 $(\boldsymbol{I}_i, \mathcal{C}_i)$ 及其 $B$ 个反事实变体 $(\tilde{\boldsymbol{I}}_{i,j}, \mathcal{C}_i)$，从冻结的 LVLM 提取中间层隐表征。设 $\boldsymbol{h}_{\ell,k}^{(i)}$ 为第 $\ell$ 层第 $k$ 个标注 token 的隐状态，对所有标注 token 做均值池化：

$$\boldsymbol{h}_\ell^{(i)} = \frac{1}{N}\sum_{k=1}^{N} \boldsymbol{h}_{\ell,k}^{(i)}, \quad \tilde{\boldsymbol{h}}_\ell^{(i)} = \frac{1}{B}\sum_{j=1}^{B} \tilde{\boldsymbol{h}}_\ell^{(i,j)}$$

计算样本 $i$ 在第 $\ell$ 层的**幻觉方向向量**：

$$\boldsymbol{\delta}_\ell^{(i)} = \tilde{\boldsymbol{h}}_\ell^{(i)} - \boldsymbol{h}_\ell^{(i)}$$

将所有样本的幻觉方向堆叠为差异矩阵 $\boldsymbol{\Delta}_\ell \in \mathbb{R}^{M \times d}$，执行奇异值分解：

$$\boldsymbol{\Delta}_\ell = \boldsymbol{U}_\ell \boldsymbol{\Sigma}_\ell \boldsymbol{V}_\ell^\top$$

保留前 $r$ 个右奇异向量 $\boldsymbol{V}_{\ell,r} = [\boldsymbol{v}_{\ell,1}, \dots, \boldsymbol{v}_{\ell,r}]$ 作为该层的**幻觉基向量库（Hallucination Basis Bank）**。这些向量张成的子空间捕获了视觉诱导幻觉的主方向。

### 关键设计三：测试时幻觉消除

推理时，在每个自回归解码步 $k$ 和选定层 $\ell$，将隐状态投影到幻觉子空间的正交补空间：

$$\boldsymbol{h}_{\ell,k}^{\text{clean}} = \boldsymbol{h}_{\ell,k}^{\text{test}} - \sum_{j=1}^{r} \langle \boldsymbol{h}_{\ell,k}^{\text{test}}, \boldsymbol{v}_{\ell,j} \rangle \boldsymbol{v}_{\ell,j}$$

等价地用投影矩阵表达：

$$\boldsymbol{h}_{\ell,k}^{\text{clean}} = \boldsymbol{P}_\ell \boldsymbol{h}_{\ell,k}^{\text{test}}, \quad \boldsymbol{P}_\ell = \boldsymbol{I} - \boldsymbol{V}_{\ell,r} \boldsymbol{V}_{\ell,r}^\top$$

此操作在每个 token 解码前执行，移除与幻觉方向对齐的分量，同时保留核心语义信息。

### 实现细节

- **干预层选择**：在 Transformer 上层（16–32 层）施加投影
- **秩选择**：LLaVA-1.5 $r=8$，MiniGPT-4 $r=64$，mPLUG-Owl2 $r=32$（网格搜索确定）
- **扩散步数**：$t_h = 0.5T$，平衡结构保留与语义替换
- **Classifier-free guidance scale**：7.5
- **解码设置**：beam size=3，CHAIR 最大 64 tokens，OPOPE 最大 256 tokens
- **零额外推理开销**：投影为轻量矩阵运算，吞吐量与 greedy decoding 相同

## 实验

### 主实验一：CHAIR 基准（幻觉率 + 流畅度）

| 方法 | LLaVA CHAIR$_S$↓ | CHAIR$_I$↓ | BLEU↑ | MiniGPT-4 CHAIR$_S$↓ | CHAIR$_I$↓ | BLEU↑ | mPLUG CHAIR$_S$↓ | CHAIR$_I$↓ | BLEU↑ |
|------|-----------------|----------|-------|---------------------|----------|-------|----------------|----------|-------|
| Greedy | 20.40 | 7.08 | 15.72 | 32.40 | 12.20 | 14.57 | 22.90 | 8.62 | 15.01 |
| DoLa | 20.20 | 6.75 | 15.68 | 31.90 | 12.15 | 14.54 | 22.40 | 8.36 | 15.13 |
| OPERA | 17.50 | 6.07 | 16.02 | 29.70 | 11.96 | 14.82 | 20.07 | 7.18 | 15.41 |
| VCD | 20.30 | 7.28 | 14.53 | 29.00 | 12.64 | 14.42 | 22.80 | 8.68 | 15.14 |
| Woodpecker | 23.85 | 7.50 | 17.05 | 28.87 | 10.20 | 15.30 | 26.33 | 8.43 | 16.43 |
| LURE | 19.48 | 6.50 | 15.97 | 27.88 | 10.20 | 15.03 | 21.27 | 7.67 | 15.65 |
| HALC | 16.90 | 5.72 | 16.02 | 25.20 | 9.42 | 14.91 | 18.80 | 7.00 | 15.33 |
| Nullu | 15.20 | 5.30 | 15.69 | 21.40 | 8.99 | 14.81 | 15.60 | 5.77 | 15.45 |
| **CIPHER** | **13.05** | **4.53** | 15.82 | **18.48** | **8.33** | 15.10 | **13.60** | **4.92** | **16.25** |

CIPHER 在所有模型上均取得最低幻觉率。在 LLaVA 上 CHAIR$_S$ 较 Nullu 降低 2.15%，较 Greedy 降低 7.35%；在 MiniGPT-4 上较 Greedy 降低 13.92%。BLEU 分数保持或提升，幻觉抑制未牺牲生成流畅度。

### 主实验二：OPOPE 基准（物体幻觉检测）

| 方法 | LLaVA Acc↑ | Prec↑ | F1↑ | MiniGPT-4 Acc↑ | Prec↑ | F1↑ | mPLUG Acc↑ | Prec↑ | F1↑ |
|------|-----------|-------|-----|----------------|-------|-----|-----------|-------|-----|
| Greedy | 79.14 | 91.98 | 90.45 | 71.22 | 93.72 | 90.04 | 76.46 | 88.85 | 87.29 |
| Nullu | 79.52 | 93.46 | 91.79 | 71.92 | 95.96 | 92.07 | 77.09 | 92.83 | 90.80 |
| **CIPHER** | **80.05** | **93.72** | **92.11** | **72.25** | **96.50** | **92.58** | **77.87** | **92.93** | **90.95** |

在已接近饱和的 OPOPE 基准上仍持续超越所有基线，精确率提升尤为显著。

### 消融实验：幻觉来源分析

| 文本扰动 | 图像扰动 | CHAIR$_S$↓ | CHAIR$_I$↓ | BLEU↑ |
|---------|---------|----------|----------|-------|
| Yes | No | 15.20 | 5.30 | 15.69 |
| **No** | **Yes** | **13.05** | **4.53** | **15.82** |
| Yes | Yes | 15.71 | 5.32 | 15.66 |

仅图像扰动效果最佳；联合扰动反而略差，说明两种幻觉方向可能存在相互干扰。

### 推理效率对比

| 指标 | Greedy | OPERA | HALC | Nullu | **CIPHER** |
|------|--------|-------|------|-------|------------|
| CHAIR$_S$↓ | 20.40 | 17.50 | 16.90 | 15.20 | **13.05** |
| 吞吐量↑ (items/s) | 0.70 | 0.10 | 0.05 | 0.70 | **0.70** |

CIPHER 吞吐量与标准 Greedy 解码完全一致（0.70 items/s），远超 OPERA（7x 加速）和 HALC（14x 加速）。

### 关键发现

- **线性探测证明视觉扰动优于文本扰动**：文本扰动隐表征可分性仅 0.73–0.80（跨层不稳定），视觉扰动达 0.86–0.89（跨层稳定）
- **扩散步数 $t_h=0.5T$ 最优**：过小保留过多原始语义，过大破坏结构一致性
- **子空间秩 $r=8$（LLaVA）最优**：CHAIR 和 BLEU 同时最优
- **视觉噪声鲁棒性强**：$\sigma$ 从 0 到 1 大范围噪声测试中 CIPHER 始终优于基线，高噪声场景优势更显著
- **MMHal-Bench 全类别提升**：属性、环境、整体描述和对抗场景获益最大
- **LLaVA-Bench**：准确度从 6.79 提升至 7.08，详细度从 6.33 提升至 6.75

## 亮点

- **视觉反事实的独到视角**：首次通过"幻觉图像"而非"幻觉文本"来提取幻觉方向，线性探测实验充分证明视觉幻觉信号更强且更稳定
- **零额外推理开销**：仅需单次前向传播 + 轻量矩阵投影，吞吐量与无干预基线完全一致
- **训练免调即插即用**：无需重训模型、无需额外标注、无需修改架构，离线算一次子空间即可
- **跨模型通用性**：在 LLaVA-1.5、MiniGPT-4、mPLUG-Owl2 三个架构上均有一致显著提升
- **数学优雅**：反事实 → 差向量 → SVD → 正交投影，逻辑链条完整、算法简洁可解释

## 局限性

- 离线阶段依赖 Stable Diffusion 和 GPT-3.5 生成反事实数据，一次性构建成本非零
- 幻觉子空间是**静态固定**的，对所有测试输入使用同一投影矩阵，缺乏输入自适应能力
- 秩 $r$ 和干预层需对每个模型做网格搜索，不同模型最优配置差异较大（$r$ 从 8 到 64）
- 联合文本+图像扰动效果反而不如单图像扰动，作者对此现象分析不够深入
- 主要验证物体级别幻觉，细粒度属性/关系幻觉的效果有待进一步探索

## 评分

⭐⭐⭐⭐ — 方法简洁优雅且实验有效，"用幻觉图像对抗幻觉"的反事实构思新颖，四个基准全面验证说服力强。主要不足是静态投影缺乏自适应性，以及联合扰动失效的原因分析欠缺。

<!-- RELATED:START -->

## 相关论文

- [MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)
- [Retrieving Counterfactuals Improves Visual In-Context Learning](retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](../../AAAI2026/causal_inference/causally-grounded_dual-path_attention_intervention_for_objec.md)
- [RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [Action-Guided Attention for Video Action Anticipation](../../ICLR2026/causal_inference/action-guided_attention_for_video_action_anticipation.md)

<!-- RELATED:END -->
