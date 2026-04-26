---
title: >-
  [论文解读] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models
description: >-
  [ACL 2025][LLM/NLP][合成数据] DiffLM 提出基于 VAE + 潜在扩散 + 冻结 LLM 解码器的可控数据合成框架，通过在潜在空间引入扩散过程来精确建模真实数据分布，并以 soft prompt 方式将分布信息注入 LLM，在表格、代码和工具三类结构化数据上合成质量超越真实数据 2%-7%。
tags:
  - ACL 2025
  - LLM/NLP
  - 合成数据
  - VAE
  - 扩散模型
  - LLM
  - 结构化数据生成
---

# DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models

**会议**: ACL 2025  
**arXiv**: [2411.03250](https://arxiv.org/abs/2411.03250)  
**代码**: [bytedance/DiffLM](https://github.com/bytedance/DiffLM)  
**领域**: 数据合成、生成模型  
**关键词**: 合成数据、VAE、扩散模型、LLM、结构化数据生成  

## 一句话总结

DiffLM 提出基于 VAE + 潜在扩散 + 冻结 LLM 解码器的可控数据合成框架，通过在潜在空间引入扩散过程来精确建模真实数据分布，并以 soft prompt 方式将分布信息注入 LLM，在表格、代码和工具三类结构化数据上合成质量超越真实数据 2%-7%。

## 研究背景与动机

- **LLM 合成数据的核心挑战**：现有 LLM 数据合成方法面临两大问题——(1) LLM 缺乏对目标数据分布的全局理解，生成数据多样性低且容易出现 data copying；(2) 依赖复杂的 prompt 工程和多智能体框架，难以快速适配新任务。
- **VAE 直接应用的失败**：虽然 VAE 和扩散模型在图像合成中表现出色，但直接从 VAE 学到的潜在分布采样用于文本生成时，产生的文本与目标分布完全无关。原因是编码器后验 $q_\phi(z|x)$ 与先验 $p(z)$ 之间存在显著差异，导致潜在空间存在大量未利用或空白区域。
- **解耦思路**：将数据分布学习与 LLM 生成能力解耦，使 LLM 既保留内部知识又能受真实数据分布引导，实现高质量结构化合成数据生成。

## 方法详解

### 整体框架

DiffLM 包含三个核心组件：(1) 可训练的 Transformer 编码器（VAE 编码器），将文本映射到连续潜在空间；(2) 潜在扩散模块，在 VAE 潜在空间上训练去噪网络以更精确地建模潜在分布；(3) 冻结参数的 LLM 解码器，通过 soft prompt 注入接收潜在特征进行重建/生成。

### 关键设计

1. **VAE 表示学习与分布解耦**：使用可训练编码器将结构化文本 $s_i$ 映射为 $x_i \in \mathbb{R}^{d \times 2}$，拆分为均值 $\mu$ 和标准差 $\sigma$，通过重参数化得到 $z = \mu + \sigma \odot \epsilon$。LLM 解码器参数全部冻结，避免重训练并保留通用知识，将分布学习与生成目标彻底解耦。
2. **潜在空间扩散去噪**：针对 VAE 采样质量差的核心问题，从训练好的 VAE 提取潜在向量 $z_0$，运行前向扩散 $z_t = z_0 + \sigma(t)\epsilon$ 并训练去噪网络学习反向过程恢复 $z_0$。
3. **Soft Prompt 潜在注入**：将潜在表示 $z$ 通过 MLP 投影为 $k$ 个 soft prompt 嵌入 $\mathbf{H}_{\text{latent}} \in \mathbb{R}^{k \times d}$，拼接在 BOS token 前作为引导向量参与 LLM 解码，实现即插即用的分布控制。

### 损失函数

VAE 训练采用 $\beta$-VAE 策略：

$$\text{ELBO}_\beta = L_{rec} - \beta L_{kl}$$

- $L_{rec}$：语言模型重建似然
- $L_{kl} = D_{\text{KL}}(q_\phi(z|x) \| p(z))$：KL 散度正则化
- 采用**递减 $\beta$ 策略**：初始较大 $\beta$ 强约束潜在空间平滑性，重建损失收敛后逐步降低 $\beta$ 提升重建精度
- 扩散模块使用去噪分数匹配损失：$\mathcal{L}_{\text{diff}} = \mathbb{E}\|\epsilon_\theta(z_t, t) - \epsilon\|^2$

## 实验

### 主实验：表格数据生成

| 方法 | Adult MLE↑ | Default MLE↑ | Magic MLE↑ | Shoppers MLE↑ | Beijing RMSE↓ |
|------|-----------|-------------|-----------|-------------|-------------|
| Real Data | 0.927 | 0.770 | 0.946 | 0.926 | 0.423 |
| GReaT (微调GPT-2) | 0.913 | 0.755 | 0.888 | 0.902 | 0.653 |
| GPT-4 ICL | 0.889 | — | 0.864 | 0.835 | 0.992 |
| TabSyn (扩散) | 0.915 | 0.764 | 0.938 | 0.920 | 0.582 |
| **DiffLM** | **0.906** | **0.794** | **0.917** | **0.915** | **0.696** |

代码生成实验中，DiffLM 合成数据继续预训练的 Mistral-7B 在 HumanEval pass@1 达 35.37%，显著超过真实数据版本（28.58%）和 CodeLLaMA-7B（33.50%）。12B 模型上提升更大：42.24% vs 36.97%。

### 消融实验：注入方式与 β 策略

| 配置 | 重建损失 | 下游性能 |
|------|---------|---------|
| 递减 β + Soft Prompt 注入 | 最低 | 最优 |
| 循环 β + Soft Prompt 注入 | 较高 | 次优 |
| 递减 β + Prefix Injection | 中等 | 中等 |
| 递减 β + Cross Attention | 较高 | 较差 |

### 关键发现

- DiffLM 在 Default 数据集上合成数据的下游表现**超过真实数据**（MLE 0.794 vs 0.770），说明合成数据包含了额外知识
- 代码场景中真实数据继续预训练在 MBPP 上性能下降，而 DiffLM 合成数据反而全面提升
- 潜在扩散模块是关键组件：去掉后 VAE 采样文本与目标分布无关
- 工具合成中 DiffLM 单个工具的 GPT-4 评分高于真实数据，约 1/3 类别在多样性上持平或更优
- 递减 $\beta$ 策略显著优于循环退火策略

## 论文亮点

- 首次将 VAE + 潜在扩散 + LLM 三者结合用于高质量结构化数据合成，框架优雅且即插即用
- 通过解耦设计冻结 LLM 参数即可实现分布控制，无需微调大模型
- 在表格/代码/工具三类场景全面验证，部分场景合成数据超越真实数据
- 递减 $\beta$ 策略和 soft prompt 注入经消融验证为最优组合

## 局限性

- 仅验证无条件合成场景，缺乏对条件生成（按类别/属性控制）的探索
- 表格合成在列分布密度指标上未超过 TabSyn 等专用模型
- LLM 解码器冻结可能限制对某些特殊领域数据格式的适配
- 缺少对合成数据的隐私性和公平性等角度的评估

## 相关工作

- **LLM 数据合成**：GReaT (Borisov et al., 2023) 微调 GPT-2 做表格合成；ICL prompting 方法多样性低且对复杂结构数据效果差
- **潜在变量文本生成**：Optimus (Li et al., 2020) 等探索潜在空间与 LM 结合，但在复杂结构化数据上效果有限
- **表格数据生成**：CTGAN、TVAE、TabSyn 等专用模型表现优但不具通用性

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)
- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)

<!-- RELATED:END -->
