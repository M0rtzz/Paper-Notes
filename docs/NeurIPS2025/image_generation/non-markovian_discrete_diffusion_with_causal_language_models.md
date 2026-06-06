---
title: >-
  [论文解读] Non-Markovian Discrete Diffusion with Causal Language Models
description: >-
  [NeurIPS 2025][图像生成][离散扩散模型] 提出CaDDi框架，通过非马尔可夫离散扩散过程让每步去噪都能访问完整生成轨迹，并将其统一到因果语言模型架构中，使预训练LLM可直接复用为离散扩散模型。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "离散扩散模型"
  - "非马尔可夫"
  - "因果语言模型"
  - "序列生成"
  - "自回归"
---

# Non-Markovian Discrete Diffusion with Causal Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.09767](https://arxiv.org/abs/2502.09767)  
**代码**: [https://github.com/](https://github.com/) (未提供)  
**领域**: 图像/文本生成 (Discrete Diffusion)  
**关键词**: 离散扩散模型, 非马尔可夫, 因果语言模型, 序列生成, 自回归

## 一句话总结

提出CaDDi框架，通过非马尔可夫离散扩散过程让每步去噪都能访问完整生成轨迹，并将其统一到因果语言模型架构中，使预训练LLM可直接复用为离散扩散模型。

## 研究背景与动机

离散扩散模型（如D3PM、SEDD、MDLM等）在结构化序列生成方面展现了灵活、可控的优势，特别是在文本填充、双向生成等场景中具有天然优势。然而，它们在生成质量上仍然落后于自回归语言模型。

**核心痛点**：现有离散扩散模型依赖马尔可夫假设——每一步去噪只能看到当前状态 $\mathbf{x}_t$，而无法利用之前的生成历史。这导致所有信息被压缩到单一状态中，一旦某步预测出错，误差会沿时间步不可逆地累积。此外，去噪分布的独立分解假设 $p_\theta(\mathbf{x}_0|\mathbf{x}_t) = \prod_i p_\theta(\mathbf{x}_0^i|\mathbf{x}_t)$ 使模型无法捕捉token间的依赖关系，进一步限制了自我纠错能力。

**核心矛盾**：自回归LM生成质量高但缺乏双向灵活性；离散扩散模型灵活但质量差。能否统一二者？

**本文切入角度**：将离散扩散视为一般化的层级VAE（HVAE），打破马尔可夫约束，让反向过程 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t:T})$ 可以访问完整的未来轨迹。关键观察是：这种非马尔可夫自回归推理结构天然对齐了因果语言模型的结构——只需增加一个时间维度。因此可以在单一decoder-only Transformer中统一序列维度（token顺序）和时间维度（扩散时间步），并且标准因果LM是其T=1的特例，使得预训练LLM可直接微调为离散扩散模型。

## 方法详解

### 整体框架

CaDDi是一个统一序列（因果）和时间（扩散）建模的非马尔可夫离散扩散框架。输入由一条完整的非马尔可夫前向轨迹 $(\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_0)$ 拼接而成，使用block-wise因果mask的decoder-only Transformer处理。推理时，从纯噪声 $\mathbf{x}_T$ 开始，逐步自回归预测干净数据 $\tilde{\mathbf{x}}_0$，然后通过前向核重新加噪得到 $\mathbf{x}_{t-1}$。

### 关键设计

1. **非马尔可夫前向过程**：不同于标准马尔可夫链中 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 的逐步加噪，本文采用独立加噪：$q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_0)$。各时间步的噪声独立于彼此（给定 $\mathbf{x}_0$），带来根本不同的轨迹结构——不同时间步携带互补信息。前向过程只需边缘corruption核 $q(\mathbf{x}_t|\mathbf{x}_0)$，可复用absorbing或uniform核。

2. **非马尔可夫反向推理**：反向过程 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t:T}) = q(\mathbf{x}_{t-1}|\mathbf{x}_0 = \mu_\theta(\mathbf{x}_{t:T}, t))$ 将对完整历史轨迹的依赖转移到去噪器 $\mu_\theta$ 上。独立corruption核使得后验形式得以大幅简化——不再需要 $\mathbf{x}_{t:T}$ 直接参与后验计算,只需要去噪器预测 $\tilde{\mathbf{x}}_0$，再用前向核采样 $\mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1}|\tilde{\mathbf{x}}_0)$。

3. **2D旋转位置编码（2D RoPE）**：现有因果LM的RoPE只编码序列维度。本文扩展为block-diagonal结构：$\mathbf{R}_t^{(i)} = \text{diag}[\mathbf{R}_{\text{seq}}^{(i)}, \mathbf{R}_{\text{time}}^{(t)}]$，在query/key的不同子空间分别编码token位置和扩散时间步。同一时间步内注意力模式与标准因果LM完全一致，保证向后兼容。

4. **CaDDi-AR变体**：在每个时间步内进一步做token级自回归分解 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t:T}) = \prod_{i} p_\theta(\mathbf{x}_{t-1}^i|\mathbf{x}_{t-1}^{0:i-1}, \mathbf{x}_{t:T})$，历史轨迹作为"prompt"进行自回归生成。当T=1时退化为标准因果LM，因此可直接加载预训练LLM权重微调。

5. **半投机解码（Semi-Speculative Decoding）**：CaDDi-AR的朴素生成需要 $O(L \times T)$ 次前向调用。利用所有时间步共享去噪目标 $\mathbf{x}_0$ 的特性，复用前一时间步的预测 $\tilde{\mathbf{x}}_0^{\text{prev}}$ 作为当前步的draft，模型并行验证所有token，只需从首个被拒绝的位置重新采样，显著加速推理。

### 损失函数 / 训练策略

对absorbing核，ELBO简化为加权交叉熵：$\mathcal{L}_{\text{absorb}} = \mathbb{E} \sum_{t=1}^T [\alpha_{t-1} \mathbf{x}_0^\top \log \mu_\theta(\mathbf{x}_{t:T}, t)]$，权重 $\alpha_{t-1}$ 反映corruption程度。CaDDi-AR则使用next-token prediction loss。实践中采用latent truncation和trajectory re-composition压缩上下文窗口。

## 实验关键数据

### 主实验

**LM1B数据集 - 生成困惑度（PPL，越低越好）**

| 模型 | GPT2 (T=0.5) | Llama-2 (T=0.7) | Llama-3 (T=0.5) |
|------|-------------|-----------------|-----------------|
| UDLM | 328.99 | 111.86 | 231.05 |
| D3PM | 133.38 | 55.54 | 110.86 |
| SEDD | 81.44 | 66.54 | 60.00 |
| MDLM | 106.20 | 62.34 | 104.71 |
| DFM | 106.03 | 38.93 | 102.89 |
| **CaDDi** | **45.96** | **35.40** | **36.79** |
| **CaDDi-AR** | 67.59 | **35.38** | 44.54 |

**Text8数据集 - BPD（64步离散化）**

| 模型 | BPD↓ | PPL↓ | NLL↓ |
|------|------|------|------|
| D3PM | ≤1.51 | ≤2.85 | ≤1.05 |
| SEDD | ≤1.46 | ≤2.75 | ≤1.01 |
| MDLM | ≤1.46 | ≤2.75 | ≤1.01 |
| **CaDDi** | **≤1.41** | **≤2.66** | **≤0.98** |

### 消融实验

**推理鲁棒性测试（注入噪声）**

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| CaDDi (无噪声注入) | 最优PPL | 基线 |
| CaDDi (早期注入噪声) | PPL下降较小 | 非马尔可夫轨迹提供纠错能力 |
| D3PM/MDLM (同等噪声) | PPL大幅下降 | 马尔可夫模型误差累积严重 |

**LLM微调推理任务（CaDDi-AR基于QWen2-1.5B）**

| 模型 | ARC-Chal. | BoolQ | LAMBADA |
|------|-----------|-------|---------|
| QWen2-1.5B | 33.7 | 72.6 | 63.9 |
| CaDDi-AR | **34.2** (+1.9%) | 71.6 | **66.3** (+2.4%) |

### 关键发现

- CaDDi在LM1B上较MDLM降低PPL约57%（低温设置下），显著缩小与自回归LM的差距
- 低温下block-level CaDDi甚至优于token-level CaDDi-AR，因为低温缓解了block生成的长尾问题
- 基于预训练LLM微调的CaDDi-AR在推理任务上超越基座模型，说明非马尔可夫扩散的"回顾和修正"能力有助于推理
- 半投机解码大幅降低CaDDi-AR的推理开销

## 亮点与洞察

- **统一性极强**：将自回归LM和离散扩散统一在同一框架下（T=1即为AR），这使得大量预训练LLM资源可直接用于扩散生成
- **2D RoPE设计精妙**：既编码时间维度又保持与标准RoPE完全兼容，无需任何架构修改
- **非马尔可夫框架的信息论基础扎实**：Proposition 3.1 建立了非马尔可夫与马尔可夫扩散的互信息等价性，为噪声调度提供理论指导
- **半投机解码**思路新颖——利用扩散模型自身前一步的预测作为draft，天然匹配speculative decoding范式

## 局限与展望

- 完整轨迹 $\mathbf{x}_{t:T}$ 会占用大量上下文窗口，虽然有truncation策略但仍受限于上下文长度
- CaDDi-AR虽然质量更高但推理开销大 $O(L \times T)$；半投机解码缓解但未完全解决
- 实验主要在NLP任务上验证，未涉及图像、蛋白质等其他离散序列领域
- 未与Discrete Flow Matching的最新变体做更深入比较

## 相关工作与启发

- 与DART（连续域非马尔可夫扩散）互补：DART处理连续空间，CaDDi处理离散空间
- 2D RoPE的思路可推广到其他需要多维度位置编码的场景（如视频、多模态等）
- 非马尔可夫扩散 + 预训练LLM的组合为可控文本生成（如infilling、conditional generation）开辟新路径

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 非马尔可夫离散扩散将AR和扩散统一的视角非常新颖且优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖LM1B/Text8/推理任务等多个维度，但缺少更多领域的验证
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，动机阐述充分，理论和实验衔接紧密
- 价值: ⭐⭐⭐⭐⭐ 统一框架+预训练LLM复用的价值极高，可能开辟新的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
