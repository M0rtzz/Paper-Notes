---
title: >-
  [论文解读] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models
description: >-
  [ICCV2025][多模态][知识蒸馏] 提出 LLaVA-KD 框架，通过多模态蒸馏（MDist）和关系蒸馏（RDist）两种策略，结合三阶段训练方案（DPT-SFT-DFT），将大规模 MLLM 的知识高效迁移至小规模 MLLM，在不修改模型架构的前提下显著提升小模型性能。
tags:
  - ICCV2025
  - 多模态
  - 知识蒸馏
  - 多模态大模型
  - 小模型训练
  - 视觉-语言对齐
---

# LLaVA-KD: A Framework of Distilling Multimodal Large Language Models

**会议**: ICCV2025  
**arXiv**: [2410.16236](https://arxiv.org/abs/2410.16236)  
**代码**: [GitHub](https://github.com/Fantasyele/LLaVA-KD)  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 多模态大模型, 小模型压缩, 视觉语言对齐, 关系蒸馏  

## 一句话总结

提出 LLaVA-KD 框架，通过多模态蒸馏(MDist)和关系蒸馏(RDist)策略配合三阶段训练方案(DPT-SFT-DFT)，将大规模 MLLM 的知识迁移到小规模 MLLM，在不修改模型架构的前提下显著提升小模型性能。

## 研究背景与动机

多模态大语言模型(MLLM)在视觉-语言理解方面取得了巨大成功，但模型规模和计算复杂度的增加限制了其在资源受限场景的部署。现有的轻量化方案主要有两条路径：

**直接使用小型 LLM 骨干**（如 LLaVA-Phi、TinyLLaVA），但遵循大模型的两阶段训练范式(PT+SFT)会导致性能显著下降。例如 4B TinyLLaVA 能达到 65.0%，但缩到 0.5B 后性能骤降至 54.7%。

**优化模型结构或数据质量**（如 MoE-LLaVA 引入专家混合、Bunny 做数据清洗），但这些方法需要架构修改或数据工程。

作者观察到两个关键问题：
- 现有 LLM 蒸馏方法**仅关注文本模态**的知识迁移，忽略了视觉模态对跨模态理解的重要性
- 直接在 SFT 阶段引入蒸馏效果有限，训练范式本身需要重新设计

## 方法详解

### 整体框架

LLaVA-KD 包含两个核心设计：**MLLM 导向的蒸馏策略**和**三阶段训练方案**。教师和学生模型均采用 LLaVA-1.5 架构（视觉编码器 SigLIP + MLP 投影器 + LLM），共享同一视觉编码器。

### 蒸馏策略

#### 多模态蒸馏 (MDist)

MDist 同时蒸馏**响应 token** 和**视觉 token** 的输出分布。

**响应蒸馏**：在响应 token 上使用标准 KL 散度对齐教师和学生的输出分布：

$$\mathcal{L}_{res} = \sum_{m=1}^{M} \text{KLD}(\phi_l(y_m|\mathbf{y}_{<m}), \phi_s(y_m|\mathbf{y}_{<m}))$$

**视觉蒸馏**：进一步对齐视觉 token 位置的输出分布：

$$\mathcal{L}_{vis} = \sum_{k=1}^{K} \sum_{j=1}^{V} \phi_l(Y_j|\mathbf{y}_{<k}) \log\frac{\phi_l(Y_j|\mathbf{y}_{<k})}{\phi_s(Y_j|\mathbf{y}_{<k})}$$

其中 $K$ 为视觉 token 长度，$V$ 为词汇表大小。这一设计的核心直觉是：视觉表征对 LLM 的多模态理解同样关键，仅蒸馏文本输出是不够的。

#### 关系蒸馏 (RDist)

RDist 通过迁移视觉 token 之间的结构关系来增强学生模型的细粒度视觉理解。具体地，构建视觉 token 的自相关矩阵：

$$R_v^s = \mathbf{y}_v^s \otimes \mathbf{y}_v^s \in \mathbb{R}^{N_p \times N_p}, \quad R_v^t = \mathbf{y}_v^t \otimes \mathbf{y}_v^t \in \mathbb{R}^{N_p \times N_p}$$

然后最大化教师和学生自相关矩阵的余弦相似度：

$$\mathcal{L}_{rel} = 1 - \text{Cos}(R_v^s, R_v^t){= 1 - \frac{R_v^s \cdot R_v^t}{\|R_v^s\| \|R_v^t\|}}$$

这种关系蒸馏类似于经典视觉任务中的特征关系迁移思想，但应用于 MLLM 中 LLM 生成的视觉 token，捕获视觉 token 间的空间和语义依赖关系。

### 三阶段训练方案

#### Stage 1: 蒸馏预训练 (DPT)
- 冻结视觉编码器和 s-LLM，仅优化投影器
- 引入 MDist + RDist 增强视觉-文本对齐
- $\mathcal{L}_{DPT} = \mathcal{L}_{reg} + \alpha\mathcal{L}_{res} + \beta\mathcal{L}_{vis} + \gamma\mathcal{L}_{rel}$

#### Stage 2: 监督微调 (SFT)
- 标准 SFT 过程，冻结视觉编码器，联合优化投影器和 s-LLM
- 让模型习得基础的多模态理解和指令跟随能力

#### Stage 3: 蒸馏微调 (DFT)
- 再次引入 MDist + RDist 进行精细化知识迁移
- $\mathcal{L}_{DFT} = \mathcal{L}_{reg} + \alpha'\mathcal{L}_{res} + \beta'\mathcal{L}_{vis} + \gamma'\mathcal{L}_{rel}$

关键设计动机：先通过 SFT 建立基线理解能力，再通过 DFT 精炼知识，这种分阶段策略比直接在 SFT 中加入蒸馏效果更好。

## 实验

### 主实验结果

| 方法 | LLM | VQAv2 | GQA | SciQA | TextVQA | MME | MMB | POPE | Avg₁₀ |
|------|------|-------|-----|-------|---------|-----|-----|------|-------|
| TinyLLaVA (0.5B) | Qwen1.5-0.5B | 73.9 | 57.4 | 60.9 | 47.4 | 59.8 | 55.0 | 83.7 | 54.7 |
| LLaVA-MOD | Qwen1.5-0.5B | - | 56.2 | 62.8 | 53.9 | 65.3 | 58.8 | - | 54.1 |
| **LLaVA-KD** | Qwen1.5-0.5B | **77.0** | **59.6** | 60.6 | 49.9 | **64.5** | **60.1** | **85.9** | **57.9** |
| TinyLLaVA (1.8B) | Qwen1.5-1.8B | 73.1 | 55.5 | 65.3 | 47.7 | 61.2 | 57.1 | 83.4 | 56.8 |
| LLaVA-MOD | Qwen1.5-1.8B | - | 58.7 | 68.0 | 58.5 | 66.7 | 66.3 | 87.0 | 59.9 |
| **LLaVA-KD** | Qwen1.5-1.8B | **79.0** | **62.3** | 64.7 | 53.4 | 69.1 | 64.0 | **86.3** | **62.1** |

### 消融实验

| 训练方案 | Avg₁₀ |
|---------|-------|
| PT-SFT (baseline) | 54.7 |
| DPT-SFT | 55.6 (+0.9) |
| PT-DFT | 55.8 |
| DPT-DFT | 55.9 |
| PT-SFT-DFT | 56.6 |
| **DPT-SFT-DFT** | **57.9 (+3.2)** |

- DPT 贡献 +0.9%，验证蒸馏预训练增强跨模态对齐的价值
- DFT 贡献 +2.3%（55.6→57.9），是最关键的阶段
- 去掉 SFT（DPT-DFT）性能降至 55.9%，证明 SFT 阶段不可或缺
- MDist 中视觉蒸馏 $\mathcal{L}_{vis}$ 比响应蒸馏 $\mathcal{L}_{res}$ 在 DFT 阶段贡献更大

## 亮点与洞察

1. **视觉模态蒸馏的重要性**：首次在 MLLM 蒸馏中同时优化视觉和语言两个模态的输出分布，填补了现有 LLM 蒸馏仅关注文本模态的空白
2. **三阶段训练的精巧设计**：DPT 打基础→SFT 建能力→DFT 精炼知识的递进式设计优于任何两阶段组合
3. **架构无关性**：不修改模型架构即可持续提升性能，可与其他优化策略（MoE、数据工程）正交使用
4. **数据效率**：仅用 1.2M 训练数据即超越使用 5M 数据的 LLaVA-MOD

## 局限性

- 蒸馏需要先训练好大模型作为教师，增加了整体训练开销（约 120 GPU 小时）
- 教师和学生必须共享相同的视觉编码器和词汇表，限制了跨架构蒸馏的灵活性
- 仅在 LLaVA 系列架构上验证，对其他 MLLM 架构的泛化性未知

## 相关工作

- **轻量级 MLLM**：LLaVA-Phi, TinyLLaVA, MoE-LLaVA, Bunny
- **LLM 蒸馏**：MiniLLM (反向 KLD), DistiLLM (偏斜 KLD)
- **MLLM 蒸馏**：LLaVA-MoD (KLD + 偏好蒸馏), LLaVADI

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术质量 | 4 |
| 实验充分度 | 4 |
| 写作清晰度 | 4 |
| 实用价值 | 5 |
| 总评 | 4.2 |
# LLaVA-KD: A Framework of Distilling Multimodal Large Language Models

**会议**: ICCV2025  
**arXiv**: [2410.16236](https://arxiv.org/abs/2410.16236)  
**代码**: [GitHub](https://github.com/Fantasyele/LLaVA-KD)  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 多模态大模型, 小模型训练, 视觉-语言对齐  

## 一句话总结

提出 LLaVA-KD 框架，通过多模态蒸馏（MDist）和关系蒸馏（RDist）两种策略，结合三阶段训练方案（DPT-SFT-DFT），将大规模 MLLM 的知识高效迁移至小规模 MLLM，在不修改模型架构的前提下显著提升小模型性能。

## 研究背景与动机

多模态大语言模型（MLLM）在统一理解视觉和语言信息方面取得了巨大成功，但模型规模的持续增长限制了其在资源受限场景下的部署。现有的小规模 MLLM（s-MLLM）通常采用轻量级 LLM 骨干网络来降低计算成本，但直接沿用大模型的两阶段训练范式（PT → SFT）会导致显著的性能下降。例如，4B 的 TinyLLaVA 可达 65.0%，但降至 0.5B 后性能骤降至 54.7%。

已有工作尝试通过以下方式弥补：
- **模型结构优化**：MoE-LLaVA 引入混合专家结构
- **训练数据优化**：Bunny 通过聚类+剪枝提高数据质量

但这些方法要么引入额外参数，要么增大数据成本。作者认为，**训练范式优化**是一条被忽视但极具潜力的路径。现有 LLM 蒸馏方法仅关注文本模态的知识迁移，忽略了视觉模态在多模态理解中的关键作用，且直接在 SFT 阶段引入蒸馏收益有限。

## 方法详解

### 整体框架

LLaVA-KD 包含一个大规模教师模型（l-MLLM）和一个小规模学生模型（s-MLLM），两者均采用 LLaVA-1.5 架构（Visual Encoder + Projector + LLM）。教师和学生共享相同的视觉编码器（SigLIP-B/14@384px），通过两层 MLP 投影器将视觉特征 $Z_v \in \mathbb{R}^{N_p \times C}$ 映射至文本嵌入空间 $H_v \in \mathbb{R}^{N_p \times D}$。

### 关键设计一：多模态蒸馏（MDist）

MDist 同时在**响应 token** 和**视觉 token** 两个维度进行 KL 散度蒸馏：

**响应蒸馏**：对齐教师和学生在响应 token 上的输出分布：

$$\mathcal{L}_{res} = \sum_{m=1}^{M} \text{KLD}(\phi_l(y_m | \mathbf{y}_{<m}), \phi_s(y_m | \mathbf{y}_{<m}))$$

**视觉蒸馏**：对齐教师和学生在视觉 token 上的输出分布：

$$\mathcal{L}_{vis} = \sum_{k=1}^{K} \sum_{j=1}^{V} \phi_l(Y_j | \mathbf{y}_{<k}) \log \frac{\phi_l(Y_j | \mathbf{y}_{<k})}{\phi_s(Y_j | \mathbf{y}_{<k})}$$

其中 $K$ 为视觉 token 长度，$V$ 为词表大小。与仅蒸馏响应 token 的传统 LLM 蒸馏不同，MDist 显式地将视觉模态纳入蒸馏范围，确保多模态表示的全面迁移。

### 关键设计二：关系蒸馏（RDist）

RDist 通过构建视觉 token 的自相关矩阵来迁移教师模型捕获视觉 token 间关系的能力。具体地，分别计算教师和学生的自相关矩阵：

$$R_v^s = \mathbf{y}_v^s \otimes \mathbf{y}_v^s \in \mathbb{R}^{N_p \times N_p}, \quad R_v^t = \mathbf{y}_v^t \otimes \mathbf{y}_v^t \in \mathbb{R}^{N_p \times N_p}$$

然后通过最大化余弦相似度来对齐两者：

$$\mathcal{L}_{rel} = 1 - \text{Cos}(R_v^s, R_v^t) = 1 - \frac{R_v^s \cdot R_v^t}{\|R_v^s\| \|R_v^t\|}$$

这种设计编码了视觉 token 之间的空间和语义依赖关系（如物体位置、交互关系），对复杂视觉场景理解至关重要。

### 三阶段训练方案

1. **Distilled Pre-Training（DPT）**：冻结视觉编码器和 LLM，仅训练投影器。在标准自回归损失基础上加入 MDist 和 RDist：$\mathcal{L}_{DPT} = \mathcal{L}_{reg} + \alpha \mathcal{L}_{res} + \beta \mathcal{L}_{vis} + \gamma \mathcal{L}_{rel}$，增强视觉-文本对齐质量。

2. **Supervised Fine-Tuning（SFT）**：标准 SFT，联合训练投影器和 LLM，建立基础多模态理解能力。

3. **Distilled Fine-Tuning（DFT）**：在 SFT 后再次引入蒸馏，精炼学生模型的知识：$\mathcal{L}_{DFT} = \mathcal{L}_{reg} + \alpha' \mathcal{L}_{res} + \beta' \mathcal{L}_{vis} + \gamma' \mathcal{L}_{rel}$

所有损失权重 $\{\alpha, \beta, \gamma\}$ 和 $\{\alpha', \beta', \gamma'\}$ 均设为 1.0, 1.0, 0.5。

## 实验结果

### 主实验：与 SoTA 方法对比

| 方法 | LLM | VQAv2 | GQA | SciQA | MME | MMB | POPE | Avg₁₀ |
|------|-----|-------|-----|-------|-----|-----|------|-------|
| TinyLLaVA (Qwen1.5-0.5B) | 0.5B | 73.9 | 57.4 | 60.9 | 59.8 | 55.0 | 83.7 | 54.7 |
| LLaVA-MOD (Qwen1.5-0.5B) | 0.5B | - | 56.2 | 62.8 | 65.3 | 58.8 | - | 54.1 |
| **LLaVA-KD (Qwen1.5-0.5B)** | **0.5B** | **77.0** | **59.6** | **60.6** | **64.5** | **60.1** | **85.9** | **57.9** |
| TinyLLaVA (Qwen1.5-1.8B) | 1.8B | 73.1 | 55.5 | 65.3 | 61.2 | 57.1 | 83.4 | 56.8 |
| LLaVA-MOD (Qwen1.5-1.8B) | 1.8B | - | 58.7 | 68.0 | 66.7 | 66.3 | 87.0 | 59.9 |
| **LLaVA-KD (Qwen1.5-1.8B)** | **1.8B** | **79.0** | **62.3** | **64.7** | **69.1** | **64.0** | **86.3** | **62.1** |

LLaVA-KD 在 0.5B 和 1.8B 规模上均超越基线，在 Avg₁₀ 上分别提升 3.2% 和 5.3%，且仅需 1.2M 训练样本（LLaVA-MOD 需 5M）。

### 消融实验：三阶段训练方案的效果

| 训练方案 | Avg₁₀ |
|---------|-------|
| PT-SFT（基线） | 54.7 |
| DPT-SFT | 55.6 (+0.9) |
| PT-DFT | 55.8 |
| DPT-DFT | 55.9 |
| PT-SFT-DFT | 56.6 |
| **DPT-SFT-DFT** | **57.9 (+3.2)** |
| DPT-DFT-DFT | 58.0 |

- DPT 带来 0.9% 提升，说明蒸馏式预训练增强了跨模态对齐
- DFT 贡献最大（+2.3%），说明其有效迁移了教师知识
- 跳过 SFT 阶段（DPT-DFT）性能下降，证明 SFT 对知识习得不可或缺
- DPT-DFT-DFT 性能略优但计算开销增大（120 GPU hours），DPT-SFT-DFT 是最佳性价比方案

### 蒸馏目标消融

| 蒸馏目标 | Response | Visual | Avg₁₀ |
|---------|----------|--------|-------|
| DPT: 仅 Response | ✓ | ✗ | 54.9 |
| DPT: Response + Visual | ✓ | ✓ | 55.1 |
| DFT: 仅 Response | ✓ | ✗ | 57.2 |
| DFT: Response + Visual | ✓ | ✓ | **57.7** |

在 DPT 和 DFT 阶段，对视觉 token 的蒸馏均带来额外提升，验证了 MDist 中视觉蒸馏的重要性。

## 亮点与洞察

1. **多模态蒸馏思路新颖**：首次将蒸馏从响应 token 扩展至视觉 token，弥补了现有 LLM 蒸馏方法忽略视觉模态的缺陷
2. **关系蒸馏设计巧妙**：通过视觉 token 自相关矩阵捕获空间和语义关系，而非简单的特征对齐
3. **三阶段训练方案有据可循**：DPT 增强对齐、SFT 建立基础、DFT 精炼知识，每个阶段都有明确的功能定位
4. **架构无关性强**：无需修改模型架构，可直接应用于各种 LlaVA 风格的 MLLM

## 局限性

- 教师和学生必须共享相同的视觉编码器，限制了蒸馏的灵活性
- 蒸馏增加了训练计算和显存开销（需要同时运行教师和学生模型）
- 仅在 LLaVA-1.5 架构上验证，对更先进的架构（如动态分辨率）的适用性未知
- 损失权重（α, β, γ）固定为经验值，缺乏自适应调整机制

## 相关工作

- **小规模 MLLM**：TinyLLaVA, Bunny, MoE-LLaVA, MobileVLM, MiniCPM-V 等通过轻量骨干或结构优化降低成本
- **LLM 蒸馏**：MiniLLM（反向 KLD）、DistiLLM（偏斜 KLD）、CoT 蒸馏等聚焦文本模态
- **MLLM 蒸馏**：LLaVA-MoD（输出 KLD + 偏好蒸馏 + MoE）、LLaVADI 发现多数 LLM 蒸馏策略对 MLLM 无额外收益

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术质量 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.0 |

<!-- RELATED:START -->

## 相关论文

- [EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](../../AAAI2026/multimodal_vlm/em-kd_distilling_efficient_multimodal_large_language_model_w.md)
- [LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)

<!-- RELATED:END -->
