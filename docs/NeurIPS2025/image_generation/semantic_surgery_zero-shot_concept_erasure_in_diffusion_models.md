---
title: >-
  [论文解读] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][概念擦除] 提出Semantic Surgery，一种无需重训练的零样本推理时概念擦除框架，通过在扩散过程之前对文本嵌入进行校准向量减法，结合Co-Occurrence Encoding处理多概念擦除和视觉反馈环路解决潜在概念持久性问题…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "概念擦除"
  - "扩散模型"
  - "文本嵌入操作"
  - "零样本"
  - "推理时方法"
  - "安全生成"
---

# Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.22851](https://arxiv.org/abs/2510.22851)  
**作者**: Lexiang Xiong, Chengyu Liu, Jingwen Ye, Yan Liu, Yuecong Xu (NUS, 四川大学)  
**代码**: [GitHub](https://github.com/Lexiang-Xiong/Semantic-Surgery)  
**领域**: 图像生成  
**关键词**: 概念擦除, 扩散模型, 文本嵌入操作, 零样本, 推理时方法, 安全生成  

## 一句话总结

提出Semantic Surgery，一种无需重训练的零样本推理时概念擦除框架，通过在扩散过程之前对文本嵌入进行校准向量减法，结合Co-Occurrence Encoding处理多概念擦除和视觉反馈环路解决潜在概念持久性问题，在物体/NSFW/风格/名人擦除任务上全面超越SOTA。

## 研究背景与动机

### 问题背景

文本到图像扩散模型（如Stable Diffusion）具有生成有害或侵权内容的风险（如色情材料、版权风格），需要概念擦除技术来应对。现有概念擦除方法分为两大类：参数修改方法和推理时方法。

### 已有工作的不足

- **参数修改方法**（ESD, UCE, MACE, Receler等）：通过微调或模型编辑"遗忘"概念，但面临灾难性遗忘问题，导致通用生成能力下降；建立的是静态防御，对概念变体（如改写prompt）鲁棒性差；多概念擦除时存在累积干扰
- **已有推理时方法**（SLD, SAFREE等）：在token级别或扩散中间阶段操作，但自注意力机制会将目标概念语义扩散到整个token序列中，使得局部token级干预不充分；且未解决U-Net先验导致的概念"复活"问题
- **核心矛盾**：如何在零样本/推理时策略下，同时实现擦除完整性（completeness）和局部性（locality），并保持对prompt变体的鲁棒性

### 核心动机

利用语言嵌入空间的线性结构（类似word2vec的类比关系），在扩散过程之前对全局文本嵌入进行"语义手术"——动态评估目标概念的存在强度，然后执行校准的向量减法来中和其影响。这一思路受到LLM中"激活工程"（activation engineering）的启发。

## 方法详解

### 整体框架

Semantic Surgery包含三个核心模块：(A) Semantic Analysis（语义分析）：Semantic Biopsy检测概念存在性 + Co-Occurrence Encoding构建统一擦除方向；(B) Core Surgery（核心手术）：校准向量减法生成净化嵌入；(C) Visual Feedback Loop（视觉反馈）：检测并缓解潜在概念持久性。

### 1. 语义建模与单概念擦除

基于CLIP嵌入空间的线性类比性质（如 $\phi(\text{king}) - \phi(\text{man}) \approx \phi(\text{queen}) - \phi(\text{woman})$）。定义中性参考嵌入 $e_n = \phi("")$，对于目标概念 $c$，其擦除方向为：

$$\Delta e_{\text{erase}} = e_{\text{erase}} - e_n$$

引入二值存在指示器 $\rho \in \{0,1\}$，经过语义手术后的嵌入为：

$$e'_{\text{input}} = e_{\text{input}} - \rho \cdot \Delta e_{\text{erase}}$$

当 $\rho=1$（概念存在）时执行减法投影到无概念子空间，$\rho=0$时保持不变。

### 2. Co-Occurrence Encoding（多概念擦除）

朴素的多概念擦除（逐个减去各概念向量 $\sum_i \rho_{c_i} \Delta e_{c_i}$）会过度消除语义重叠部分。例如同时擦除"海鸥"和"麻雀"会过度削弱鸟类共有特征。解决方案是利用CLIP的上下文建模能力，将所有活跃概念拼接为组合prompt，由CLIP编码器通过短语级交互自动解决语义重叠：

$$\Delta e_{\text{co}} = \phi(p_{\text{co}}) - e_n, \quad p_{\text{co}} = \bigoplus_{i=1}^{n} p_{c_i}$$

最终联合手术操作为 $\hat{e}'_{\text{input}} = e_{\text{input}} - \hat{\rho}_{\text{joint}} \cdot \Delta e_{\text{co}}$。

### 3. Semantic Biopsy（语义检测）

核心问题：如何在推理时仅从输入嵌入估计概念存在强度 $\rho$。通过Theorem 1的投影分解，概念存在强度与嵌入在擦除方向上的投影大小成正比。定义余弦相似度 $\alpha_c = \cos(e_{\text{input}}, \Delta e_{\text{erase}})$，经验发现概念存在与否的 $\alpha_c$ 分布具有显著的统计可分离性（Assumption 3.1），存在阈值 $\beta$ 使两类分布在 $[\beta-\epsilon, \beta+\epsilon]$ 区间外以高概率分开。基于此，使用sigmoid校准器估计：

$$\hat{\rho}(\alpha_c) = \sigma_{\text{sigmoid}}\left(\frac{\alpha_c - \beta}{\gamma}\right)$$

其中 $\gamma$ 控制分类锐度，理论保证在高概率下 $\hat{\rho}$ 与真实二值标签的误差不超过 $\delta_{\text{err}}$。

### 4. 视觉反馈环路（LCP缓解）

针对潜在概念持久性（LCP）问题——即使文本嵌入中已移除概念语义，U-Net的视觉先验仍可能因其他概念的隐含关联而重新生成目标内容（如prompt中的"道路"通过U-Net先验暗示"树木"）。缓解流程：

1. 用初始手术后的嵌入 $\hat{e}'_s$ 生成图像
2. 使用视觉检测器 $\mathcal{D}$ 检查生成图像中是否存在目标概念
3. 若检测到概念"复活"（$\hat{\rho}^{(k)}_{\text{im}} \geq \tau_{\text{vis}}$），将视觉检测到的概念加入擦除集合
4. 构建增强的擦除方向 $\Delta e^*_{\text{co}}$，以更强的 $\hat{\rho}^*_{\text{joint}}$ 重新执行手术

理论上证明（Theorem 3）二次手术可有效降低LCP风险，且增加的推理开销在实际安全评估中可接受。

## 实验关键数据

### 实验1：物体擦除（CIFAR-10, 10类）

使用OWL-ViT独立检测器评估。$\text{Acc}_E$: 简单prompt擦除率（↓好），$\text{Acc}_R$: 改写prompt擦除率（↓好），$\text{Acc}_L$: 非目标保留率（↑好），H: 调和均值（↑好）。

| 方法 | $\text{Acc}_E$↓ | $\text{Acc}_R$↓ | $\text{Acc}_L$↑ | H↑ |
|------|:---:|:---:|:---:|:---:|
| SD v1.4 | 99.10 | 87.20 | 87.33 | - |
| ESD-x | 22.20 | 63.20 | 85.49 | 56.03 |
| ESD-u | 12.50 | 39.40 | 81.87 | 73.72 |
| AC | 3.30 | 47.60 | 85.53 | 70.00 |
| UCE | 2.30 | 28.20 | 85.50 | 81.90 |
| Receler | 2.50 | 10.00 | 81.58 | 88.74 |
| MACE | 0.40 | 13.80 | 79.09 | 87.13 |
| **Ours** | **1.50** | **2.00** | **85.56** | **93.58** |

H-score达93.58（+4.84 vs Receler）。鲁棒性指标 $\text{Acc}_R$ 仅2.00，是Receler的1/5、MACE的1/7，体现了全局嵌入操作对prompt变体的天然抗性。同时保持最优局部性（85.56）。

### 实验2：显式内容移除 + 风格擦除 + 对抗鲁棒性

**NSFW移除（I2P, 4703 prompts）**：擦除"nude/naked/sexual/erotic"四概念组。

| 方法 | 类型 | 检测总数↓ | FID↓ | CLIP↑ |
|------|------|:---:|:---:|:---:|
| SD v1.4 | 原始 | 751 | 14.04 | 31.34 |
| ESD-u | 参数修改 | 55 | 15.1 | 30.21 |
| MACE | 参数修改 | 123 | 13.42 | 29.41 |
| SAFREE | 推理时 | 82 | - | - |
| **Ours** | **推理时** | **1** | **12.2** | **30.75** |

NSFW实例从751降至仅1例，比SAFREE(82)减少98%+，FID(12.2)甚至优于原始模型。

**艺术风格擦除（100位艺术家）**：$H_a = \text{CLIP}_s - \text{CLIP}_e$，越高越好。

| 方法 | CLIPe↓ | CLIPs↑ | $H_a$↑ | FID-30K↓ | CLIP-30K↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| UCE | 21.35 | 26.32 | 4.97 | 77.72 | 19.17 |
| ESD-u | 19.66 | 19.55 | -0.11 | 17.07 | 27.76 |
| MACE | 22.59 | 28.58 | 5.99 | 12.71 | 29.51 |
| **Ours** | **20.75** | **28.84** | **8.09** | **14.04** | **31.34** |

$H_a$=8.09（+2.1 vs MACE），FID/CLIP与原始SD v1.4完全一致，通用质量零损害。

**对抗攻击鲁棒性**：

| 攻击类型 | 方法 | ASR↓ |
|---------|------|:---:|
| 黑盒(RAB, 380条) | SLD | 78.68% |
| 黑盒 | SAFREE | 55.80% |
| 黑盒 | MACE | 3.95% |
| **黑盒** | **Ours** | **1.05%** |
| **白盒(UnlearnDiffAtk)** | **Ours** | **0.0%** |

黑盒ASR仅1.05%（p=0.0089 vs MACE），白盒ASR为0%。Semantic Biopsy兼具威胁检测功能。

**多概念名人擦除（100名人）**：$H_c$ 达0.965，显著超越MACE(0.892)、UCE(0.554)、Receler(0.441)，且FID/CLIP保持与原始模型一致。

## 亮点

- **无需训练的全局嵌入操作**：与所有参数修改方法不同，Semantic Surgery在扩散前直接操作文本嵌入，保持原模型完全不变，同时在多个任务上超越需要重训练的SOTA方法
- **Co-Occurrence Encoding**：利用CLIP上下文建模能力解决多概念擦除中的语义重叠问题，避免朴素向量减法的过度削减，理论与实践均证明其优越性
- **LCP视觉反馈机制**：首次识别并解决潜在概念持久性问题，通过视觉检测+二次手术闭环缓解U-Net先验导致的概念"复活"
- **天然对抗鲁棒性**：Semantic Biopsy基于全局语义相似度判断概念存在性，对prompt改写和对抗攻击天然免疫（白盒ASR=0%），且可兼作威胁检测系统
- **通用质量零损害**：在风格/名人擦除等任务中FID/CLIP与原始模型一致，彻底解决了参数修改方法的质量退化顽疾

## 局限与展望

- **依赖CLIP嵌入空间的线性假设**：方法理论基础建立在嵌入空间的线性类比关系上，对于高度非线性的复杂语义关系可能失效
- **视觉反馈增加推理开销**：LCP环路需额外生成一轮图像+视觉检测，虽然仅用于安全关键任务，仍增加延迟
- **阈值参数需任务调整**：决策阈值 $\beta$ 仍需根据任务经验设置，$\gamma=0.02$ 和 $\tau=0.5$ 虽跨任务固定但未必最优
- **仅在SD v1.4上验证**：未展示对SDXL、SD3、Flux等更新架构的适用性
- **单概念简单prompt擦除效力略低于MACE**（$\text{Acc}_E$ 1.50 vs 0.40），但鲁棒性大幅领先
- **概念嵌入质量依赖文本编码器**：若CLIP对某些概念编码不佳，向量减法效果会受限

## 与相关工作的对比

- **ESD (ICCV'23)**：通过微调U-Net，ESD-u/ESD-x在擦除-保留权衡上差（H=73.72/56.03 vs 93.58），且鲁棒性差
- **UCE (WACV'24)**：统一概念编辑，多概念场景下通用质量严重退化（风格擦除FID=77.72）
- **MACE (NeurIPS'24)**：多概念能力最强的参数修改方法，但鲁棒性远不如本文（$\text{Acc}_R$ 13.80 vs 2.00），且多任务下质量有下降
- **Receler (ECCV'24)**：轻量级参数修改方法，物体擦除表现好但局部性低（$\text{Acc}_L$ 81.58 vs 85.56）
- **SLD (ICCV'23)**：基于安全引导的推理时方法，精度不足（I2P: 149 vs 1），对抗鲁棒性极差（ASR=78.68%）
- **SAFREE (NeurIPS'24)**：token级嵌入投影的推理时方法，性能优于SLD但远不及本文（I2P: 82 vs 1），因局部token操作无法阻止自注意力扩散语义

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次提出全局文本嵌入层面的零样本概念擦除，Co-Occurrence Encoding和LCP反馈机制原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖物体/NSFW/风格/名人/对抗攻击五大维度，与参数修改和推理时两类方法全面对比
- 写作质量: ⭐⭐⭐⭐ — 问题形式化严谨，理论与实践结合好，但公式符号较多增加阅读负担
- 价值: ⭐⭐⭐⭐⭐ — 所有主要任务上全面SOTA且无需重训练，极具实用价值，可直接部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](../../ICML2026/image_generation/orthogonal_concept_erasure_for_diffusion_models.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
