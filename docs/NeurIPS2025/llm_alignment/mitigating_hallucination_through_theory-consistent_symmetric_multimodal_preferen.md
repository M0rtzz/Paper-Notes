---
title: >-
  [论文解读] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization
description: >-
  [NeurIPS 2025][LLM对齐][DPO] 提出 SymMPO（对称多模态偏好优化），通过对比图像的对称配对偏好学习和偏好边际一致性正则化，解决了现有视觉增强型 DPO 方法中目标函数不严格和间接偏好监督两大局限，在五个幻觉评测基准上取得了一致的性能提升。
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "DPO"
  - "多模态幻觉"
  - "偏好优化"
  - "视觉理解"
  - "对称偏好学习"
---

# Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2506.11712](https://arxiv.org/abs/2506.11712)  
**代码**: [https://github.com/Liuwq-bit/SymMPO](https://github.com/Liuwq-bit/SymMPO)  
**领域**: LLM对齐  
**关键词**: DPO, 多模态幻觉, 偏好优化, 视觉理解, 对称偏好学习

## 一句话总结

提出 SymMPO（对称多模态偏好优化），通过对比图像的对称配对偏好学习和偏好边际一致性正则化，解决了现有视觉增强型 DPO 方法中目标函数不严格和间接偏好监督两大局限，在五个幻觉评测基准上取得了一致的性能提升。

## 研究背景与动机

多模态大语言模型 (MLLMs) 虽然在视觉问答、图像描述等任务上表现出色，但严重受到**幻觉问题**困扰——生成与输入图像不一致的内容。DPO 已被广泛应用于减轻 MLLM 幻觉，现有方法通常包含两个组件：

**回复导向偏好学习**：比较同一输入下好的回复 $y_w$ 和差的回复 $y_l$

**视觉导向偏好学习**（可选）：使用对比图像 $(m_w, m_l)$ 搭配相同回复 $y_w$，增强模型对视觉输入的关注

然而，作者深入分析后发现现有方法存在两个关键缺陷：

**缺陷1：目标函数不严格。** 在视觉导向偏好学习中，对比图像 $m_w \neq m_l$ 导致配分函数 $Z(m_w, x)$ 和 $Z(m_l, x)$ **不能**直接消去。现有方法直接假设它们可消除，与标准 DPO 的理论推导不一致。作者在附录 B 中给出了详细的数学证明。

**缺陷2：间接偏好监督。** 现有的视觉导向方法使用对比图像+相同回复的三元组，本质上依赖图像对比而非回复对比。这偏离了 DPO 的核心原理——通过配对回复建立偏好关系，导致视觉理解提升效果有限。

## 方法详解

### 整体框架

SymMPO 的完整损失函数为：

$$\mathcal{L}_{SymMPO} = \mathcal{L}_{DPO_m} + \lambda \mathcal{L}_{Pair} + \gamma \mathcal{L}_{Margin} + \eta \mathcal{L}_{AncPO}$$

其中四个组件分别负责回复质量保证、对称配对偏好学习、偏好边际一致性正则化和锚定偏好正则化。

### 关键设计

**1. 对称配对偏好学习 ($\mathcal{L}_{Pair}$)**

核心创新在于为对比图像 $m'$ 额外生成其最优回复 $y_{w}'$。由于 $m'$ 和 $m$ 高度相似（通过 CLIP 相似度选择），它们的最优回复也相似但有微小差异，自然构成"硬负例"。

对称偏好建模：

$$P_{BT}(y_w \succ y_{w}'|m,x) \wedge P_{BT}(y_{w}' \succ y_w|m',x)$$

即在图像 $m$ 下 $y_w$ 优于 $y_{w}'$，在图像 $m'$ 下 $y_{w}'$ 优于 $y_w$。由于使用了回复对作为偏好监督（而非图像对），配分函数可以自然消去，目标函数严格符合标准 DPO 推导。

**2. 偏好边际一致性正则化 ($\mathcal{L}_{Margin}$)**

Bradley-Terry 模型只约束偏好的序关系，但对于高度相似的输入 $(m,x)$ 和 $(m',x)$，偏好边际也应当一致。因此引入：

$$\mathcal{L}_{Margin} = \mathbb{E}\left(\Delta(m,x,y_w,y_{w}') - \Delta(m',x,y_{w}',y_w)\right)^2$$

其中 $\Delta(m,x,y_w,y_{w}') = r(m,x,y_w) - r(m,x,y_{w}')$，确保两个方向的偏好差距量级一致。

**3. 锚定偏好正则化 ($\mathcal{L}_{AncPO}$)**

防止模型通过降低正面回复概率来扩大偏好差距：

$$\mathcal{L}_{AncPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|m,x)}{\pi_{ref}(y_w|m,x)} - \delta\right) + \log\sigma\left(\beta\log\frac{\pi_\theta(y_{w}'|m',x)}{\pi_{ref}(y_{w}'|m',x)} - \delta\right)\right]$$

**4. 低成本偏好数据构建管道**

提出 Caption-Anchored Claim Extraction-and-Rewriting 四阶段管道：
1. 开源 MLLM（Qwen2.5-VL-32B）生成图像描述
2. 参考模型多次采样生成回复
3. LLM（DeepSeek-V3）对比回复与描述，提取一致/不一致声明
4. LLM 基于声明改写生成正/负回复

对比图像通过 CLIP 最近邻而非传统的图像变换生成，保持语义相似性。

### 损失函数 / 训练策略

超参数设置：$\beta=0.1$，$\delta=0$，$\lambda=0.5$，$\gamma=1e-4$，$\eta=1.0$。训练 2 个 epoch，学习率 5e-6，batch size 64，4 × NVIDIA A100-40GB。训练数据来自 TPO 的 21.4k 图像-提示对。

## 实验关键数据

### 主实验

在 LLaVA-1.5-7B 上的严格对比（相同数据/训练条件下 DPO vs mDPO vs SymMPO）：

| 方法 | HallusionBench aAcc↑ | MMHal Score↑ | AMBER Acc↑ | AMBER F1↑ | MMStar↑ |
|------|---------------------|-------------|-----------|----------|---------|
| DPO | 40.21 | 2.44 | 71.3 | 82.6 | 33.4 |
| mDPO | 42.78 | 2.71 | 80.6 | 86.3 | 34.2 |
| **SymMPO** | **44.28** | **2.89** | **82.6** | **87.7** | **34.8** |

在 LLaVA-1.5-13B 上同样保持优势：

| 方法 | HallusionBench aAcc↑ | MMHal Score↑ | AMBER Acc↑ | AMBER F1↑ | MMStar↑ |
|------|---------------------|-------------|-----------|----------|---------|
| DPO | 39.50 | 2.65 | 69.2 | 84.6 | 33.0 |
| mDPO | 39.85 | 2.93 | 83.8 | 88.8 | 35.0 |
| **SymMPO** | **44.55** | **3.01** | **84.9** | **89.1** | **35.2** |

### 消融实验

LLaVA-1.5-7B 上的组件消融：

| 变体 | HallusionBench aAcc↑ | MMHal Score↑ | AMBER Acc↑ | MMStar↑ |
|------|---------------------|-------------|-----------|---------|
| SymMPO (完整) | 44.28 | 2.89 | 82.6 | 34.8 |
| w/o $\mathcal{L}_{Pair}$ | 43.22 | 2.53 | 81.7 | 33.8 |
| w/o $\mathcal{L}_{Margin}$ | 44.46 | 2.40 | 82.0 | 34.5 |
| w/o $\mathcal{L}_{AncPO}$ | 40.83 | 2.39 | 79.5 | 36.2 |

**对比图像类型实验**测试了 5 种策略（Similar/Black/Cropped/Noisy/Synthetic），发现：
- SymMPO 在几乎所有类型下优于 mDPO（Black 除外）
- Similar、Noisy、Synthetic 效果优于 Black 和 Cropped，因为前三者更好地保留了语义相似性

### 关键发现

1. 配对偏好学习 ($\mathcal{L}_{Pair}$) 对整体性能贡献最大，移除后 MMHal Score 下降 0.36
2. 锚定正则化对 HallusionBench 至关重要（移除后 aAcc 降 3.45 个百分点）
3. SymMPO 和 mDPO 在 Object-HalBench 上均弱于 DPO，这与数据构建管道偏向场景概述而非细粒度视觉描述有关
4. CLIP 最近邻对比图像比传统的噪声注入/裁剪策略更有效

## 亮点与洞察

1. **理论严格性**：深入分析了配分函数在多模态 DPO 中的角色，指出现有方法直接消去 $Z(m_w,x)$ 和 $Z(m_l,x)$ 的理论缺陷，并给出了严格的替代方案
2. **对称设计自然解决配分函数问题**：通过让每个方向都使用相同多模态输入+不同回复，配分函数自然消去
3. **量化偏好边际**：超越传统的序偏好（$y_w \succ y_l$），引入偏好差距大小的一致性约束
4. **实用的数据构建管道**：避免了昂贵的 GPT-4V API 调用，使用开源模型 + DeepSeek-V3 的组合方案

## 局限与展望

1. **细粒度视觉理解有限**：在 Object-HalBench 上的表现不佳，说明数据构建管道需要增强对细节描述的关注
2. **额外计算开销**：需要为对比图像构建对应的最优回复，增加了数据准备成本
3. 仅在 LLaVA-1.5 架构上验证，未测试更强的基础模型（如 Qwen-VL、InternVL）
4. 对比图像使用 CLIP 最近邻可能引入选择偏差，对数据集分布敏感

## 相关工作与启发

- **mDPO** (Wang et al., 2024)：引入视觉导向对比学习，但目标函数不严格。SymMPO 直接修正了其理论缺陷
- **RLAIF-V** (Yu et al., 2024)：去混淆的候选回复生成，SymMPO 在数据构建上更高效
- **TPO** (He et al., 2024)：主题级自纠正范式，与 SymMPO 关注不同层面
- **OPA-DPO** (Yang et al., 2025)：自适应探索-利用平衡，但仍使用间接偏好监督
- SymMPO 的对称思想可以推广到其他需要多视角对比的偏好学习场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 对称偏好学习和边际一致性正则化是有价值的创新，理论分析揭示了被忽略的问题
- 实验充分度: ⭐⭐⭐⭐ 五个基准、两个模型规模、完整消融和对比图像类型分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数学推导严谨，但部分符号较为密集
- 价值: ⭐⭐⭐⭐ 对多模态偏好优化的理论基础做出了实质性贡献，方法通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](../../ACL2025/llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](../../ICLR2026/llm_alignment/mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[ICML 2025\] D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](../../ICML2025/llm_alignment/d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)
- [\[ICCV 2025\] MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](../../ICCV2025/llm_alignment/magicid_hybrid_preference_optimization_for_id-consistent_and_dynamic-preserved_v.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)

</div>

<!-- RELATED:END -->
