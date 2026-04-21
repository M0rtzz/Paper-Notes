---
title: >-
  [论文解读] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance
description: >-
  [ICCV 2025][图像生成][视觉风格提示] 提出 **负视觉查询引导（NVQG）** 方法，通过在 self-attention 层中将参考图的 query 注入作为负向引导来抑制内容泄漏，实现了无需训练的高质量视觉风格提示，在风格相似度和文本对齐上均优于现有方法。
tags:
  - ICCV 2025
  - 图像生成
  - 视觉风格提示
  - 内容泄漏
  - Classifier-Free Guidance
  - 自注意力特征交换
  - 负视觉查询引导
  - 无训练风格迁移
  - 扩散模型
---

# StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance

**会议**: ICCV 2025  
**arXiv**: [2510.06827](https://arxiv.org/abs/2510.06827)  
**代码**: [GitHub](https://github.com/) (作者已公开)  
**领域**: image_generation / style_transfer  
**关键词**: 视觉风格提示, 内容泄漏, Classifier-Free Guidance, 自注意力特征交换, 负视觉查询引导, 无训练风格迁移, 扩散模型

## 一句话总结

提出 **负视觉查询引导（NVQG）** 方法，通过在 self-attention 层中将参考图的 query 注入作为负向引导来抑制内容泄漏，实现了无需训练的高质量视觉风格提示，在风格相似度和文本对齐上均优于现有方法。

## 研究背景与动机

**核心问题**：文本到图像扩散模型中，使用参考图像作为视觉风格提示（visual style prompt）时，存在「内容泄漏」（content leakage）难题——参考图中的姿态、布局、物体等非风格元素会不可避免地渗入生成结果，降低多样性和文本对齐度。

**现有方法的不足**：

- **训练式方法**（DreamBooth、LoRA、Textual Inversion、IP-Adapter 等）：需要额外训练，计算开销大，且存在风格与内容的固有权衡
- **无训练方法**（StyleAligned、CrossAttn、StyleID 等）：通过交换 self-attention 的 key/value 实现风格迁移，但无法完全消除内容泄漏；部分方法不支持真实图像作为参考，或主要面向 I2I 而非 T2I 场景
- **CFG 未被纳入考量**：先前工作在操作特征时忽略了 Classifier-Free Guidance 的作用，导致图像质量和文本对齐变差

**动机**：需要一种方法能够**独立控制**来自视觉提示的风格强度和内容强度，使生成结果在保持参考风格的同时与文本提示完全对齐，且无需任何额外训练。

## 方法详解

### 整体框架

StyleKeeper 接收一个文本提示和一个视觉风格提示，通过四个核心组件生成无内容泄漏的风格化图像：

1. **CFG + 交换自注意力**（CFG with Swapping Self-Attention）
2. **负视觉查询引导**（Negative Visual Query Guidance, NVQG）
3. **随机编码**（Stochastic Encoding）——用于真实参考图
4. **颜色校准**（Color Calibration）——用于真实参考图

### 关键设计 1：CFG 与交换自注意力

核心思想是维护两个去噪过程——原始过程（由文本提示驱动）和参考过程（由视觉风格提示驱动），在 self-attention 层中将参考过程的 **key 和 value** 注入原始过程：

$$\text{Attention}(Q_{\text{text}}, K_{\text{visual}}, V_{\text{visual}}) = \text{Softmax}\left(\frac{Q_{\text{text}} K_{\text{visual}}^\top}{\sqrt{d}}\right) V_{\text{visual}}$$

作者首次将这一操作与 CFG 结合，设计了统一的引导公式：

$$\tilde{\epsilon}_\theta = (1+w) \cdot \ddot{\epsilon}_\theta(x_t, Q_{\text{text}}, KV_{\text{visual}}) - w \cdot \epsilon_\theta(x_t, \emptyset)$$

其中 query 来自原始过程（保留内容），key/value 来自参考过程（携带风格）。CFG 的引入显著提升了图像质量和文本对齐。

### 关键设计 2：负视觉查询引导（NVQG）

**核心洞察**：KV 注入虽然倾向于保留内容、传递风格，但参考图的内容信息仍然通过 KV 泄漏。NVQG 利用贝叶斯规则，将视觉提示的条件分解为风格和内容两个因子，然后通过**负向引导**减弱内容因子的影响。

具体地，通过 query 注入（将参考图的 query 注入原始过程）来近似「仅包含参考图内容」的得分：

$$\ddot{\epsilon}_\theta(x_t, Q_\emptyset, KV_{\text{visual}}^{\text{content}}) \approx \ddot{\epsilon}_\theta(x_t, Q_{\text{visual}}, KV_\emptyset)$$

然后将其作为负向引导项，从最终得分中减去，从而有效抑制参考图内容的泄漏。该方法本质上是「故意模拟内容泄漏场景，再将其反向减去」。

### 关键设计 3：自注意力层的选择

扩散模型包含 downblock、bottleneck 和 upblock 三部分：

- **Bottleneck**：包含图像的内容元素，不应交换（否则直接泄漏内容）
- **Downblock**：特征图的内容布局不清晰，交换会导致生成图像散乱
- **Upblock**：仅在 upblock 应用交换自注意力，能有效传递风格而不泄漏内容

进一步实验发现，从 SDXL 的第 24 层开始交换是最优的平衡点，且该最优层对不同参考图保持一致。

### 关键设计 4：真实图像支持

- **随机编码**：直接通过前向扩散过程  $x_t^{\text{visual}} = \sqrt{\alpha_t} \cdot x_0^{\text{visual}} + \sqrt{1-\alpha_t} \cdot \epsilon_t$  获得中间潜变量，避免 DDIM inversion 的累积误差和伪影
- **颜色校准**：在去噪过程中对预测的 $x_0$ 执行 AdaIN 操作，匹配参考图的通道均值和标准差，实现精确的色彩对齐

### 损失函数 / 采样策略

本方法为 **无训练**方法，不涉及损失函数训练。核心在于采样阶段的引导公式设计，通过组合以下三个得分实现风格-内容分离：

- 正向引导：$\ddot{\epsilon}_\theta(x_t, Q_{\text{text}}, KV_{\text{visual}})$ — 携带目标风格的条件得分
- 负向引导：$\ddot{\epsilon}_\theta(x_t, Q_{\text{visual}}, KV_\emptyset)$ — 携带参考内容的得分（需减去）
- 无条件得分：$\epsilon_\theta(x_t, \emptyset)$ — CFG 的无条件基线

## 实验关键数据

### 主实验：定量比较

| 方法 | Style Similarity (DINO↑) | Text Alignment (CLIP↑) | Diversity (LPIPS↑) | Gram Matrix↑ |
|------|------------------------|----------------------|-------------------|-------------|
| **StyleKeeper (Ours)** | **最优** | **最优** | **最优** | **0.791** |
| StyleAligned | 次优 | 中等 | 差（内容泄漏） | 0.759 |
| IP-Adapter | 高（但牺牲文本） | 最差 | 中等 | 0.768 |
| DreamBooth-LoRA | 中等 | 中等 | 中等 | 0.759 |
| StyleDrop | 最差 | 中等 | 中等 | 0.659 |

评估设置：40 张参考图 × 120 个文本提示 × 6 个初始噪声 = **720 张**生成图像。

### 用户研究

| 方法 | 用户偏好比例 |
|------|------------|
| **StyleKeeper** | **58.15%** |
| IP-Adapter | 18.47% |
| StyleAligned | 13.15% |
| DreamBooth-LoRA | 7.66% |
| StyleDrop | 2.58% |

62 名参与者，20 组评测。超过半数用户认为 StyleKeeper 在风格对齐和文本对齐上最优。

### 消融实验

| 配置 | 效果 |
|------|------|
| 无 CFG + 无 NVQG | 严重伪影，图像质量极差 |
| CFG + 无 NVQG | 质量提升，但内容泄漏严重（布局、结构） |
| CFG + NVQG（完整方法） | 最佳结果，内容与风格清晰分离 |
| DDIM inversion vs. 随机编码 | 随机编码在所有指标上优于 DDIM inversion |
| 移除颜色校准 | 风格相似度下降，色彩匹配变差 |

### 关键发现

1. **NVQG 对消除内容泄漏至关重要**：无 NVQG 时，参考图的姿态/布局/物体会渗入结果；开启 NVQG 后，各种复杂场景（名画风格、多实例、指定姿态）均可正确生成
2. **仅在 upblock 交换自注意力是最优策略**：存在一个跨参考图一致的「拐点层」，所有指标在此处发生突变
3. **随机编码 > DDIM inversion**：K-S 检验表明随机编码的潜变量更接近标准高斯分布，P 值 > 0.05
4. **方法可推广**：兼容 ControlNet（I2I 风格迁移）、DreamBooth-LoRA、Stable Diffusion v1.5 和 Pixart-α 等不同模型

## 亮点与洞察

1. **NVQG 的设计思路极为精妙**：不是直接「阻止」内容泄漏，而是故意模拟泄漏场景作为负样本，再通过 CFG 的减法操作将其消除——这种「以毒攻毒」的策略简洁而高效
2. **首次将 CFG 与特征交换机制统一**：先前工作忽略了 CFG 在特征操作场景下的作用，本文证明 CFG 对提升质量和文本对齐不可或缺
3. **层选择的系统分析**：通过可视化注意力图揭示了 upblock 晚层聚焦风格对应区域、早层注意力过于宽泛导致泄漏的机制
4. **随机编码替代 DDIM inversion**：一步操作即可获得统计对齐的中间潜变量，既高效又无累积误差，还免去存储中间状态
5. **完全无训练**：不修改模型权重，不需要额外数据集，仅在采样阶段操作，即插即用

## 局限性

1. **受限于预训练模型能力**：无法生成模型训练集范围外的概念（如 "stone golem" 生成失败）
2. **视觉风格与文本风格冲突时视觉风格占主导**：当文本描述的风格与参考图风格矛盾时，参考图风格会压制文本
3. **额外计算开销**：需要同时运行原始和参考两个去噪过程，以及 NVQG 的额外前向传播，推理时间约为标准生成的 3 倍
4. **仅验证了 T2I 和 I2I 场景**：未扩展到视频生成等其他模态
5. **最优层选择依赖具体架构**：SDXL 的第 24 层最优，换用其他架构需重新搜索

## 相关工作与启发

- **StyleAligned [Hertz et al., 2023]**：共享 self-attention 实现风格对齐，但保留原始特征导致内容泄漏——本文通过 NVQG 解决这一核心缺陷
- **CrossAttn [Alaluf et al., 2024]** / **StyleID [Chung et al., 2024]**：基于 DDIM inversion 的 KV 注入，主要面向 I2I，风格反映不足——本文的随机编码和颜色校准提供了更优的真实图像处理方案
- **Composable Diffusion [Liu et al., 2022]**：概念组合的贝叶斯分解框架——本文将其扩展到特征空间的风格/内容分解
- **MasaCtrl [Cao et al., 2023]**：mutual self-attention 控制——本文揭示了不同 block 层的语义差异，为特征操作方法提供了选层指导
- **AdaIN [Gatys, 2015]**：经典风格迁移的通道统计匹配——本文在扩散模型中间步骤使用 AdaIN 进行颜色校准，是对经典方法的巧妙复用

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | NVQG 的负向引导思路新颖；CFG 与特征交换统一有见地 |
| 技术深度 | ⭐⭐⭐⭐ | 贝叶斯分解推导严谨，层选择分析系统 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 720 张图定量评估 + 62 人用户研究 + 全面消融 + 多模型验证 |
| 实用价值 | ⭐⭐⭐⭐ | 无训练即插即用，兼容多种模型和 ControlNet |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图示丰富，但部分公式符号较密 |
| **总评** | **⭐⭐⭐⭐** | 扎实的工作，在无训练视觉风格提示领域做出了有意义的推进 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [ReNeg: Learning Negative Embedding with Reward Guidance](../../CVPR2025/image_generation/reneg_learning_negative_embedding_with_reward_guidance.md)
- [Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [Visual Generation Without Guidance](../../ICML2025/image_generation/visual_generation_without_guidance.md)
- [Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)

<!-- RELATED:END -->
