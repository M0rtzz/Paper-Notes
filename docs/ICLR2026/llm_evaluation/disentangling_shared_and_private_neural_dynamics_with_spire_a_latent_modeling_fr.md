---
title: >-
  [论文解读] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation
description: >-
  [ICLR2026][latent variable model] 提出 SPIRE（Shared–Private Inter-Regional Encoder），一种非线性双潜空间自编码器框架，通过跨区域对齐与正交解缠损失将多脑区颅内记录分解为共享与专属子空间，仅在基线数据训练即可检测 DBS 刺激引发的频率依赖性网络重组。
tags:
  - ICLR2026
  - latent variable model
  - shared-private disentanglement
  - deep brain stimulation
  - multi-region neural dynamics
  - autoencoder
---

# Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation

**会议**: ICLR2026  
**arXiv**: [2510.25023](https://arxiv.org/abs/2510.25023)  
**代码**: [GitHub](https://github.com/Rahil-Soroush/spire-iclr2026)  
**领域**: others  
**关键词**: latent variable model, shared-private disentanglement, deep brain stimulation, multi-region neural dynamics, autoencoder

## 一句话总结

提出 SPIRE（Shared–Private Inter-Regional Encoder），一种非线性双潜空间自编码器框架，通过跨区域对齐与正交解缠损失将多脑区颅内记录分解为共享与专属子空间，仅在基线数据训练即可检测 DBS 刺激引发的频率依赖性网络重组。

## 研究背景与动机

**领域现状**：运动障碍（肌张力障碍、帕金森病等）涉及基底神经节-丘脑-皮层回路的功能失调。深脑刺激（DBS）靶向苍白球内侧部（GPi）和丘脑底核（STN）在临床上疗效显著，但其调制跨区域神经动态的网络级机制仍不清楚。

**现有痛点**：大多数 DBS 分析聚焦于局部特征（频谱功率、诱发电位），忽视了跨区域协调模式的变化。现有潜变量模型存在关键限制：(1) GPFA 和 CCA 假设线性关系，无法捕捉真实神经数据的非线性结构；(2) DLAG 虽能分解 shared/private 成分，但受限于线性高斯框架且主要针对 spike 数据；(3) SharedAE、MMVAE 等多模态模型对齐共享空间，但并非为颅内刺激记录设计，缺少显式的 shared-private 解缠机制。

**核心矛盾**：缺少一个同时满足三个条件的框架——非线性建模能力、显式的 shared vs. private 分解、以及适用于人类 LFP 数据在外部扰动下的分析。理解刺激如何重组固有的跨区域协调模式，对于揭示 DBS 的回路级作用机制至关重要。

**切入角度**：设计一个"训练于基线、推理于扰动"的双潜空间框架——在无刺激数据上建立内在协调的参考模型，然后在 DBS 条件下观察共享潜空间的重组模式，借此揭示刺激对网络级动态的影响。

## 方法详解

### 整体框架

SPIRE 为每个脑区 $r$ 配备独立的 GRU encoder-decoder。编码器将多通道输入 $x^{(r)} \in \mathbb{R}^{B \times T \times C_r}$ 映射为隐状态 $h^{(r)}$，经线性投影分别得到共享潜变量 $z_{\text{sh}}^{(r)} \in \mathbb{R}^{B \times T \times d_{\text{sh}}}$ 和专属潜变量 $z_{\text{pr}}^{(r)} \in \mathbb{R}^{B \times T \times d_{\text{pr}}}$。解码器从拼接的潜变量重建原始信号：$\hat{x}^{(r)} = f_{\text{dec}}^{(r)}([z_{\text{sh}}^{(r)}, z_{\text{pr}}^{(r)}])$。

### 关键设计

1. **跨区域对齐模块**：

    - 使用轻量级线性映射 $M^{(s \to r)}$（初始化为单位矩阵）和深度一维卷积 ConvAlign（初始化为脉冲响应）对不同区域的 shared 潜变量进行时空对齐
    - ConvAlign 为每个 shared 维度维护一个滤波器，容许小的相位偏移以建模跨区域传导延迟
    - 映射是有方向性的（$s \to r$ 与 $r \to s$ 独立学习），不假设对称性，反映了脑区间信号传导的方向性
    - 设计动机：真实神经信号在不同脑区间存在毫秒级的时间偏移和非线性子空间旋转，纯矩阵映射无法建模时间延迟

2. **9 项多目标训练损失**：

    - **重建目标**：$\mathcal{L}_{\text{rec}}$（shared+private 自重建）、$\mathcal{L}_{\text{cross}}$（用另一区域 shared 重建）、$\mathcal{L}_{\text{self}}$（仅用自身 shared 重建），确保 shared 潜变量承载有意义的方差
    - **对齐目标**：$\mathcal{L}_{\text{align}}$ 采用 VICReg 正则化（方差-不变性-协方差）对齐跨区域 shared 潜变量，在保持区域视角差异的同时强制子空间重叠
    - **解缠目标**：$\mathcal{L}_{\text{orth}}$ 惩罚 shared 与 private 间的交叉协方差；方差守卫 $\mathcal{L}_{\text{var-sh}}, \mathcal{L}_{\text{var-pr}}$ 防止退化解（shared 坍缩或 private 消失）
    - **对齐模块正则**：$\mathcal{L}_{\text{mapid}}$ 偏置线性映射趋向单位矩阵、$\mathcal{L}_{\text{align-reg}}$ 正则化 ConvAlign 滤波器趋向脉冲响应，确保可解释性

3. **"训练于基线、推理于扰动"范式**：

    - 仅在 off-stimulation 基线数据上训练，建立内在跨区域协调的参考框架
    - 推理时输入 DBS 刺激条件数据，观察共享潜空间如何被刺激重组，private 潜空间如何反映局部效应
    - 避免了刺激伪影直接污染模型参数，同时利用基线参考框架作为度量重组程度的锚点

## 实验关键数据

### 合成数据验证

三个合成数据集从线性到非线性逐步递增复杂度，每个含 100 trials × 250 时间步（0.5s@500Hz），3 shared + 3 private 维度：

| 数据集 | 混合方式 | 噪声 | SPIRE CCA (shared) | DLAG CCA (shared) |
|--------|----------|------|-------------------|-------------------|
| D0 | 线性 + 高斯噪声 | 高斯 | 与 DLAG 相当 | 线性友好基准 |
| D1 | 非线性扭曲 + 双线性混合 | 1/f + AR(1) | (0.92, 0.91, 0.71) | (0.86, 0.79, 0.60) |
| D2 | D1 + 时变正弦延迟 | 同 D1 | 优于 DLAG | 进一步退化 |

SPIRE 在恢复 private 潜变量方面统计显著优于 DLAG（$p < 0.05$）。在非线性（D1）和时变延迟（D2）条件下，SPIRE 对 shared 潜变量恢复也优于 DLAG。D0 是线性场景，对 DLAG 友好，SPIRE 并无劣势但优势也不显著。

### 人类 DBS 记录数据

- 10 名肌张力障碍儿科患者（5-23岁）的颅内 LFP，电极覆盖 GPi 和 STN，共 17 个半球
- 刺激条件：GPi 85/185/250 Hz、STN 85/185 Hz 及 off-stimulation
- 预处理：双极参考、降采样至 500 Hz、50 Hz 低通 Butterworth 滤波去除高频刺激伪影
- 0.5s 非重叠窗口分段，0-3 阶时间延迟特征增强

### 解缠与重建验证

| 指标 | GPi | STN |
|------|-----|-----|
| Shared GPi/STN CCA 中位数 | ≈1.0 | ≈1.0 |
| Shared-Private CCA 中位数 | 0.55–0.65 | 0.55–0.65 |
| Full (shared+private) 重建 MSE | 0.00211 | 0.000983 |
| 仅 private 重建 MSE | 0.544 | 0.391 |
| 仅 shared（同区域）重建 MSE | 0.0462 | 0.0178 |

解读：shared 子空间跨区域高度一致（CCA≈1.0），而 shared-private 间仅弱相关（0.55-0.65），表明有效解缠。重建分析显示大部分可恢复的神经动态位于 shared 流形中——仅用 private 重建误差增大两个数量级，而仅用 shared 就能获得接近完整重建的效果。

### 刺激频率解码

随机森林从各类潜变量解码刺激频率（GPi 4类 / STN 3类），shared 潜变量在两个解码任务中均显著优于 private 潜变量（$p < 0.001$），且 GPi-shared 与 STN-shared 无显著差异——说明 shared 空间编码了跨区域泛化的刺激特征签名，刺激以频率依赖的方式系统性重组跨区域协调模式。

### 与基线方法对比

| 方法 | 非线性 | Shared/Private 分解 | 适用 LFP | DBS 场景 | 重建 MSE |
|------|--------|---------------------|----------|----------|----------|
| GPFA / CCA | ✗ | ✗ | ✓ | ✗ | — |
| DLAG | ✗ | ✓ | ✗（spike） | ✗（无法收敛） | 数值不稳定 |
| SharedAE | ✓ | 部分（无时间分辨率） | ✗ | ✗ | 高于 SPIRE |
| MMVAE | ✓ | ✗（无显式解缠） | ✗ | ✗ | 高于 SPIRE |
| **SPIRE** | **✓** | **✓** | **✓** | **✓** | **最低** |

DLAG 在真实颅内数据上高斯过程优化数值不稳定无法收敛。SPIRE 在 GPi 和 STN 的重建误差上均显著低于 SharedAE 和 MMVAE。

## 亮点与洞察

- **首个非线性 shared-private 分解框架**用于人类多区域颅内记录，填补了线性模型与真实非线性 LFP 数据间的方法论空白
- **训练范式设计巧妙**：仅在基线数据上训练避免刺激伪影污染模型参数，同时建立了度量刺激引起重组程度的参考锚点。这一"训练于控制、推理于处理"的范式具有广泛迁移价值
- **9 项损失函数各有物理含义**：VICReg 对齐 + 正交解缠 + ConvAlign 时间对齐的组合经过精心设计，方差守卫防止退化解，对齐模块正则确保可解释性
- 首次在儿科 DBS 数据上展示共享潜变量编码频率依赖的网络重组，为 DBS 的分布式网络调制理论提供了定量证据
- shared-private 解缠思想可迁移到多模态 AI 模型中，如视觉-语言模型中分解模态共享与模态专属表示

## 局限与展望

- 仅限较短时间尺度刺激，未验证长期慢性刺激效果及可塑性变化
- 仅使用 LFP 信号，未整合 spike 数据或行为学模态，多模态融合是自然扩展方向
- 缺少概率性目标函数（如 VAE 的 ELBO），无法量化潜变量的不确定性
- 目前仅验证双区域（GPi-STN），尚未扩展到皮层、丘脑等更多区域的多区域场景
- 共享潜变量维度是统计抽象，赋予精确生物物理含义需要互补实验和多模态验证
- 样本量较小（10 名患者，17 个半球），跨病因和年龄段的泛化性有待更大规模数据验证

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个将非线性 shared-private 分解应用于人类颅内 DBS 记录，"训练于基线推理于扰动"范式有新意
- 实验充分度: ⭐⭐⭐⭐ — 合成基准+真实临床数据+多基线对比+解缠验证+解码分析，但样本量偏小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，损失函数定义完整，图表丰富直观
- 价值: ⭐⭐⭐⭐ — 对计算神经科学和 DBS 机制理解有实质贡献，方法可迁移到其他多视角动态系统

<!-- RELATED:START -->

## 相关论文

- [AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/llm_evaluation/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [Disentangling and Integrating Relational and Sensory Information in Transformer Architectures](../../ICML2025/llm_evaluation/disentangling_and_integrating_relational_and_sensory_information_in_transformer_.md)

<!-- RELATED:END -->
