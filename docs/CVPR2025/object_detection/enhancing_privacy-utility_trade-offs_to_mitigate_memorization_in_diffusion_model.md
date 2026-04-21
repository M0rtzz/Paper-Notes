---
title: >-
  [论文解读] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models
description: >-
  [CVPR 2025][目标检测][扩散模型记忆化] 本文提出 PRSS 方法，通过 Prompt Re-anchoring（将记忆化 prompt 重新用作 CFG 的锚点引导生成偏离记忆内容）和 Semantic Prompt Search（用 LLM 搜索语义相似但不触发记忆的替代 prompt）两个策略，在不修改模型和不需要训练数据的推理阶段改进 CFG 方程，实现了扩散模型记忆化缓解中的最优隐私-效用平衡。
tags:
  - CVPR 2025
  - 目标检测
  - 扩散模型记忆化
  - 隐私保护
  - 提示学习
  - 语义提示搜索
  - Classifier-Free Guidance
---

# Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2504.18032](https://arxiv.org/abs/2504.18032)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 扩散模型记忆化, 隐私保护, Prompt重锚定, 语义提示搜索, Classifier-Free Guidance

## 一句话总结
本文提出 PRSS 方法，通过 Prompt Re-anchoring（将记忆化 prompt 重新用作 CFG 的锚点引导生成偏离记忆内容）和 Semantic Prompt Search（用 LLM 搜索语义相似但不触发记忆的替代 prompt）两个策略，在不修改模型和不需要训练数据的推理阶段改进 CFG 方程，实现了扩散模型记忆化缓解中的最优隐私-效用平衡。

## 研究背景与动机

**领域现状**：文本到图像扩散模型（如 Stable Diffusion、Midjourney）能生成高度逼真的图像，但会记忆训练数据——在推理时部分或完整复制训练图像。当训练数据包含版权或敏感内容时，这构成严重的法律和隐私风险。已有多起针对 Stability AI 等公司的诉讼。

**现有痛点**：现有推理阶段缓解策略（如 prompt engineering）面临严重的隐私-效用权衡困难。要提高隐私（降低记忆化风险），必须大幅修改用户 prompt，导致生成结果偏离用户意图（降低效用/文本对齐度）。反之，保持高文本对齐度则无法有效阻止记忆化。训练阶段方法虽然理论可行，但在 LAION5B 全量数据上微调不切实际。

**核心矛盾**：CFG 方程中改善隐私的唯一杠杆是修改 prompt embedding——通过优化 prompt 降低检测信号（magnitude）来减少记忆化概率。但 prompt 修改越大，文本对齐越差。问题的根源在于：(1) 无条件项 $\epsilon_\theta(x_t, e_\phi)$ 作为"锚点"对隐私保护贡献不足；(2) 用梯度优化的 engineered prompt $e^*$ 虽然降低了记忆信号但语义偏离严重。

**本文目标** (1) 找到比 prompt engineering 更高效的隐私增强路径（以更少的效用损失实现相同隐私提升）；(2) 找到保持语义一致性的隐私安全 prompt 替代方案；(3) 两者协同实现不同隐私级别下的最优权衡。

**切入角度**：深入分析 CFG 方程的几何结构——不同 prompt 在嵌入空间中对应不同的 magnitude 等高线，相同等高线上的点隐私相同但效用不同。用记忆化 prompt 重新锚定 CFG 的对比方向可以更高效地引导生成远离记忆路径；用 LLM 在语言空间搜索可以找到语义相近但 magnitude 更低的替代 prompt。

**核心 idea**：用记忆化 prompt 替换 CFG 的无条件锚点（PR）增强隐私，用 LLM 搜索语义等价的低风险 prompt（SS）保障效用，二者协同优化隐私-效用权衡。

## 方法详解

### 整体框架
PRSS 在推理阶段修改 CFG 方程，无需训练或微调。流程：(1) 用户输入 prompt $e_p$；(2) 在第一个去噪步骤 $T-1$ 计算 magnitude $m_{T-1}$，判断是否触发记忆化风险（$m_{T-1} > \lambda$）；(3) 若安全，使用标准 CFG；(4) 若存在风险，首先用 LLM（GPT-4）搜索最多 $n_s=25$ 个语义相似的替代 prompt $e_p^{ss}$（早停于 magnitude < $\lambda$ 的那一个），然后将 CFG 的无条件项替换为原始 prompt $e_p$ 的条件预测（Re-anchoring），最终用 $e_p^{ss}$ 作为目标条件。

### 关键设计

1. **Prompt Re-anchoring (PR)**:

    - 功能：在 CFG 中用更高效的"对比方向"引导生成远离记忆内容，以较低效用成本实现隐私增强
    - 核心思路：标准 CFG 从无条件预测 $\epsilon_\theta(x_t, e_\phi)$ 向条件预测方向引导。PR 将"不希望的生成"定义为记忆化 prompt $e_p$ 的条件预测，用它替换无条件锚点：$\hat{\epsilon} \leftarrow \epsilon_\theta(x_t, e_p) + s(\epsilon_\theta(x_t, e^{ss}_p) - \epsilon_\theta(x_t, e_p))$。这样 CFG 的对比方向从"向任意方向走"变为"专门远离记忆化路径"。几何上，PR 的引导方向直接指向低 magnitude 区域，而标准 CFG 的引导方向是随机的。
    - 设计动机：基线方法需要更多 prompt 优化步骤（修改 prompt 更多）来降低 magnitude，导致大幅偏离用户意图。PR 利用记忆化 prompt 的信息——它精确标记了"应该远离的方向"——因此可以用更少的 prompt 修改达到相同的隐私级别。此外，PR 的效果贯穿整个推理过程，不像基线只在第一步工程 prompt 后就"放手"，避免了后续步骤 magnitude 反弹。

2. **Semantic Prompt Search (SS)**:

    - 功能：在语言空间找到语义相似但记忆化风险更低的 prompt 替代方案，以最小隐私代价最大化效用
    - 核心思路：调用 GPT-4 API 生成最多 $n_s=25$ 个与原始 prompt 语义相似的替代文本。逐个计算每个替代 prompt 的首步 magnitude $m_{T-1}$，一旦找到低于阈值 $\lambda$ 的就采用。如果全部高于 $\lambda$，选择 magnitude 最低的那个。搜索在语言空间而非 embedding 空间进行，天然保持了语义一致性。例如，"The No Limits Business Woman Podcast" → "The Empowered Business Woman's Podcast"，magnitude 从原始 7.48 降至 0.78，但 CLIP 相似度大幅提升。
    - 设计动机：基线的 prompt engineering 在 embedding 空间通过梯度下降优化 prompt embedding，结果虽然降低了 magnitude 但语义偏离严重。SS 利用 LLM 的语言理解能力在意义层面搜索替代方案，保持了用户意图的核心语义。

3. **PR+SS 协同效应**:

    - 功能：解决各自的局限性，实现跨所有隐私级别的最优权衡
    - 核心思路：当 SS 找到完全安全的替代 prompt（magnitude < $\lambda$），单独使用 SS 即可。当 SS 无法将 magnitude 降至阈值以下时，PR 补充提供持续的记忆化偏转。关键在于 SS 先降低了 magnitude 的基线（如从 7.48→6.02），使得 PR 需要做的"偏转工作"更少，从而效用损失更小。最终 CFG 方程为 $\hat{\epsilon} \leftarrow [\text{标准CFG}]\mathbbm{1}_{m<\lambda} + [\epsilon_\theta(x_t, e_p) + s(\epsilon_\theta(x_t, e_p^{ss}) - \epsilon_\theta(x_t, e_p))]\mathbbm{1}_{m>\lambda}$。
    - 设计动机：从误检角度分析——高 $\lambda$ 提高效用但增加 FN（漏检记忆化），PR 缓解 FN 的隐私风险；低 $\lambda$ 提高隐私但增加 FP（误判安全 prompt），SS 缓解 FP 的效用损失。二者互补覆盖所有场景。

### 损失函数 / 训练策略
- 无训练，纯推理阶段方法
- 检测信号可替换：支持原始 magnitude $m_{T-1}$ 和增强的 masked magnitude $m'_{T-1}$
- LLM 搜索成本极低：每个替代 prompt 约 0.9 秒生成，成本约 $0.02
- 完全向后兼容：对安全 prompt（$m_{T-1} < \lambda$）不做任何修改

## 实验关键数据

### 主实验

| 方法 | 检测信号 | 全局记忆化 SSCD↓ | 文本对齐 CLIP↑ | 局部记忆化 SSCD↓ | 文本对齐 CLIP↑ |
|------|---------|----------------|---------------|----------------|---------------|
| PE | $m$ | 0.35 | 0.23 | 0.42 | 0.24 |
| PE | $m'$ | 0.33 | 0.23 | 0.38 | 0.24 |
| **PRSS** | $m$ | **0.22** | **0.27** | **0.36** | **0.26** |
| **PRSS** | $m'$ | **0.18** | **0.28** | **0.33** | **0.27** |

注：数值为从论文图6中近似读取的代表性点，在相同隐私级别(λ)下对比。

### 消融实验

| 配置 | 全局SSCD↓ | CLIP↑ | 说明 |
|------|----------|-------|------|
| 标准 SD | 0.65 | 0.30 | 无缓解 |
| PE (baseline) | 0.35 | 0.23 | 仅 prompt engineering |
| PR only | 0.25 | 0.22 | 高隐私但低效用 |
| SS only | 0.30 | 0.28 | 高效用但隐私不足 |
| **PR+SS** | **0.22** | **0.27** | 最优权衡 |

### 关键发现
- **PR 在全局记忆化上效果突出**：相比基线 PE，PRSS 在全局记忆化场景下的改善远大于局部记忆化。这是因为全局记忆化的"偏转方向"更明确——记忆化 prompt 精确定位了需要远离的全局模式
- **SS 对效用的保持至关重要**：单独使用 PR 虽然隐私最好，但 CLIP 分数下降显著。加入 SS 后效用大幅恢复，且隐私进一步改善
- **PR+SS 的协同效应在局部记忆化上尤为明显**：局部记忆化更难缓解，单独 PR 或 SS 效果有限，但结合使用后显著改善
- **PRSS 可无缝集成更好的检测信号**：从 $m$ 升级到 $m'$ 后 PRSS 性能进一步提升，展现了良好的模块化设计
- 定性案例显示 SS 找到的替代 prompt 在保持语义核心的同时有效规避记忆化触发

## 亮点与洞察
- **对 CFG 方程的深刻几何分析**：将隐私-效用权衡可视化为 magnitude 等高线上的移动，清晰揭示了基线方法低效的原因（沿低效方向移动）和 PR/SS 的改进机制（改变移动方向/起点）。这种分析框架可迁移到其他 CFG 变体的设计
- **"用记忆化 prompt 自身来对抗记忆化"的反直觉设计**：基线丢弃了记忆化 prompt，但 PRSS 保留它作为锚点——因为它精确标记了要远离的方向。这是对 CFG"正-负对比"思想的创造性重用
- **LLM 辅助的语义搜索完全不需要训练数据**：只需 API 调用就能找到安全替代 prompt，隐私友好（不暴露训练集），成本极低。这种"LLM 作为工具"的范式可推广到其他需要语义等价变换的场景
- 方法高度模块化：检测信号、搜索策略、锚定方式都可独立替换升级

## 局限与展望
- **依赖检测准确性**：PRSS 和所有基线一样，当检测信号 $m_{T-1}$ 误判时会退化为标准 SD。检测准确率是整个框架的瓶颈
- **语义搜索的上限**：LLM 生成的 $n_s=25$ 个替代 prompt 可能仍然全部触发记忆化，特别是对于非常独特的概念（如人名、品牌名）。延长搜索或结合 prompt embedding 优化可能有帮助
- 仅在 Stable Diffusion v1-4 上实验，未验证在更新的 SDXL、SD3 等模型上的效果
- 500 个 prompt 的测试集较小，可能无法覆盖所有记忆化类型
- PR 在多次推理时行为的稳定性未充分讨论——不同随机种子下的效果方差如何？

## 相关工作与启发
- **vs PE (Wen et al.)**: PE 仅通过梯度下降优化 prompt embedding 来降低 magnitude，是 PRSS 的直接基线。PRSS 在 PE 基础上改变了 CFG 锚点（PR）和 prompt 搜索空间（SS），实现了全面超越
- **vs BEA (Chen et al.)**: BEA 提出 masked magnitude 作为更好的检测信号和局部记忆化掩码。PRSS 无缝采用了这个信号后进一步刷新 SOTA，展现了 PRSS 作为缓解策略对检测方法的正交性
- **vs Anti-Memorization (Somepalli et al.)**: 该方法在训练数据去重上下功夫，计算代价大且效果有限。PRSS 完全在推理阶段操作，无需接触训练数据
- **vs Negative Prompting**: 负面提示在 SD 推理中常用但是启发式方法。PRSS 的 PR 策略可看作一种理论化的"负面提示"——用记忆化 prompt 作为系统性的负面引导

## 评分
- 新颖性: ⭐⭐⭐⭐ PR 和 SS 分别从隐私和效用两端改进，设计互补且协同。几何分析框架清晰
- 实验充分度: ⭐⭐⭐⭐ 多种检测信号+全局/局部记忆化+消融+定性案例，但测试集较小
- 写作质量: ⭐⭐⭐⭐⭐ 分析深入，图示直观，逻辑推进层层递进
- 价值: ⭐⭐⭐⭐ 对实际部署扩散模型的隐私保护有直接意义，方法简洁可部署

<!-- RELATED:START -->

## 相关论文

- [Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness](verbdiff_text-only_diffusion_models_with_enhanced_interaction_awareness.md)
- [DiffDoctor: Diagnosing Image Diffusion Models Before Treating](../../ICCV2025/object_detection/diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [Implicit Concept Removal of Diffusion Models](../../ECCV2024/object_detection/implicit_concept_removal_of_diffusion_models.md)

<!-- RELATED:END -->
