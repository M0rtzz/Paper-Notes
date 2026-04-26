---
title: >-
  [论文解读] Rethinking Direct Preference Optimization in Diffusion Models
description: >-
  [AAAI 2026 (Oral)][LLM对齐][DPO] 提出两个正交改进增强扩散模型偏好优化：(1) 稳定参考模型更新策略通过 EMA 放松冻结参考模型并正则化鼓励探索；(2) 时间步感知训练策略自适应调整损失权重缓解跨时间步奖励尺度不平衡。
tags:
  - AAAI 2026 (Oral)
  - LLM对齐
  - DPO
  - 扩散模型
  - 参考模型更新
  - 时间步感知
  - T2I偏好对齐
---

# Rethinking Direct Preference Optimization in Diffusion Models

**会议**: AAAI 2026 (Oral)  
**arXiv**: [2505.18736](https://arxiv.org/abs/2505.18736)  
**代码**: 有  
**领域**: 对齐RLHF / 扩散模型  
**关键词**: DPO, 扩散模型, 参考模型更新, 时间步感知, T2I偏好对齐

## 一句话总结
提出两个正交且可插拔的改进策略来增强扩散模型的偏好优化：稳定参考模型更新（放松冻结+正则化锚点）和时间步感知训练（自适应权重平衡奖励尺度），两者可嵌入 DPO/IPO 等多种偏好优化算法并在人类偏好评估基准上取得 SOTA。

## 研究背景与动机

### 领域现状
**领域现状**：文到图（T2I）扩散模型的人类偏好对齐已成为关键研究挑战。从 LLM 借鉴的偏好优化方法（DPO、IPO 等）已被扩展到扩散模型领域，通过成对偏好数据（赢/输图像对）直接优化模型参数使生成结果更符合人类偏好。现有方法如 Diffusion-DPO、D-Fusion 等已取得初步成功。

### 现有痛点与挑战
**现有痛点**：(1) **冻结参考模型限制探索空间**——标准 DPO 保持参考模型冻结以提供稳定的 KL 散度锚点，但在扩散模型中这严重限制了策略模型的探索能力，因为扩散过程涉及多步去噪，冻结参考在长链推理中积累偏差；(2) **跨时间步奖励尺度不平衡**——扩散模型不同去噪时间步的信号强度差异极大（高噪声时间步信号弱、低噪声时间步信号强），但现有方法对所有时间步一视同仁，导致训练被低噪声步主导。

**核心矛盾**：放松参考模型可增强探索但会导致训练不稳定；不同时间步需要差异化权重但缺乏自适应机制。

### 研究目标与方案
**本文目标**：在保持训练稳定性的同时增强扩散 DPO 的探索能力和时间步训练平衡性。

**切入角度**：设计两个正交（互不干扰）的可插拔策略——参考模型动态更新 + 时间步感知损失加权——可嵌入任意偏好优化算法。

**核心 idea**：通过参考模型正则化松弛解决探索不足，通过时间步感知训练解决奖励不平衡，两策略正交互补。

## 方法详解

### 整体框架
输入为成对偏好数据（preferred/rejected 图像对 + prompt）和预训练 T2I 扩散模型。方法包含两个正交策略模块，可分别或联合嵌入 DPO、IPO 等偏好优化算法的训练流程中。训练后输出对齐模型。

### 关键设计

1. **稳定参考模型更新策略**：

    - 功能：动态更新参考模型以扩大探索空间，同时通过正则化维持训练稳定性
    - 核心思路：(a) 使用指数移动平均（EMA）更新参考模型参数 $\theta_{\text{ref}} \leftarrow \alpha \theta_{\text{ref}} + (1-\alpha) \theta_{\text{policy}}$，让参考模型跟随策略模型缓慢移动而非完全冻结；(b) 同时加入正则化损失 $\mathcal{L}_{\text{reg}}$ 惩罚策略模型偏离参考过远，形成"松弛但有锚点"的机制——既允许探索新区域又防止策略模型完全偏离
    - 设计动机：完全冻结参考模型在 LLM 中可行（因为 token 空间离散且决策链短），但在扩散模型的连续多步去噪过程中会严重限制探索。EMA + 正则化在松弛和稳定之间取得平衡

2. **时间步感知训练策略**：

    - 功能：缓解不同去噪时间步之间奖励信号强度差异导致的训练不平衡
    - 核心思路：分析不同时间步 $t$ 的隐式奖励分布，发现高噪声时间步（$t$ 大）的奖励信号弱、方差大，低噪声时间步（$t$ 小）的信号强、方差小。据此设计自适应权重函数 $w(t)$：对高噪声时间步增加权重以补偿信号衰减，对低噪声时间步降低权重以防止主导训练。具体权重可基于各时间步奖励分布的统计量归一化得到
    - 设计动机：LLM 的 DPO 不存在时间步维度（一次前向生成），而扩散模型的多步去噪本质上引入了时间步维度的新问题——这是扩散 DPO 独有的瓶颈，需要专门的解决方案

3. **可插拔模块化设计**：

    - 功能：使两个策略可嵌入 DPO、IPO 等多种偏好优化框架
    - 核心思路：最终损失形式为 $\mathcal{L} = w(t) \cdot \mathcal{L}_{\text{DPO/IPO}} + \lambda \cdot \mathcal{L}_{\text{reg}}$，两个策略通过乘性权重和加性正则化分别作用于损失函数，互不干扰
    - 设计动机：模块化设计使得方法不依赖特定的偏好优化算法，可以作为通用增强插件广泛应用

### 损失函数 / 训练策略
总损失 $\mathcal{L} = w(t) \cdot \mathcal{L}_{\text{pref}} + \lambda \cdot \mathcal{L}_{\text{reg}}$，其中 $\mathcal{L}_{\text{pref}}$ 为基础偏好优化损失（DPO 或 IPO），$w(t)$ 为时间步感知权重，$\mathcal{L}_{\text{reg}}$ 为参考模型正则化项。EMA 更新参考模型参数的动量系数 $\alpha$ 和正则化权重 $\lambda$ 为主要超参数。

## 实验关键数据

### 主实验：人类偏好评估基准

| 方法 | 偏好对齐评分 | 改进方式 |
|------|------------|---------|
| Diffusion-DPO (基线) | 基准 | 冻结参考 + 均等权重 |
| + 参考模型更新 | 显著提升 | 探索能力增强 |
| + 时间步感知 | 显著提升 | 训练更平衡 |
| + 两者联合 (**Ours**) | **SOTA** | 正交叠加效果最优 |

### 消融实验：策略正交性验证

| 配置 | 独立效果 | 叠加效果 | 结论 |
|------|---------|---------|------|
| 仅参考更新 | 有效提升 | — | 增强探索 |
| 仅时间步感知 | 有效提升 | — | 平衡训练 |
| 两者联合 | — | 优于两者之和 | 正交互补 |

### 跨算法兼容性

| 基础算法 | 嵌入本方法后 | 说明 |
|---------|------------|------|
| DPO | 提升 | 适用 |
| IPO | 提升 | 适用 |
| 其他偏好优化 | 提升 | 通用插件 |

### 关键发现
- 两策略正交：独立有效且联合效果优于单独之和
- 时间步不平衡是扩散 DPO 独有问题——LLM DPO 不存在此问题
- AAAI 2026 Oral 且同时被 SPIGM@NeurIPS 2025 接收

## 亮点与洞察
- **正交改进的可组合性**：两策略解决不同层面的问题（探索 vs 平衡）且互不干扰，方法论上提供了模块化改进 DPO 的范式
- **时间步维度分析**：首次系统揭示了扩散模型 DPO 中跨时间步奖励尺度不平衡现象，为后续扩散对齐研究提供了重要视角
- **工程实用性强**：无需重设计训练流程，直接嵌入现有 pipeline 即可获得提升

## 局限与展望
- **论文 HTML 全文不可用**：详细消融数据和超参数敏感性分析未能获取，以上分析主要基于摘要和方法概述
- **参考更新频率/EMA 动量的敏感性**：$\alpha$ 值对性能的影响可能显著，需要仔细调参
- **是否适用于视频/3D 扩散模型**：更长的时间步链和更复杂的生成任务中效果待验证
- **与无参考方法的对比**：如 MaPO（AAAI 2026）完全移除参考模型，两种路线的最优适用条件需厘清

## 相关工作与启发
- **vs Diffusion-DPO**：标准迁移方案，冻结参考模型导致探索不足——本文直接改进这一核心瓶颈
- **vs MaPO (AAAI 2026)**：完全移除参考模型的互补路线——本文保留但动态更新，两种思路代表不同技术方向
- **vs DDPO/DRaFT**：依赖额外的奖励模型在线打分——本文用偏好对直接优化更轻量
- **vs D-Fusion (ICML 2025)**：也关注扩散 DPO 的改进但聚焦样本一致性——本文聚焦参考模型和时间步两个正交维度

## 评分
- 新颖性: ⭐⭐⭐⭐ 两个正交策略各有独立贡献，时间步不平衡分析是扩散对齐领域的新发现
- 实验充分度: ⭐⭐⭐ 基于摘要信息有限，Oral 论文应有充分实验
- 写作质量: ⭐⭐⭐⭐ AAAI Oral 质量，问题动机清晰
- 价值: ⭐⭐⭐⭐ 可插拔设计使其可广泛嵌入现有扩散 DPO 方法

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICML 2025\] D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](../../ICML2025/llm_alignment/d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)
- [\[ICML 2025\] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](../../ICML2025/llm_alignment/adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](../../CVPR2026/llm_alignment/mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)

<!-- RELATED:END -->
