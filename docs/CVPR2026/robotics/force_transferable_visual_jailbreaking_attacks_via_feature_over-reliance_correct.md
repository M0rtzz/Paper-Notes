---
title: >-
  [论文解读] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction
description: >-
  [CVPR 2026][机器人][视觉越狱攻击] 通过分析视觉越狱攻击在层特征和频谱域的过度依赖问题，提出FORCE方法纠正非泛化性特征依赖，引导攻击探索更平坦的损失景观，从而显著提升跨模型迁移性。
tags:
  - CVPR 2026
  - 机器人
  - 视觉越狱攻击
  - 多模态大语言模型
  - 对抗可迁移性
  - 频谱分析
  - 损失景观
---

# FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction

**会议**: CVPR 2026  
**arXiv**: [2509.21029](https://arxiv.org/abs/2509.21029)  
**代码**: [GitHub](https://github.com/tmllab/2026_CVPR_FORCE)  
**领域**: 机器人/AI安全  
**关键词**: 视觉越狱攻击, 多模态大语言模型, 对抗可迁移性, 频谱分析, 损失景观

## 一句话总结

通过分析视觉越狱攻击在层特征和频谱域的过度依赖问题，提出FORCE方法纠正非泛化性特征依赖，引导攻击探索更平坦的损失景观，从而显著提升跨模型迁移性。

## 研究背景与动机

多模态大语言模型（MLLMs）虽然在文本安全对齐方面表现出强大的防护，但新引入的视觉模态带来了额外的脆弱性。基于优化的视觉越狱攻击（如PGD）能够可靠地突破开源MLLM的安全护栏，但其跨模型迁移性极为有限——在源模型上几乎100%成功的攻击，迁移到目标模型后往往大幅失败。

这一限制的核心原因在于：
- **损失景观尖锐**：生成的攻击位于高锐度区域，即使微小的参数变化也会导致损失急剧上升，攻击失效
- **模型特异性特征依赖**：攻击过度依赖源模型特有的特征，无法泛化到参数不同的目标模型
- 这使得视觉越狱攻击无法有效评估闭源商业MLLM的安全性

## 方法详解

### 整体框架

FORCE（Feature Over-Reliance CorrEction）在标准PGD攻击框架基础上，增加两个正则化组件：**层特征正则化**和**频谱特征重缩放**，分别纠正中间层和频谱域的非泛化性特征依赖，引导攻击探索更平坦的可行域。

### 关键设计

1. **损失景观分析**：作者首先可视化了视觉越狱攻击的输入和权重损失景观。发现攻击陷入源模型的局部最优，仅0.03像素扰动或0.0002权重扰动即可使攻击失效。这种尖锐景观说明攻击过度依赖模型特异性特征。

2. **层特征正则化（Layer-aware Regularisation）**：分析发现浅层特征的可行域极为狭窄（第11层需保留>90%对抗特征才能成功），而深层特征更具鲁棒性。因此在浅层施加更强的正则化，鼓励攻击在邻域 $\eta$ 内采样参考点，最大化其与攻击样本的层特征 $L_2$ 距离：

$$d_l = \|f_{\theta,l}(\mathbf{x}_{\text{img}}+\delta, \mathbf{x}_{\text{txt}}) - f_{\theta,l}(\mathbf{x}_{\text{img}}+\delta+\eta, \mathbf{x}_{\text{txt}})\|_2^2$$

正则化强度随层深度递减：$\lambda_l = \lambda \cdot \max(1-(2l/L)^2, 0)$，确保浅层得到更强约束。

3. **频谱特征重缩放（Spectral Feature Rescaling）**：分析发现随着优化进行，攻击越来越依赖高频成分（语义贫乏），偏离了自然图像中低频语义主导的分布。对频谱带进行自适应缩放：

$$w_m = \min\left(\beta, \frac{\ell_{m-1}}{\ell_m} \cdot \beta\right)$$

当高频带影响超过相邻低频带的 $\beta$ 倍时进行抑制，然后通过FFT/IFFT重建扰动 $\delta_{\text{rescaled}} = \text{IFFT}((A \odot S) \odot e^{i\Phi})$。

### 损失函数 / 训练策略

总体优化目标结合原始越狱损失和层正则化损失：

$$\ell_{\text{reg}} = \frac{1}{N}\sum_{n=1}^{N}\sum_{l=1}^{L} \lambda_l \cdot \frac{\ell_{\text{ref}}}{d_l}$$

其中 $\ell_{\text{ref}}$ 确保参考样本仍在可行域内，$d_l$ 鼓励扩大层表征区域。两组件集成到标准PGD流程中：先频谱重缩放，再层特征探索。

## 实验关键数据

### 主实验

源模型为LLaVA-v1.5-7B，迁移到多个黑盒目标模型（MaliciousInstruct数据集）：

| 目标模型类型 | 目标模型 | 指标(ASR↑) | PGD | FORCE | 提升 |
|---|---|---|---|---|---|
| Adapter-Based | LLaVA-v1.6-mistral-7b | ASR | 61.00% | 69.00% | +12.3% |
| Adapter-Based | InstructBLIP-Vicuna-7B | ASR | 84.00% | 92.00% | +9.5% |
| Adapter-Based | Idefics3-8B-Llama3 | ASR | 53.00% | 64.00% | +20.8% |
| Early-Fusion | Qwen2.5-VL-7B | ASR | 5.00% | 11.00% | +120% |
| Early-Fusion | LLaMA-3.2-11B | ASR | 1.00% | 2.00% | +100% |
| 商业模型 | GPT-5 | ASR | 1.00% | 2.00% | +100% |
| 商业模型 | Gemini-2.5-Pro | ASR | 10.00% | 10.00% | +0% |

### 消融实验

| 配置 | ASR↑ | Query↓ | 说明 |
|---|---|---|---|
| Baseline (PGD) | 53.00% | 50.73 | 无任何正则化 |
| + Layer Feature | 55.00% | 48.46 | 层特征正则化贡献+3.8% |
| + Frequency Feature | 59.00% | 44.03 | 频谱重缩放贡献+11.3% |
| FORCE (Both) | 64.00% | 39.59 | 协同效应，总提升+20.6% |

### 关键发现

- 频谱重缩放（+11.3%）比层正则化（+3.8%）贡献更大，说明高频过度依赖是迁移性差的主要瓶颈
- 对early-fusion架构（如Qwen2.5-VL、LLaMA-3.2），基线攻击几乎完全失效（93%失败率），FORCE实现约100%的相对提升
- 零样本场景下FORCE仍有效，在Gemini-2.5-Pro上从1%提升到3%
- 可视化验证FORCE确实扩大了浅层可行域并恢复了自然频谱分布

## 亮点与洞察

- **深刻的分析框架**：从损失景观→层特征→频谱域三个维度系统剖析了视觉攻击迁移性差的原因
- **发现"浅层脆弱性"**：越靠近浅层，攻击可行域越窄，这为理解MLLM视觉编码器的安全机制提供了新视角
- **频谱视角的创新**：揭示攻击优化过程中高频依赖的反常增长趋势，并通过简洁的频谱重缩放有效修正
- **实用性**：无需访问目标模型参数，可直接用于闭源MLLM的红队评估

## 局限与展望

- 对闭源商业模型（GPT-5、Claude-Sonnet-4）绝对ASR仍然很低（1-2%），实际威胁有限
- 仅在LLaVA-v1.5-7B作为源模型进行评估，更大/更多源模型组合值得探索
- Early-fusion架构的安全护栏明显更强，针对其特殊结构可能需要更专门的迁移策略
- 方法增加了频谱分析和多参考点采样的计算开销

## 相关工作与启发

- 与分类任务中的对抗迁移性研究（SAM、频谱方法）形成对照，将这些思路引入LLM安全领域
- 对MLLM安全评估具有直接意义：当前视觉安全对齐远弱于文本对齐
- 层特征分析的思路可启发防御方法：加强浅层对抗鲁棒性可能是有效的防御策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次从层特征和频谱双视角系统分析视觉越狱攻击迁移性
- **实验充分度**: ⭐⭐⭐⭐ 覆盖3类架构8个模型3个数据集，消融完整
- **写作质量**: ⭐⭐⭐⭐ 分析清晰，可视化丰富，逻辑链完整
- **价值**: ⭐⭐⭐⭐ 对MLLM红队评估和安全研究有重要参考价值

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)
- [Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)
- [The Coherence Trap: MLLM-Crafted Narratives Exploit Manipulated Visual Contexts](the_coherence_trap_when_mllmcrafted_narratives_exp.md)

<!-- RELATED:END -->
