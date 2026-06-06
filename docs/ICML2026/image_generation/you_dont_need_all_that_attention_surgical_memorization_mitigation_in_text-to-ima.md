---
title: >-
  [论文解读] You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models
description: >-
  [ICML2026][图像生成][文生图扩散模型] 本文提出 GUARD，一个推理时的文生图扩散模型记忆缓解框架，通过对标准 classifier-free guidance 加入“远离原始记忆提示”的 repulsion 和“靠近安全条件预测”的 attraction…
tags:
  - "ICML2026"
  - "图像生成"
  - "文生图扩散模型"
  - "训练数据记忆"
  - "推理时缓解"
  - "注意力机制"
  - "GUARD"
---

# You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models

**会议**: ICML2026  
**arXiv**: [2603.00133](https://arxiv.org/abs/2603.00133)  
**代码**: https://github.com/kairanzhao/GUARD  
**领域**: 图像生成 / 扩散模型 / 记忆缓解  
**关键词**: 文生图扩散模型, 训练数据记忆, 推理时缓解, cross-attention, GUARD  

## 一句话总结
本文提出 GUARD，一个推理时的文生图扩散模型记忆缓解框架，通过对标准 classifier-free guidance 加入“远离原始记忆提示”的 repulsion 和“靠近安全条件预测”的 attraction，并用动态 cross-attention spike 检测与衰减实例化 positive target，在降低训练图像复现的同时尽量保持图像质量和 prompt 对齐。

## 研究背景与动机
**领域现状**：文生图扩散模型已经被证明会在特定 prompt 下复现训练集中某些图像，可能带来隐私泄露和版权风险。已有缓解方法大致分为训练时预防、finetuning/unlearning 和推理时干预三类。

**现有痛点**：训练时方法要求能控制原始训练过程，但很多实际应用直接基于开源或商业预训练模型；unlearning 需要为 forget set 额外微调，成本高且可能不稳；已有推理时方法通常只缩放条件分量、修改初始噪声，或固定处理 EOT/padding 之类 token，难以同时覆盖 verbatim memorization 和 template memorization。

**核心矛盾**：如果强行把 prompt 条件信号压低，确实可能减少复现训练样本，但图像语义和质量会一起掉；如果只做固定 token 的 attention 重分配，又会漏掉不同 prompt 中真正触发记忆的 token。本文要解决的是“推理时精确压制记忆触发因素，同时保留 prompt 相关生成能力”。

**本文目标**：作者希望设计一个不改权重、不重训、不需要提前知道训练过程的 inference-time 方法，能够针对每个 prompt 动态定位记忆相关的 cross-attention spike，并在标准去噪过程里把生成轨迹从记忆样本附近拉开。

**切入角度**：论文重新分析了 memorized vs non-memorized prompts 的 cross-attention 分布，发现 verbatim memorization 中 EOT 常有强 spike，但其他 token 也可能更尖；template memorization 中 EOT 甚至不一定是主要问题。因此固定 attenuate EOT 并不可靠，必须做 per-prompt 统计检测。

**核心 idea**：把推理时引导写成 attractive-repulsive dynamics：repulsion 远离原始记忆 prompt 的条件预测，attraction 则指向一个经过动态 attention spike attenuation 的安全条件预测。

## 方法详解
GUARD 不是训练新 diffusion model，而是在每个 denoising step 改写噪声预测组合方式。它保留 unconditional prediction，额外计算两种 conditional prediction：一种是原始 prompt 的 standard conditional prediction，作为负向目标；另一种是对 prompt-specific attention spikes 做衰减后的 conditional prediction，作为正向目标。

### 整体框架
标准 classifier-free guidance 把噪声预测写成 unconditional prediction 加上 prompt conditional 与 unconditional 的差。GUARD 在这个公式上加入两股力：用权重 $s$ 吸引到正向条件预测 $\epsilon_\theta^+$，用权重 $r$ 排斥原始记忆条件预测 $\epsilon_\theta^-$. 直观上，原始 prompt 若会把轨迹拉向训练图像，repulsion 就把这条方向减掉；但仅仅远离它会损伤质量，所以正向目标需要给出一个仍然 prompt-aligned、但不那么记忆化的替代方向。

本文的具体实例叫 CA-in-GUARD。它先在原始条件分支中读取 cross-attention maps，自动找出当前 prompt 的 spike token 集合 $S(p)$，再在 positive branch 中对这些 token 的 attention logits 做缩放，得到 spike-attenuated conditional prediction。实现上，unconditional、原始 conditional 和 spike-attenuated conditional 可以拼成 batch 做一次 U-Net forward，从而减少额外开销。

### 关键设计
1. **Attractive-repulsive guidance**:

	- 功能：在生成轨迹层面同时减少记忆复现并维持图像质量。
	- 核心思路：将最终噪声预测写成 $\hat{\epsilon}=\epsilon_\theta(x_t,e_\phi)+s(\epsilon_\theta^+(x_t,e_p)-\epsilon_\theta(x_t,e_\phi))-r(\epsilon_\theta^-(x_t,e_p)-\epsilon_\theta(x_t,e_\phi))$。其中 $\epsilon_\theta^-$ 是标准 prompt 条件预测，$\epsilon_\theta^+$ 是经过安全处理的 positive target。
	- 设计动机：单纯削弱 prompt 条件会让生成退化；单纯增加替代目标又可能仍然贴近记忆方向。把 attraction 和 repulsion 分开后，可以独立调节记忆缓解与质量保持。

2. **Per-prompt cross-attention spike detector**:

	- 功能：动态定位每个 prompt 中最可能触发训练图像复现的 token 位置。
	- 核心思路：从原始 conditional pass 的 cross-attention distribution 中，对每个 token 计算跨空间 query 的最大 attention mass $M_i$，再用 $Z_i=(M_i-\mu)/\sigma$ 做 outlier detection，超过阈值 $\tau$ 的 token 被加入 $S(p)$。这个集合可以包含 EOT，也可以包含任意 prompt-specific token。
	- 设计动机：verbatim 和 template memorization 的触发模式不同，固定处理 EOT/padding 既不充分也可能反向伤害。统计 outlier 机制让方法随 prompt 和 denoising step 自适应。

3. **Surgical CA-logit attenuation**:

	- 功能：构造 GUARD 的 positive target，让模型仍听 prompt，但不再过度依赖记忆触发 token。
	- 核心思路：在 selected U-Net cross-attention modules 中，对 $i \in S(p)$ 的 attention logits 做乘性缩放 $\ell'_{q,i}=\ell_{q,i}\cdot\alpha$，再进入 softmax。默认在 down/mid blocks 处理，避免 late up blocks 质量退化；可以选择所有 heads 或 hot heads，论文发现 all-heads 是强而简单的默认。
	- 设计动机：直接把 token 删除或把注意力置零太粗暴，可能破坏语义。logit attenuation 是更细的干预，只压低异常尖峰，保留其他正常 cross-attention pattern。

### 损失函数 / 训练策略
本文没有训练损失，也不更新模型权重。主要超参是 attention spike threshold $\tau$、attenuation factor $\alpha$ 和 GUARD repulsion strength $r$。作者为不同架构和记忆类型做 grid search，并采用质量约束的选择策略：在 CLIP 不超过参考值 15% degradation 的配置中，优先选择 SSCD 更低的设置。

## 实验关键数据

### 主实验
实验使用 Webster 识别的 500 个 memorized prompts，模型包括 SD v1.4 和 SD v2.0；SD v1.4 同时评估 verbatim 与 template memorization，SD v2.0 主要评估 template memorization。记忆程度用 SSCD 衡量，质量用 CLIP 和 FID 衡量。

| 设置 | 方法 | SSCD↓ | CLIP↑ | FID↓ | 说明 |
|------|------|-------|-------|------|------|
| SD v1.4 verbatim | No mitigation | 0.875 | 0.346 | 243.056 | 原模型高度复现训练图 |
| SD v1.4 verbatim | Wen et al. | 0.115 | 0.267 | 162.848 | 强 baseline |
| SD v1.4 verbatim | Ren et al. | 0.113 | 0.258 | 164.638 | 固定 CA 处理方法 |
| SD v1.4 verbatim | CA attenuation | 0.109 | 0.282 | 164.660 | 动态 spike 衰减已超过 Ren |
| SD v1.4 verbatim | CA-in-GUARD | 0.079 | 0.266 | 158.115 | 最低 SSCD，FID 也最好 |
| SD v1.4 template | Han et al. | 0.479 | 0.188 | 210.839 | prior best in this setting |
| SD v1.4 template | CA-in-GUARD | 0.517 | 0.186 | 210.983 | 接近 prior best，整体更稳 |
| SD v2.0 template | Wen et al. | 0.260 | 0.183 | 188.914 | 强 baseline |
| SD v2.0 template | CA attenuation | 0.193 | 0.184 | 245.850 | SSCD 大幅降低但 FID 变差 |
| SD v2.0 template | CA-in-GUARD | 0.193 | 0.183 | 212.727 | 保持低 SSCD，同时缓解 FID 退化 |

### 消融实验
论文的重要分析包括 CA attenuation 单独作用、GUARD 组合后的效果，以及非记忆 prompt 上的副作用。下表整理了最能说明机制的对比。

| 分析项 | 对比 | 关键指标 | 结论 |
|--------|------|----------|------|
| 固定 EOT vs 动态 spike | Ren et al. vs CA attenuation on SD v2.0 template | SSCD 0.356 vs 0.193 | 仅处理 EOT/padding 会漏掉 template memorization 的触发 token |
| Positive target 是否足够 | CA attenuation vs CA-in-GUARD on SD v1.4 verbatim | SSCD 0.109 vs 0.079，FID 164.660 vs 158.115 | repulsion 与 spike-attenuated attraction 有协同作用 |
| 质量-记忆 trade-off | CA attenuation vs CA-in-GUARD on SD v2.0 template | FID 245.850 vs 212.727 | GUARD 能缓解纯 attention 衰减带来的质量损伤 |
| 非记忆 prompt 鲁棒性 | No mitigation vs CA attenuation on SD v1.4 | SSCD 0.071 vs 0.069，CLIP 0.299 vs 0.298 | 对非记忆 prompt 基本无显著负面影响 |
| 非记忆 prompt 鲁棒性 | No mitigation vs CA attenuation on SD v2.0 | SSCD 0.074 vs 0.072，CLIP 0.322 vs 0.320 | 说明方法不一定需要预先知道哪些 prompt 被记忆 |

### 关键发现
- Template memorization 比 verbatim memorization 更难。许多 prior method 在 SD v1.4 verbatim 上有效，但迁到 template 或 SD v2.0 后明显退化。
- CA spike 不等于 EOT spike。论文的 attention 分析解释了为什么 Ren et al. 这类固定 token 规则在 template setting 下会失败。
- CA-in-GUARD 的优势不只是最低 SSCD，而是跨架构、跨记忆类型、跨质量指标更稳定；作者还报告它在 sampler、step count、CFG scale、DINO retrieval metric 和 SD v3.0 上保持较强表现。

## 亮点与洞察
- GUARD 的公式很有解释性：memorized conditional prediction 是负向目标，attenuated conditional prediction 是正向目标。这比“调小某个超参”更清楚地表达了生成轨迹要远离什么、靠近什么。
- 动态 spike detector 把 prior work 中的人工规则变成 prompt-level statistical test，特别适合长尾 prompt 和 template memorization 这种触发 token 不固定的场景。
- 论文对评测协议的修正很重要：只看低 memorization examples 会高估安全性；按 verbatim/template 分开报告，能暴露很多方法的隐藏短板。

## 局限与展望
- GUARD 属于推理时缓解，不能删除模型权重中的记忆信息。白盒攻击者仍可能通过其他方式提取 memorized data。
- 方法需要额外 conditional branch 和 attention hook。虽然作者用 batched forward 降低开销，但部署到高度优化或闭源推理引擎时可能不如标准 CFG 直接。
- Hyperparameter tuning 依赖架构和记忆类型，实际系统中如何自动选择 $\tau,\alpha,r$ 仍需要工程化。
- 当前实验主要围绕 Stable Diffusion 系列和图像相似度指标，未来还需要研究更大多模态生成模型、视频扩散模型以及更严格的版权/隐私风险定义。

## 相关工作与启发
- **vs training-time memorization mitigation**: 训练时方法试图预防记忆写入权重，但要求掌控训练过程；GUARD 接受权重中可能已有记忆，只在推理轨迹上阻止其显现。
- **vs diffusion unlearning**: Unlearning 需要为 forget set 微调且可能不稳定；GUARD 不改权重，适合快速部署，但无法应对白盒提取威胁。
- **vs Ren et al. 的 CA redistribution**: Ren et al. 固定处理 EOT/padding/BOT，GUARD 的 CA-in-GUARD 用 per-prompt spike detection 找真实异常位置，因此对 template memorization 更稳。
- **vs Han et al. 的初始噪声调整**: Han et al. 从 sample-time basin 角度逃离记忆吸引域，GUARD 则直接修改每步条件预测方向；两者可能可以组合。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ GUARD 框架加动态 CA spike attenuation 的组合很清晰，也解释了 prior CA 方法为何不稳。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 SD v1.4/v2.0、verbatim/template、主指标、trade-off、非记忆 prompt 和附加鲁棒性分析。
- 写作质量: ⭐⭐⭐⭐☆ 动机和机制解释充分，评测协议也讲得细；表格和附录较多，完整复现实验细节需要来回查。
- 价值: ⭐⭐⭐⭐☆ 对文生图部署中的版权和隐私风险有直接意义，尤其适合无法重训/微调模型的使用场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](../../CVPR2026/image_generation/low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](../../NeurIPS2025/image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)

</div>

<!-- RELATED:END -->
