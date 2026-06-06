---
title: >-
  [论文解读] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection
description: >-
  [ICML 2026][LLM安全][deepfake detection] 作者用一个 115k 带推理标注的 DF-R5 数据集 + 把 CLIP ViT 换成 ConvNeXT 的 DX-LLaVA 架构，并提出 PRPO —— 段落级别 GRPO 变体…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "deepfake detection"
  - "GRPO"
  - "paragraph-level reward"
  - "visual grounding"
  - "MLLM reasoning"
---

# PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection

**会议**: ICML 2026  
**arXiv**: [2509.26272](https://arxiv.org/abs/2509.26272)  
**代码**: https://github.com/tuanrpt/PRPO (有)  
**领域**: AI 安全 / 多模态 VLM / 深度伪造检测 / RLHF  
**关键词**: deepfake detection, GRPO, paragraph-level reward, visual grounding, MLLM reasoning

## 一句话总结
作者用一个 115k 带推理标注的 DF-R5 数据集 + 把 CLIP ViT 换成 ConvNeXT 的 DX-LLaVA 架构，并提出 PRPO —— 段落级别 GRPO 变体，每段以 CLIP-文本-图像相似度（VCR）+ 推理-结论多数票一致性（PCR）为 reward，把跨域 deepfake 检测 F1 从 SOTA 75.26% 推到 89.91%，推理质量从 4.2/5 提到 4.55/5。

## 研究背景与动机
**领域现状**：深度伪造图像随扩散模型 / GAN 几乎与真图无差，二分类检测器（CLIP-ViT、ConvNeXT、频域特征）虽强，但完全不可解释。MLLM（LLaVA、GPT-4o、Gemini）有强推理能力，理论上能给出"为何这张图是 fake"的依据，但实际检测准确率反而很差。

**现有痛点**：(1) Data 稀缺 —— 现有 deepfake 数据集几乎没有高质量推理标注，直接用 QA 蒸馏只能学到「短答案」式预测；(2) 架构问题 —— LLaVA 用 CLIP ViT 抓全局语义，对 deepfake 关键的局部高频纹理（毛发、毛孔、背景不连续）不敏感；(3) 推理质量 —— MLLM 频繁「先下结论再补理由」，导致结论与图像证据脱钩，甚至幻觉描述图中根本不存在的瑕疵。

**核心矛盾**：RL 算法（GRPO / PPO）的 reward 普遍只盯最终标签，而 deepfake reasoning 是「分段描述多个伪造线索 + 综合结论」的结构化文本，token-level / sequence-level reward 既给不出段间一致性信号，也无法直接逼模型把每段对齐到图像证据上。

**本文目标**：(1) 造一份大规模带高质量段落式推理标注的数据集；(2) 用一个对局部纹理敏感的 backbone 做监督微调；(3) 设计一种段落级 RL，让模型在推理过程中持续视觉对齐、保持段间结论一致，且能在 test-time 用无标签 reward 持续自改进。

**切入角度**：从「推理本身就是分段」的天然结构出发，把每段当作 RL 中独立的 trajectory unit，分别奖励其与图像的视觉一致性（VCR）和与最终结论的语义一致性（PCR），用 GRPO 的 group-relative advantage 来加权学习。

**核心 idea**：把 GRPO 的 token-level advantage 提升为「段落级」，每段 reward 由 frozen CLIP-ConvNeXT 计算的图文相似度 + 段间多数票一致性合成，使推理段落必须既能描述图中真实证据、又能彼此自洽。

## 方法详解

### 整体框架
三阶段：(1) **DF-R5 数据合成** —— 用 4 个 MLLM 池化 200 个候选 deepfake 特征 → Gemini 对每图打分 → 把分数聚成 ≤7 个语义组 → 生成 115k 段落式推理；(2) **DX-LLaVA 微调** —— 把 LLaVA 的 CLIP ViT 换成 CLIP ConvNeXT Stage-3 输出（10×10 像素级特征展平为 100 token），projector + Vicuna + 顶部加二分类 head 共同训练，损失为 $\mathcal L_{\text{lm}}+\alpha\mathcal L_{\text{binary}}$（$\alpha=10$）；(3) **PRPO test-time RL** —— 对每张图 sample $L$ 条完整 reasoning，按段切成 $\{p_1^{(i)},\dots,p_{M_i+1}^{(i)}\}$（最后一段是 final answer），每段算 reward $R(p_j^{(i)})=\tfrac12(R_{\text{VCR}}+R_{\text{PCR}})$，组内归一化得 advantage $A_j^{(i)}$，按 PPO-clip 形式更新策略 + KL 正则。

### 关键设计

1. **Visual Consistency Reward (VCR)**:

    - 功能：让每段推理文字必须真的描述图像里实际存在的视觉证据，杜绝幻觉。
    - 核心思路：用 YAKE 无监督关键词抽取从段落 $p_j^{(i)}$ 抽出 $s_j^{(i)}$，喂入 frozen CLIP-ConvNeXT 的 text encoder，与图像 encoder 输出算 cosine：$R_{\text{VCR}}(p_j^{(i)})=\tfrac12[\text{sim}(\text{CLIP}_{\text{txt}}(s_j^{(i)}),\text{CLIP}_{\text{img}}(x))+1]\in[0,1]$。
    - 设计动机：直接把整段塞 CLIP 会超长且语义被稀释；YAKE 抽词后再算相似度既符合 CLIP 输入限制，又把信号集中在「这段提到的具体特征」上。复用同一个 ConvNeXT（架构里已用）做 reward 模型省去外部模型与额外算力。

2. **Prediction Consistency Reward (PCR)**:

    - 功能：保证最终结论与多数推理段落一致，缓解「证据指向 fake 但结论说 real」的内部矛盾。
    - 核心思路：用预定义词表 $\mathcal F$（unnatural、inconsistent…）/ $\mathcal R$（authentic、natural…）/ $\mathcal N$（no、not…）规则化为段级标签 $\hat y(p_j^{(i)})$；中间段 reward 恒为 1（默认与图一致），final 段 reward 为 $\mathbb I[\hat y(p_{M_i+1}^{(i)})=\hat y_{\text{maj}}^{(i)}]$，其中 $\hat y_{\text{maj}}^{(i)}$ 是前面所有段的多数投票。
    - 设计动机：在 deepfake 推理这种「无 step-wise gold」的场景里没法借鉴数学推理的 process reward；用模型自身段间一致性作为 label-free signal 又能避免外部模型/标注成本，刚好对应 test-time RL 的需求。

3. **段落级 GRPO 损失（PRPO）**:

    - 功能：把 GRPO 从 token / sequence 粒度升级为段落粒度，让每段独立得到 advantage 并被 PPO-clip 加权更新。
    - 核心思路：对组 $\mathcal O=\{o^{(1)},\dots,o^{(L)}\}$ 内所有段统一算均值方差 $\mu_R,\sigma_R$，归一化 $A_j^{(i)}=(R(p_j^{(i)})-\mu_R)/(\sigma_R+\epsilon)$；策略比 $r_j^{(i)}=\pi_\theta(p_j^{(i)}|v,z)/\pi_{\text{old}}(p_j^{(i)}|v,z)$；损失 $\mathcal L_{\text{PRPO}}=\mathbb E\sum_{i,j}\min(r_j^{(i)}A_j^{(i)},\text{clip}(r_j^{(i)},1-\epsilon,1+\epsilon)A_j^{(i)})$；额外加段级 KL 项 $\mathcal L_{\text{KL}}$ 对齐 reference model；总目标 $\max_\theta\mathcal J=\mathcal L_{\text{PRPO}}-\beta\mathcal L_{\text{KL}}$（$\beta=0.01$）。
    - 设计动机：token-level GRPO 让一整条 reasoning 共享同一 advantage，导致少数「优秀段落」与「错段落」同奖同罚；段落级让奖励信号精确落到对应文字上，且组内归一化避免了 reward 数值漂移。

### 损失函数 / 训练策略
微调阶段 $\mathcal L_{\text{total}}=\mathcal L_{\text{lm}}+\alpha\mathcal L_{\text{binary}}$（$\alpha=10$，binary head 用 GAP 后线性分类）；PRPO 阶段 $\mathcal J=\mathcal L_{\text{PRPO}}-\beta\mathcal L_{\text{KL}}$（$\beta=0.01$）。LR：微调 $2\times 10^{-5}$，PRPO $3\times 10^{-7}$。CLIP ConvNeXT 冻结，仅微调 projector + Vicuna + 分类头。8 卡 H200，verl 框架。

## 实验关键数据

### 主实验
在 DF-40 上做 leave-one-domain-out 跨域测试（训 4 域，测第 5 域），F1：

| 方法 | →DDIM | →PixArt | →SD | →SiT | →StyleGAN | 平均 |
|------|------|--------|----|-----|-----------|------|
| LLaVA | 49.86 | 65.46 | 26.54 | 15.36 | 57.03 | 42.85 |
| DE-FAKE | 8.83 | 86.45 | 95.80 | 4.55 | 76.50 | 54.43 |
| FakeShield | 31.84 | 88.57 | 92.28 | 33.22 | 98.70 | 68.92 |
| UnivCLIP | 74.85 | 89.31 | 74.81 | 40.01 | 86.46 | 73.09 |
| SIDA | 70.07 | 73.86 | 92.37 | 46.53 | 94.98 | 75.26 |
| DX-LLaVA (ours, SFT) | 92.34 | 83.11 | 89.35 | 26.46 | 99.13 | 78.08 |
| **PRPO (ours, RL)** | **95.88** | **88.10** | **94.99** | **71.26** | **99.32** | **89.91** |

跨域平均 F1 较 SIDA 提升 14.65 pp，最难的 SiT 域上跃升 24.7 pp（46.53→71.26）。

### 消融实验

| 配置 | F1 / 关键指标 | 说明 |
|------|--------------|------|
| LLaVA + $\mathcal L_{\text{lm}}$（仅语言损失） | 35.82 (inter-domain) | 高 precision 低 recall，模型全部猜 real |
| LLaVA + $\mathcal L_{\text{lm}}+\alpha\mathcal L_{\text{binary}}$ | 61.66 | binary head 显著增强判别 |
| 换成 ConvNeXT backbone（DX-LLaVA） | 78.08 | 局部纹理优势 |
| + PRPO | **89.91** | 段落级 RL 进一步把推理与图像锁紧 |
| 推理质量评分 (Gemini judge) | 4.55/5（PRPO） vs 4.20/5（Gemini-2.5） | 首次反超教师模型 |

### 关键发现
- PRPO 在最难、几乎不可区分的 SiT 域增益最大，说明段落级 reward 真的把模型从「靠 backbone 区分纹理」拉到「靠系统化推理多线索」上。
- 仅靠 SFT + ConvNeXT 不够 —— 必须有 RL 才能从 78→89。
- PRPO 用纯 label-free reward（CLIP 相似度 + 多数票），却带来比传统监督 baseline 大幅的下游收益，说明在 test-time 持续自一致 + 自对齐有效。
- 推理质量分（4.55）首次超过 Gemini-2.5（4.20），表明结构化 reward 比单纯 scaling 更能改善 explanation 质量。

## 亮点与洞察
- 把 reward 粒度从 token 推到「段落 = 一个语义单元」，是 RLHF / GRPO 框架在长结构化推理上的自然延伸 —— 同套思路可迁移到法律文书、医学报告、code review 等任何「分段、内部要自洽」的任务。
- VCR 用现成 frozen CLIP 当 reward model，避免训练 reward model 的成本与不稳定；PCR 用预定义词典 + 多数票当 prediction 一致性信号，整套 reward 几乎"零成本"，对 test-time RL 极友好。
- 在 LLaVA 这种 OSS 模型上叠 RL 微调反超 GPT-4o / Gemini-2.5 这种闭源模型，说明在垂直任务上「合适的 reward 结构 + RL」性价比远高于堆参数。

## 局限与展望
- 跨域评测仅覆盖 5 个生成器域（DDIM / PixArt / SD / SiT / StyleGAN），最新模型如 SD-3 / Flux / Sora / 视频 deepfake 没覆盖，泛化能力未知。
- PCR 依赖人手设计的关键词典 $\mathcal F/\mathcal R/\mathcal N$，对不同语言、不同伪造类型可能需要重新设计；多数票规则在「全部段都错」时仍能给高一致性 reward，存在隐患。
- VCR 用 CLIP-ConvNeXT 当 judge，本质把检测器自己当 reward —— 与训练目标耦合，可能放大 backbone 的固有偏置（reward hacking 风险）。
- 没有覆盖视频 / 音频 deepfake，也未讨论对抗扰动下 reward 的鲁棒性。

## 相关工作与启发
- **vs GRPO (Shao et al. 2024)**：GRPO 在 group 内归一化 token-level advantage；PRPO 把粒度提到段落，更适配长结构化推理。
- **vs TTRL (Zuo et al. 2026) / self-certainty reward (Zhao et al. 2026)**：同样 label-free，但 TTRL 用整体 majority vote 当 reward，PRPO 进一步细化到段落 × 视觉一致性，信号更密。
- **vs SIDA / FakeShield**：传统 deepfake 检测以二分类 + 局部特征为主；PRPO 用「推理 + 反思」结构同时拉高检测精度与可解释性。

## 评分
- 新颖性: ⭐⭐⭐⭐ PRPO 把 reward 粒度提到段落 + 全 label-free reward，是对 GRPO 一族的实用改造
- 实验充分度: ⭐⭐⭐⭐ 5 域 leave-one-out + 多个 MLLM 基线 + 推理质量评分 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 三阶段 pipeline 讲得清晰，公式与算法位置合理
- 价值: ⭐⭐⭐⭐ 给"可解释 deepfake 检测"这一安全关键任务给出 SOTA 同时具备明显工程可复制性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](../../ICLR2026/llm_safety/reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/llm_safety/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[ICML 2025\] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](../../ICML2025/llm_safety/unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)
- [\[CVPR 2026\] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](../../CVPR2026/llm_safety/tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)

</div>

<!-- RELATED:END -->
