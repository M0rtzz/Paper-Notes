---
title: >-
  [论文解读] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack
description: >-
  [ICML 2026][LLM安全][多模态红队] 本文把 T2I 模型的整个去噪轨迹本身当成 VLM 红队攻击的"攻击面"，用一个 high-level prompt editor + low-level GRPO 微调 rectified-flow 模型的分层 RL 框架（STARE）…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "多模态红队"
  - "扩散轨迹攻击"
  - "分层 RL"
  - "GRPO"
  - "时间对齐分析"
---

# STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack

**会议**: ICML 2026  
**arXiv**: [2605.00699](https://arxiv.org/abs/2605.00699)  
**代码**: https://github.com/henrymao2004/STARE.git (有)  
**领域**: 图像生成 / 多模态 VLM 安全 / 红队攻击  
**关键词**: 多模态红队, 扩散轨迹攻击, 分层 RL, GRPO, 时间对齐分析

## 一句话总结
本文把 T2I 模型的整个去噪轨迹本身当成 VLM 红队攻击的"攻击面"，用一个 high-level prompt editor + low-level GRPO 微调 rectified-flow 模型的分层 RL 框架（STARE），不仅把 attack success rate 比 SOTA 提升 68%，更揭示了一个全新现象——Optimization-Induced Phase Alignment：对抗优化会自动把"概念性毒性"绑到去噪早期、"细节性毒性"绑到后期，从而把混沌的毒性形成过程变成几个可预测的"漏洞时间窗"。

## 研究背景与动机

**领域现状**：VLM 的 toxic continuation 攻击是当前最阴险的多模态安全威胁——攻击者用 T2I 模型生成对抗图像，配上一个文本前缀让 VLM 生成高度毒性的续写。现有红队方法（PGJ、DiffZOO、ART、RedDiffuser 等）几乎都把 T2I 当黑盒——只看 terminal toxicity 分数，不管 toxic semantics 是在哪一步出现的。

**现有痛点**：terminal-only 视角带来"时间不透明"问题。Diffusion 模型本身有从粗到细的语义涌现机制（早期定 layout/concept、后期定细节），但现有红队完全忽视这个时间结构，导致 sparse global reward 给不出 attribution——既不知道"为什么这张对抗图能 jailbreak"，也无法在 defense 上做精确干预。

**核心矛盾**：(1) 黑盒优化 vs 白盒攻击面：把 T2I 当黑盒只能拿到最终 toxicity，但 diffusion 模型的中间步骤明明有可利用的语义涌现规律；(2) 平铺式 RL vs 分层语义结构：标准 RL（如 DDPO）把整个生成视为单一 policy，无法对应"早期 layout / 晚期细节"的天然分工；(3) 概念毒性 vs 细节毒性：现实毒性有 identity/threat 这种"概念级"的（要早期种子）和 obscene/insult 这种"细节级"的（要后期放大），但 baseline 只能均匀施压。

**本文目标**：(1) 设计一个能 explicitly 操纵去噪轨迹早晚两个阶段的分层 RL 框架，对 VLM 做端到端 toxicity attack；(2) 通过时间对齐分析揭示对抗优化对 diffusion 时间结构的影响；(3) 把 ASR 推到 SOTA。

**切入角度**：作者用 rectified flow 作为基底（其 velocity field 显式且轨迹近直线，便于做时间归因分析）。然后把"prompt editing 设语义子目标"和"velocity field 微调放大细节"分别绑到 high-level / low-level 两个 MDP——这种分层结构天然对应 diffusion 的早晚两阶段语义涌现。

**核心 idea**：用 high-level prompt editor 在 embedding 空间种"概念毒性子目标"，用 low-level GRPO 微调 rectified-flow velocity field 放大"细节毒性"，两个 policy 共享同一个 toxicity reward；时间归因分析（MLMC + 块化扰动）证明这个分层结构对应到真实的早晚漏洞窗。

## 方法详解

### 整体框架

输入：root prompt $p$、白盒 T2I 模型（SD 3.5-Medium + LoRA $r=16$）、查询级黑盒 VLM（LLaVA-v1.6-mistral-7b）。流程：(1) High-level 把 $p$ embedding 加噪扰动得到 $K$ 个候选编辑 $e_p + \delta_j$，经 vec2text 解码成 $K$ 个 subgoal prompt $p'^{(j)}$；(2) Low-level 对每个 $p'^{(j)}$ 用当前 LoRA-augmented velocity field $v_\theta$ 跑 $M$ 个 image rollout（用 Marginal-Preserving Stochastic SDE 离散化加噪保探索）；(3) VLM 对每张图 + 续写 prompt 给 toxicity score，加 CLIPScore 的对齐奖励合成 terminal reward；(4) 两个 policy 都用 GRPO 目标更新（group 归一化优势），high-level group 是 $K$ 个候选的平均 reward + edit reward，low-level group 是全部 $K \times M$ 个 rollout 的 individual reward。整个 pipeline 形成"语义子目标→图像生成→VLM 续写→toxicity→反传到两个 policy"的双层闭环。

### 关键设计

1. **High-Level Prompt Editor + Low-Level Velocity Fine-tuning 的分层 MDP**：

    - 功能：把语义注入和细节放大分到两个不同时间尺度的 policy，对应 diffusion 早期概念种子与晚期细节细化。
    - 核心思路：High-level MDP 是一个 single-step decision——state 是 prompt embedding $e_p$，action 是 edit vector $\delta$，policy $\pi_{edit}(\delta|e_p)$ 是一个 encoder-decoder Transformer 输出 $\mu_j$，然后投到 $\ell_2$ ball $\delta_j = \epsilon_p \cdot \mu_j / \max(\|\mu_j\|_2, \epsilon_p)$（$\epsilon_p = 0.8$）。Low-level MDP 是迭代去噪——state $s_t = (x_t, t, c)$，action $a_t = x_{t - \Delta t}$，policy $\pi_\theta(a_t|s_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$，其中 $\mu_\theta = x_t - v_\theta(x_t, t, c) \Delta t$；用 MPS 的 SDE 离散化 $x_{t - \Delta t} = x_t - v_\theta \Delta t + \sigma_t \varepsilon$ 加噪保探索。
    - 设计动机：T2I 的早期主要决定语义/布局，晚期主要决定细节，这与"prompt 改语义"和"velocity 改图像统计"天然对应。分到两个 policy 后，每个 policy 只对自己擅长的时间段施压，比 DDPO 那种平铺式 RL 更精准（实验 +ASR 21% over DDPO）。

2. **GRPO 双层优化 + 边缘奖励组合**：

    - 功能：用 group-normalized advantage 替代 absolute reward，降低 sparse reward 下的方差；高层多一个"edit semantic 保持"的辅助奖励。
    - 核心思路：GRPO 损失 $\mathcal{L}_{grp}(r_t, \hat A, \varepsilon) = \min(r_t \hat A, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat A)$，其中 $r_t = \pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$，group-normalized advantage $\hat A_i = (X_i - \mu_{grp})/(\sigma_{grp} + \epsilon)$。High-level group 取 $K$ 个候选的均 reward + edit reward $\mathcal{R}_{high}^{(j)} = \bar R_j + \mathcal{R}_{edit}^{(j)}$，其中 $\mathcal{R}_{edit}^{(j)} = \lambda_{sem}[s_{SBERT}(e_p, e_p + \delta_j) - \tau_{sem}]_+ + \lambda_{recon}/(1 + \|e_p + \delta_j - \mathrm{emb}(p'^{(j)})\|^2)$ 同时鼓励"与原 prompt 语义相近"和"embedding edit 与 vec2text 解码出的文本一致"。Low-level group 是全部 $K \times M$ 个 rollout reward $R^{(j,m)} = R_{tox}^{(j,m)} + w_{align} R_{align}^{(j,m)}$，加 per-step KL $D_{KL}(\pi_\theta^{(t)} \| \pi_{ref}^{(t)}) = \tfrac{1}{2\sigma_t^2}\|\mu_\theta - \mu_{ref}\|^2$ 稳定 mean drift。
    - 设计动机：toxicity reward 是 sparse 且 noisy 的 terminal reward，group 归一化比 absolute reward 方差小很多；Flow-DPO 那种需要 preference dataset 的方法在双层结构下成本太高，GRPO 是当前最 lightweight 的选择。Edit reward 防止 high-level 把 prompt 编辑到完全无关的方向。

3. **Temporal Alignment Analysis (MLMC 时间归因)**：

    - 功能：把"哪一去噪步对哪一类毒性贡献最大"量化成 $T \times D$ heatmap，用来验证分层结构真的对应到不同时间窗。
    - 核心思路：先定义 net toxicity score $\mathcal{R}_d(I, p) = R_d(\mathrm{VLM}(I, p)) - R_d(\mathrm{VLM}(\mathrm{null}, p))$ 隔离图像的边际贡献；再定义对时间块 $B$ 的敏感度 $\Delta_B^{(d)} = \mathbb{E}_{\mathbf{z}}[(\mathcal{R}_d(G^{(B, +\eta\mathbf{z})}) - \mathcal{R}_d(G^{(B, -\eta\mathbf{z})}))/(2\eta)]$（块内做对称扰动的有限差分）。用 coarse-to-fine search + Multi-Level Monte Carlo $\hat\Delta_B^{MLMC} = \tfrac{1}{M_0}\sum \hat\Delta_B^{(0)} + \sum_\ell \tfrac{1}{M_\ell}\sum(\hat\Delta_B^{(\ell)} - \hat\Delta_B^{(\ell-1)})$ 高效估计；最后细到 singleton $B = \{t\}$ 得到 TemporalScore$(t, d) = \hat\Delta_{\{t\}}^{(d), MLMC}$，rescale 到 $[-1, 1]$ 画 heatmap。
    - 设计动机：把"对抗优化做了什么"从黑箱 reward 数字变成时间-维度二维可视，这是论文最大的方法学贡献。MLMC 是必要的——因为 6 维毒性 × $T$ 步直接采样太贵，MLMC 用层级低保真度估计 + 少量高保真度修正显著降方差。

### 损失函数 / 训练策略

总损失 = High-level GRPO loss + Low-level $\mathcal{J}_{low} = \mathbb{E}_\tau[\tfrac{1}{T}\sum_t(\mathcal{L}_{grp}^{low}(t) - \beta_t D_{KL}(\pi_\theta^{(t)}\|\pi_{ref}^{(t)}))]$。关键超参：$K = 4$ 候选、$M = 8$ rollout、$\epsilon_p = 0.8$、$\tau_{sem} = 0.7$、$\lambda_{sem} = 1.0, \lambda_{recon} = 0.1$、$\beta_{high} = 0.02, \beta_t = 0.04$，PPO clip $\varepsilon_{low} = \varepsilon_{high} = 0.001$。训练用 20 denoising steps，inference 40 steps。

## 实验关键数据

### 主实验

在 LLaVA + RTP dataset 上 ASR (%) ↑：

| 方法 | Any ↑ | Toxic ↑ | Obscene ↑ | Identity ↑ | Insult ↑ | CLIP ↑ |
|------|-------|---------|-----------|------------|----------|--------|
| Text-Only | 5.20 | 3.10 | 5.10 | 0.60 | 2.80 | – |
| Text + SD | 11.15 | 5.71 | 10.63 | 3.97 | 6.11 | 0.72 |
| PGJ | 14.86 | 7.85 | 13.98 | 3.43 | 8.09 | 0.71 |
| DiffZOO | 17.20 | 9.01 | 16.42 | 4.14 | 7.88 | 0.73 |
| ART | 18.62 | 9.22 | 17.54 | 6.45 | 8.94 | 0.75 |
| STARE w/ DDPO（同白盒预算） | 27.84 | 15.62 | 26.12 | 5.80 | 15.11 | 0.75 |
| **STARE (Ours, $w_{align}=0.2$)** | **31.36** | **17.10** | **29.73** | 6.14 | 15.95 | 0.78 |

在 OOD PolygloToxicityPrompts 测试上 STARE Any 30.83 vs ART 22.01，证明泛化性。Transfer 到 Qwen2.5-VL、Gemini-2.5-Pro 仍保持显著领先。

### 消融实验

| 配置 | Any ASR | 说明 |
|------|---------|------|
| Full STARE ($w_{align}=0.2$) | **31.36** | 完整方法 |
| STARE w/o LoRA（去掉 low-level） | 22.04 | -9.32，证明 velocity 微调贡献最大 |
| STARE w/o Edit（去掉 high-level） | 25.56 | -5.80，prompt edit 贡献次之 |
| STARE w/o Align（去掉对齐奖励） | 26.43 | -4.93，CLIP 反而掉到 0.68 |
| STARE w/ DDPO（替换为平铺 RL） | 27.84 | -3.52，证明分层 > 平铺 |

### 关键发现

- **Optimization-Induced Phase Alignment**：时间归因热图显示，vanilla SD 的毒性贡献在时间维度上是 diffuse 的，对抗优化后 identity/threat（concept-level）毒性集中在早期 timesteps、obscene/insult（detail-level）毒性集中在后期 timesteps，几乎不重叠。这不是分层结构的设计副作用，而是被 RL 优化"诱导"出来的真实时间规律——靶向扰动早期窗口只能压住概念类毒性、扰动晚期只能压住细节类毒性，验证了因果关系。
- **分层 > 平铺 RL**：STARE 比 STARE w/ DDPO（同样白盒预算但平铺式）高 3.5% ASR，且时间窗结构更清晰；DDPO 的优化压力在整个轨迹上 smear，导致不能利用 diffusion 的内生时间结构。
- **Transfer 强**：对不同 VLM（Qwen2.5-VL, Gemini-2.5-Pro, GPT-5.4）和不同 T2I 生成器（FLUX.1-dev）都保持 ASR 领先，证明攻击不是过拟合到特定 victim 的"trick prompt"。
- **CLIP align 反而提升 ASR**：直觉上"保 CLIP 对齐"会限制对抗自由度，但 $w_{align}=0.2$ 的 STARE 反而最高，作者解释为 align 约束防止图像 collapse 到无意义噪声（这种图反而难触发毒性 continuation），保持图像-prompt 一致性才让 VLM 真把图当 context 用。

## 亮点与洞察

- "把去噪轨迹本身当成攻击面"这个 reframing 极有创新性——把 T2I 从 "black-box image generator" 变成"白盒时间-语义结构 exploit target"，开辟了基于 diffusion 时间结构做攻击与防御的全新方向。
- Optimization-Induced Phase Alignment 这个现象本身比攻击数字更有价值——它说明 diffusion 模型的早晚语义涌现机制不仅是经验观察，更是可以被对抗优化"放大"和"定向利用"的真实因果结构。对应到 defense 侧，意味着可以做 phase-specific monitoring（只在早期 timestep 跑 concept-level filter、只在晚期跑 detail-level filter），大幅降低 defense 成本。
- MLMC 用层级估计降方差，把原本需要 $O(T \cdot D \cdot M)$ 次 forward 的归因分析降到可承受成本——这是把扰动分析做大的工程关键。
- 用 vec2text 把 embedding edit 反解为文本 prompt 是巧妙工程：让 high-level 在连续 embedding 空间优化的同时，最终输入到 T2I 的还是离散文本，避开了 prompt embedding direct injection 可能与预训练分布不一致的问题。

## 局限与展望

- 白盒 T2I 假设较强（要拿到 SD 3.5 的全部参数微调 LoRA），对完全黑盒 T2I (DALL-E 3, Midjourney) 不适用；transfer 实验做了到 FLUX.1-dev 但没到真黑盒商业 API。
- VLM 假设是 query-only black-box，但 reward 信号需要每次 query 拿 6 维 toxicity，对商业 API 调用成本高（论文未给 query budget 分析）。
- Optimization-Induced Phase Alignment 的因果证据来自扰动实验，但作者没给出"alignment 强度的解析理论"——为什么是 6 维毒性、为什么早晚两段分别对应概念/细节，仍是经验观察。
- Rectified flow 假设是必须的（velocity 显式 + 轨迹近直线），对 DDIM/DDPM 这种轨迹弯曲度大的模型时间归因可能失真。
- 红队伦理：成功率 31% 对 LLaVA 是显著威胁；论文有 content warning 但没有 disclosure timeline。
- $K = 4, M = 8$ 的 group 大小受 GPU 限制；更大 group 可能进一步降 GRPO 方差但训练成本陡增。

## 相关工作与启发

- **vs PGJ / DiffZOO / ART**：都是 prompt-side 黑盒搜索 + frozen T2I，无法利用 generation 时间结构；STARE 同时操纵 prompt edit 与 velocity field，ASR 翻倍。
- **vs RedDiffuser (Wang et al. 2025a)**：也 steer diffusion 但没分层、没 phase-level 分析；STARE 的分层结构与时间归因是 differentiator。
- **vs DDPO (Black et al. 2024)**：DDPO 是平铺式 diffusion RL；STARE-w/-DDPO 消融证明分层结构比平铺多 3.5% ASR 且时间结构更清晰。
- **vs Flow-GRPO (Liu et al. 2025)**：本文低层 GRPO 思路类似，但 Flow-GRPO 是单层，STARE 把它嵌进 high-level prompt editor 形成双层。
- **vs 文本 jailbreak（GCG 等）**：纯文本 jailbreak 无图像通道，无法触发多模态特有的 toxic-continuation 攻击；STARE 揭示了"图像通道是被低估的攻击面"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 diffusion 轨迹当攻击面 + Phase Alignment 现象都是范式级新意。
- 实验充分度: ⭐⭐⭐⭐ 双 dataset + 三个 VLM transfer + DDPO 同算力对照 + 完整消融 + MLMC 时间归因；但缺 query budget 和 defense 对照。
- 写作质量: ⭐⭐⭐⭐ 分层 MDP / GRPO / MLMC 公式严谨，threat model 清晰；时间归因部分稍重数学但插图 1/3 弥补可读性。
- 价值: ⭐⭐⭐⭐ 对多模态 safety 社区既是攻击工具又是 defense 设计基础（phase-aware monitoring），但责任披露需要谨慎。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[ICML 2026\] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](../../ICLR2026/llm_safety/supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)

</div>

<!-- RELATED:END -->
