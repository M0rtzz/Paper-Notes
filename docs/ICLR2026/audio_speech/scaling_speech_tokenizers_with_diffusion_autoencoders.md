---
title: >-
  [论文解读] Scaling Speech Tokenizers with Diffusion Autoencoders
description: >-
  [ICLR 2026][音频/语音][Speech Tokenizer] 提出 SiTok（Speech Diffusion Tokenizer），采用扩散自编码器联合训练编码器-量化器-解码器（非两阶段），加入 CTC 语义正则化确保离散 token 保留语言信息…
tags:
  - "ICLR 2026"
  - "音频/语音"
  - "Speech Tokenizer"
  - "扩散模型"
  - "Semantic Regularization"
  - "Low Bitrate"
  - "CTC Loss"
---

# Scaling Speech Tokenizers with Diffusion Autoencoders

**会议**: ICLR 2026  
**arXiv**: [2602.06602](https://arxiv.org/abs/2602.06602)  
**代码**: 无（Demo: [https://sitok-demo.github.io/](https://sitok-demo.github.io/)）  
**领域**: 语音 / Token化  
**关键词**: Speech Tokenizer, Diffusion Autoencoder, Semantic Regularization, Low Bitrate, CTC Loss

## 一句话总结

提出 SiTok（Speech Diffusion Tokenizer），采用扩散自编码器联合训练编码器-量化器-解码器（非两阶段），加入 CTC 语义正则化确保离散 token 保留语言信息，规模化到 1.6B 参数和 2200 万小时语音数据，在极端低 token 率（12.5Hz / 200bps）下同时实现 3.34% WER（重建）和 4.95 WER（LLM ASR）的强性能。

## 研究背景与动机

**领域现状**：语音 tokenizer 是语音语言模型的基础接口，决定了语音如何被离散化表示。一个理想的语音 tokenizer 需要同时满足三个目标：（1）极端压缩以支持高效语言建模；（2）高保真重建以生成自然语音；（3）语义丰富表示以支持下游理解任务。

**现有痛点**：现有方法通过启发式妥协而非原则性方案来处理上述三目标的张力：（1）低比特率下重建质量差——很多方法用 RVQ（残差向量量化）增加码本层数或提高帧率来维持质量，但这直接膨胀了 token 数量（如 Mimi 75 TPS, DualCodec 75 TPS），违背压缩目标；（2）仅优化声学保真度忽略语义——导致 token 不适合理解任务（如 ASR WER 很高）；（3）两阶段训练方案——先用 SSL 模型量化语音表征，再独立训练扩散/声码器解码，量化器无法为重建优化，解码器被迫适配次优离散码。

**核心矛盾**：在传统声学重建目标下，简单增大模型或数据在低 token 率时收益递减——这是向量量化的结构性瓶颈。确定性重建损失迫使离散潜空间"坍缩不确定性"，优先保留低级信号细节而非语义结构，导致压缩越激进语义损失越大。

**切入角度**：低 token 率量化引入的不确定性需要**生成式框架**来建模——扩散模型恰好学习逆转随机退化过程，天然适合处理量化引起的信息损失。同时，直接用 CTC 损失监督量化后的潜空间，比 SSL 蒸馏更直接地注入语义信息。

**核心 idea**：用扩散自编码器（而非对抗式训练）联合优化量化和重建，加上 CTC 语义正则化，实现极低 token 率下语义和声学的双重保留。

## 方法详解

### 整体框架

SiTok 以 mel 频谱图为输入和重建目标（非原始波形），避免直接处理超长波形序列和不稳定的对抗训练。Pipeline 为：（1）下采样到 12.5Hz；（2）Llama-style 因果 Transformer 编码器（16 层）提取潜在特征 $\mathbf{z}$；（3）向量量化（65,536 entries，32 维，EMA 更新）得到离散 token $\mathbf{q}$；（4）非因果 Llama Transformer 扩散解码器（16 层）以量化嵌入 $\mathbf{z}_q$ 为条件，用 flow-matching 目标重建 mel 谱图；（5）外部 Vocos 声码器将 mel 谱图转为 24kHz 波形。同时有辅助 CTC 解码器（4 层）在量化后潜空间上预测文本转录。

### 关键设计

1. **扩散自编码器替代对抗式训练**

    - 功能：在量化后的离散 token 条件下高保真重建 mel 谱图
    - 核心思路：解码器使用 flow-matching 目标，将噪声样本 $\mathbf{x}_t = t\mathbf{x} + (1-t)\epsilon$ 的速度场 $v_\phi(\mathbf{x}_t, t, \mathbf{z}_q)$ 训练为逼近真实速度 $(\mathbf{x} - \epsilon)$。相比对抗训练的优势：（a）不需要判别器和复杂损失设计，训练更稳定；（b）扩散模型学习数据分布，能从量化表征中"脑补"丢失的细节；（c）可扩展性更好——波形级模型需要大量上下采样，mel 谱图更紧凑
    - 设计动机：确定性重建在激进压缩下会坍缩——把所有信息硬塞进 200bps 是不可能的。扩散模型承认"不是所有细节都能从 token 恢复"，转而学习条件分布 $p(\mathbf{x}|\mathbf{z}_q)$，这才是低 token 率下的正确建模方式

2. **CTC 语义正则化**

    - 功能：确保离散 token 保留语义/语言信息
    - 核心思路：在量化后嵌入 $\mathbf{z}_q$ 上接入轻量 CTC 解码器 $\mathcal{D}_{\phi_{\text{ctc}}}$（4 层 Transformer），直接预测文本转录 $\mathbf{y}$。总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{ctc}} \cdot \text{CTC}(\mathcal{D}_{\phi_{\text{ctc}}}(\mathbf{z}_q), \mathbf{y}) + \mathcal{L}_{\text{vq}}$，其中 $\lambda_{\text{ctc}}$ 是关键超参。实验显示 $\lambda_{\text{ctc}} = 0.1$ 最优，过大（1.0）反而损害重建（WER 从 4.06 升至 10.1）
    - 设计动机：区别于之前用 MSE/cosine 做 SSL 特征蒸馏的间接对齐方式，CTC 直接强制 token 能解码出文本——这是语义保留的最直接监督信号。不依赖任何外部 SSL 模型（如 HuBERT/WavLM），完全端到端

3. **高效扩散解码（Shortcut Fine-tuning）**

    - 功能：将扩散推理步数从标准的多步压缩到 2-4 步
    - 核心思路：冻结编码器和 VQ 模块，对解码器用 shortcut model 目标微调——训练网络额外接收步长 $d$ 作为条件，联合优化 flow-matching 损失（$d=0$ 对应真实速度）和自一致性损失（一大步 $2d$ 的结果 ≈ 两小步 $d$ 的连续结果），使模型学会"跳过中间步"。实际 RTF：16 步 0.041 → 4 步 0.013，加速 3.2 倍
    - 设计动机：扩散解码的多步采样是部署瓶颈，shortcut 让模型自学加速策略，比传统蒸馏更灵活

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + 0.1 \cdot \mathcal{L}_{\text{ctc}} + \mathcal{L}_{\text{vq}}$。训练用 AdamW，lr=8e-5，warmup 32K 步，单 epoch（~450K 步），2200 万小时内部语音数据。可选精炼：（1）Decoder finetuning（冻结编码器+VQ）；（2）Token CFG（10% 概率 drop token 训练无条件路径，推理时条件/无条件预测组合）。

## 实验关键数据

### 主实验（重建质量对比）

| 模型 | FPS/TPS | 码本数 | 比特率 | WER↓ | SIM↑ | UTMOS↑ |
|------|---------|-------|-------|------|------|--------|
| Ground Truth | - | - | - | 2.14 | 0.730 | 3.53 |
| DualCodec | 12.5/75 | 6 | 0.925 | 2.63 | 0.624 | 3.78 |
| X-codec 2 | 50/50 | 1 | 0.80 | 2.63 | 0.620 | 3.68 |
| Mimi | 12.5/75 | 6 | 0.825 | 4.51 | 0.527 | 3.09 |
| FireRedTTS | 25/25 | 1 | 0.35 | 3.35 | 0.597 | 3.40 |
| CosyVoice | 25/25 | 1 | 0.30 | 5.63 | 0.465 | 3.65 |
| **SiTok (CN=1)** | **12.5/12.5** | **1** | **0.20** | **4.06** | **0.641** | **3.44** |
| + Decoder FT | 12.5/12.5 | 1 | 0.20 | 3.79 | **0.682** | 3.48 |
| + Token CFG | 12.5/12.5 | 1 | 0.20 | **3.34** | 0.635 | **3.60** |

SiTok 在仅 200bps（所有基线最低比特率）下，WER 3.34%、SIM 0.682 均达到强竞争力。

### 消融实验（语义正则化效果）

| CTC 正则化 | TPS | 重建 WER↓ | SIM↑ | UTMOS↑ | LLM ASR↓ | ER↑ | SV↓ | KS↑ |
|-----------|-----|----------|------|--------|----------|-----|-----|-----|
| ✓ (λ=0.1) | 12.5 | 4.06 | 0.641 | 3.44 | 4.95 | 63.5 | 13.8 | 96.9 |
| ✗ | 12.5 | **33.0** | 0.495 | 2.68 | 29.4 | 57.9 | 18.9 | 86.1 |
| ✓ (λ=0.1) | 50 | 2.80 | 0.660 | 3.46 | 4.49 | 64.4 | 8.59 | 97.7 |
| ✗ | 50 | 5.17 | 0.611 | 2.84 | 7.27 | 60.4 | 13.5 | 92.8 |

没有 CTC 正则化的 12.5 TPS 模型 WER 飙升到 33.0%，证明语义正则化不是"锦上添花"而是"不可或缺"。

### 关键发现

- **模型缩放的非单调效应**：从 0.63B (S) 到 1.61B (XL)，重建质量持续改善（WER 4.18→3.84），但理解任务在 1.12B (L) 达峰，更大模型在 SV 上反而退化（13.8→14.7），暗示过大容量可能过度编码声学细节而非抽象语义
- **Token CFG 和 Decoder FT 互补**：CFG 主要降低 WER（3.34），FT 主要提升说话人相似度（0.682），可按需组合
- **CTC 权重 $\lambda_{\text{ctc}}$ 是敏感超参**：0.1 最优，0.02 重建好但理解差，0.5-1.0 重建也恶化（过度约束潜空间）
- **仅用回归损失（R）训练的 tokenizer 表现差**：WER 4.66 且所有理解指标下降，扩散损失（D）是核心

## 亮点与洞察

- **"不确定性需要生成式建模"的洞察深刻**：低 token 率量化不可避免丢失信息，用确定性重建试图"完美恢复"注定失败，扩散模型承认不确定性并学习条件分布，这是正确的建模哲学。这一洞察可迁移到任何高压缩比离散化场景
- **CTC 监督的极简有效性**：不需要外部 SSL 模型、不需要特征对齐的复杂设计，一个 4 层 CTC 头直接预测文本就够了。关键是监督信号放在量化后（而非量化前），直接塑造离散 token 的语义性质
- **Mel 谱图作为中间表示的务实选择**：避免了波形级建模的长序列和不稳定训练，虽然需要外部 vocoder，但解耦设计使 tokenizer 和 vocoder 可独立优化升级

## 局限与展望

- **依赖外部 Vocoder**：mel 到波形的转换依赖 Vocos，整体质量受 vocoder 瓶颈限制
- **训练数据为内部数据**：2200 万小时语音数据不公开，可复现性受限
- **以英语为主**：虽声称覆盖多语言，但英语占绝大多数，多语言泛化性未充分验证
- **扩散解码延迟**：即使 shortcut 后仍需 2-4 步迭代，实时交互场景下延迟可能不够低
- **L 和 XL 模型的理解性能倒退**：更大模型在理解任务上并非更好，提示需要更好的训练策略或结构设计来平衡声学和语义

## 相关工作与启发

- **vs Mimi (Défossez et al., 2024)**：Mimi 用 8 层 RVQ 达到 1.1kbps 和 23.1 LLM ASR WER，SiTok 用单码本 200bps 达到 4.95 LLM ASR WER，压缩比高 5.5 倍且理解性能大幅领先
- **vs CosyVoice / FireRedTTS**：两阶段方法先量化 SSL 特征再训练 diffusion decoder，SiTok 的端到端联合优化避免了量化器与解码器的目标不一致
- **vs StableCodec / GLM4-Voice**：同为低 token 率设计（0.2-0.4 kbps），SiTok 的理解性能（特别是 LLM ASR）显著优于这些基线

## 评分

- 新颖性: ⭐⭐⭐⭐ 扩散自编码器 + CTC 的组合有创新性，但各组件并非全新，核心贡献在于规模化验证和系统性设计
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖重建/理解/生成三大场景，丰富的消融（损失、码本、模型规模、解码步数），对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation 论证充分，数学描述准确
- 价值: ⭐⭐⭐⭐⭐ 在极低比特率下统一理解和生成的语音 tokenizer 对语音语言模型发展有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [\[ACL 2026\] AgentV-RL: Scaling Reward Modeling with Agentic Verifier](../../ACL2026/audio_speech/agentv-rl_scaling_reward_modeling_with_agentic_verifier.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](../../NeurIPS2025/audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)
- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](../../NeurIPS2025/audio_speech/the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)

</div>

<!-- RELATED:END -->
