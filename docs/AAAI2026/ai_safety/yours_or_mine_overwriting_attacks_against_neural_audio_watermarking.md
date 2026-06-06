---
title: >-
  [论文解读] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking
description: >-
  [AAAI 2026][AI安全][音频水印] 首次系统研究神经音频水印的覆写攻击（overwriting attack），提出白盒、灰盒、黑盒三级攻击方案，在 AudioSeal、Timbre、WavMark 三种 SOTA 方法上均实现接近 100% 的攻击成功率，暴露了现有音频水印系统严重的安全缺陷。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "音频水印"
  - "覆写攻击"
  - "版权保护"
  - "对抗安全"
  - "深度水印"
---

# Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking

**会议**: AAAI 2026  
**arXiv**: [2509.05835](https://arxiv.org/abs/2509.05835)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 音频水印, 覆写攻击, 版权保护, 对抗安全, 深度水印

## 一句话总结

首次系统研究神经音频水印的覆写攻击（overwriting attack），提出白盒、灰盒、黑盒三级攻击方案，在 AudioSeal、Timbre、WavMark 三种 SOTA 方法上均实现接近 100% 的攻击成功率，暴露了现有音频水印系统严重的安全缺陷。

## 研究背景与动机

随着生成式音频模型的快速发展，AI 可以生成高度逼真的语音，带来了声音克隆欺诈和版权侵犯等社会风险。音频水印作为一种主动防御机制，通过在音频信号中嵌入不可感知的数字签名来实现版权保护和来源验证。

现有神经音频水印方法主要关注两个属性：
- **鲁棒性（Robustness）**：水印在压缩、重采样等常规处理后仍可检测
- **不可感知性（Imperceptibility）**：嵌入过程不降低音频感知质量

**被忽视的第三个属性——安全性（Security）**：

鲁棒性关注的是**无意扰动**的容忍，安全性关注的是**有意攻击**的抵抗。现有研究主要探索了两种安全攻击：
- **移除攻击（Removal）**：使水印不可检测
- **伪造攻击（Forgery）**：在干净音频中虚假嵌入水印

然而，一种更实际且更危险的攻击——**覆写攻击（Overwriting）**——几乎未被研究：攻击者将已水印音频中的合法水印替换为伪造水印，从而**劫持音频版权**。不同于移除攻击（只是消除标记），覆写攻击直接**窃取所有权**。

**为什么现有系统脆弱？** 根据 Kerckhoffs 原则，安全系统即使算法公开也应安全，安全性应依赖密钥而非保密。但神经水印系统通常缺乏显式的密钥安全机制，依赖"模型权重保密"——这在开源和逆向工程盛行的时代是脆弱的假设。

## 方法详解

### 整体框架

攻击目标：给定公开分发的已水印音频 $x_w = \mathcal{E}(x, m_{owner})$，攻击者使用嵌入器 $\mathcal{E}'$ 嵌入伪造消息 $m'_{adv}$，生成 $x'_w = \mathcal{E}'(x_w, m'_{adv})$。

成功条件：
1. 原始消息不再可恢复：$\mathcal{D}(x'_w) \neq m_{owner}$
2. 攻击者的检测器能恢复伪造消息：$\mathcal{D}'(x'_w) = m'_{adv}$
3. 感知不可区分：$d(x'_w, x_w) \leq \epsilon$

### 关键设计

#### 1. **白盒攻击（White-box Attack）**

假设：攻击者完全访问原始水印嵌入器。代表内部威胁或完全开源场景。

攻击极其简单——直接用同一嵌入器重新嵌入新消息：

$$x'_w = \mathcal{E}(x_w, m'_{adv})$$

**核心发现**：当使用同一嵌入器覆写时，原始水印的 BER 达到 ~0.5（等同随机猜测），说明原始水印被彻底摧毁。这是因为嵌入器在同一嵌入域中操作，新水印自然覆盖旧水印。

但当使用不同方法覆写时（如用 AudioSeal 覆写 Timbre），BER 极低——不同方法操作在不同嵌入域，解码机制也不同，无法交叉破坏。这一发现是灰盒和黑盒攻击的基础。

#### 2. **灰盒攻击（Gray-box Attack）**

假设：攻击者知道水印系统的架构，但不知道模型权重和训练细节。需要训练代理模型 $(\mathcal{E}', \mathcal{D}')$。

提出一个**通用水印训练框架**：

$$\mathcal{L}_{total} = \lambda_w \mathcal{L}_w + \lambda_t \mathcal{L}_{recon_t} + \lambda_f \mathcal{L}_{recon_f} + \lambda_{adv} \mathcal{L}_{adv}$$

四个损失分量：

- **水印恢复损失**：$\mathcal{L}_w = \text{BCE}(m, \mathcal{D}'(\mathcal{E}'(x, m)))$ — 确保嵌入和检测的准确性
- **时域重建损失**：$\mathcal{L}_{recon_t} = \text{MSE}(x, \mathcal{E}'(x,m))$ — 最小化可听失真
- **频域重建损失**：多分辨率 STFT 损失，包含谱收敛项和对数幅度项：

$$\mathcal{L}_{recon_f} = \frac{1}{M} \sum_{m=1}^{M} (\mathcal{L}_{sc}^{(m)} + \mathcal{L}_{mag}^{(m)})$$

- **对抗损失**：训练判别器区分原始和水印音频，嵌入器目标使水印音频不可区分：

$$\mathcal{L}_{adv} = -\log(\sigma(D(\mathcal{E}'(x,m))))$$

**设计动机**：即使不知道原始模型的训练细节，用相同架构训练的代理模型会收敛到相似的嵌入策略——它们在相似的频谱区域嵌入水印。这种**架构收敛性**使灰盒攻击高度有效。

两种灰盒设置：
- **Cross-training**：同一数据集（VoxCeleb1），不同训练流程和随机种子
- **Cross-data**：完全不同的数据集（LibriSpeech 训练代理 → 攻击 VoxCeleb1 模型）

#### 3. **黑盒攻击（Black-box Attack）**

假设：不知道架构、权重或训练数据。分两种策略：

**零查询攻击（Zero-query）**：
- 收集或复现一组公开水印模型 $\mathcal{E}'_i$
- 暴力堆叠，依次应用所有模型：

$$x_w^{(N)} = (\mathcal{E}_N \circ \mathcal{E}_{N-1} \circ \cdots \circ \mathcal{E}_1)(x_w, m'_{adv})$$

随着堆叠模型数增加，ASR 从 ~30%（1个模型）增至 ~100%（3个模型），但 SNR 从 ~24dB 降至 ~20dB。

**查询引导攻击（Query-based）**：
1. 部分训练候选代理模型（少量 epochs）
2. 用欠训练模型嵌入新消息
3. 有限次查询原始检测器 $\mathcal{D}$，评估原始水印是否被破坏
4. 找到最有效候选后继续训练至可靠覆写

**设计动机**：查询引导策略以 <10 次查询实现 50%+ 的训练迭代节省，且只需应用一个有效模型（而非堆叠多个），保持音频质量（SNR 24.19dB vs 20.63dB）。

### 损失函数 / 训练策略

- 训练数据集：LibriSpeech (~1000小时) 和 VoxCeleb1 (150k+ 条)
- 音频格式：16kHz WAV
- 目标水印方法：AudioSeal（编码器-解码器型）、Timbre（频域型）、WavMark（可逆神经网络型）
- 三种随机种子初始化（Init-1/2/3）验证可重复性
- 硬件：64 CPU + 2×A100 GPU

## 实验关键数据

### 主实验

**白盒覆写结果**

| 目标方法 | ASR (原始水印)↑ | ACC (覆写水印)↑ |
|---------|----------------|----------------|
| Timbre | 99.80% | 100.00% |
| AudioSeal | 100.00% | 100.00% |
| WavMark | 100.00% | 100.00% |

原始水印几乎被完全摧毁，覆写水印以完美精度恢复。

**灰盒 Cross-training 结果（ASR %）**

| 目标方法 | Init-1 | Init-2 | Init-3 |
|---------|--------|--------|--------|
| Timbre | 99.60 | 98.80 | 98.40 |
| AudioSeal | 100.00 | 100.00 | 100.00 |
| WavMark | 100.00 | 100.00 | 99.50 |

**灰盒 Cross-data 结果（LibriSpeech → VoxCeleb1, ASR %）**

| 目标方法 | Init-1 | Init-2 | Init-3 |
|---------|--------|--------|--------|
| Timbre | 99.80 | 99.90 | 98.80 |
| AudioSeal | 100.00 | 100.00 | 100.00 |
| WavMark | 100.00 | 100.00 | 100.00 |

即使代理模型使用完全不同的数据集训练，攻击成功率仍接近 100%。

### 消融实验

**黑盒攻击：零查询 vs 查询引导**

| 攻击类型 | 查询次数 | 训练成本 | SNR (dB) | ASR (%) |
|---------|---------|---------|----------|---------|
| 零查询 | 0 | 36,000 iters | 20.63 | 100 |
| 查询引导 | <10 | 14,000 iters | **24.19** | 100 |

查询引导攻击以 <10 次查询实现了：
- 训练成本减少 61%
- SNR 提升 3.56 dB（音频质量更好）
- 攻击成功率相同

**白盒 BER 矩阵分析**：对角线（同方法覆写）BER ≈ 0.5（随机猜测级），非对角线（跨方法覆写）BER 极低——证明覆写能力来源于嵌入域的重叠，不同方法间不共享嵌入域。

### 关键发现

1. **覆写攻击是神经音频水印的系统性安全缺陷**：三种代表性方法（三种不同嵌入范式）在所有威胁等级下均被攻破
2. **架构收敛现象**：不同数据、不同训练细节、不同随机种子训练的代理模型，都收敛到相似的嵌入策略（频谱可视化证实：所有模型在相似频谱区域嵌入水印）
3. **"模型保密 = 安全"的假设不成立**：灰盒/黑盒攻击证明，攻击者不需要知道确切权重就能进行有效覆写
4. **覆写比移除更危险**：移除攻击只是让水印不可检测，覆写攻击直接窃取版权——攻击者可以声称音频是自己创作的
5. **查询引导策略效率极高**：少于 10 次查询即可定位有效攻击模型

## 亮点与洞察

- **攻击主题新颖且实际威胁巨大**：覆写攻击比移除/伪造更具破坏性——它不仅摧毁了合法水印，还植入了虚假的所有权证明
- **三级威胁模型设计完善**：白盒→灰盒→黑盒逐级递减假设，覆盖了从内部威胁到完全外部攻击的完整光谱
- **"架构收敛性"的发现有深远含义**：揭示了当前水印方法的一个根本弱点——安全性不应依赖模型保密，而应依赖密码学密钥
- **简洁有效**：白盒攻击只需一行代码 $x'_w = \mathcal{E}(x_w, m'_{adv})$，灰盒攻击用通用训练框架即可构建代理
- **频谱可视化提供了直觉**：不同训练的模型在频谱上展现出相似的嵌入区域，直观解释了灰盒攻击有效性的原因

## 局限与展望

1. **仅评估了三种水印方法**：虽然覆盖了三种不同嵌入范式，但新兴方法（如 XattnMark、SilentCipher）未被评估
2. **未提出防御方案**：论文定位为"攻击论文"，但缺少对可能防御方向的深入讨论（如非对称水印、密钥绑定嵌入器等）
3. **音频质量评估指标有限**：仅使用 SNR，未使用更全面的感知指标如 PESQ、ViSQOL
4. **水印消息长度固定**：未分析不同消息长度对覆写攻击效果的影响
5. **黑盒攻击中假设可获得部分候选模型集**：在实际中攻击者可能无法获得足够多的公开水印模型
6. **未考虑多层水印或嵌套水印的防御策略**：一些实际系统可能嵌入多个互补水印

## 相关工作与启发

- 该工作对音频水印领域发出了重要安全警告：当前的"鲁棒性 + 不可感知性"双目标优化框架需要升级为"鲁棒性 + 不可感知性 + 安全性"三目标
- 与图像水印领域的覆写攻击研究相呼应，跨模态共享的安全问题
- 启发防御方向：
    - 非对称水印（嵌入和检测使用不同密钥，类似公钥加密）
    - 水印指纹绑定（将水印与音频内容绑定，覆写会导致校验失败）
    - 多层冗余水印（在不同嵌入域同时嵌入互补水印）
    - 基于零知识证明的所有权验证
- Kerckhoffs 原则在 AI 安全中的应用值得更多关注

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究覆写攻击，三级威胁模型设计完善，但攻击方法本身较直接
- 实验充分度: ⭐⭐⭐⭐ — 3种方法 × 3种威胁模型 × 多种设置，BER分布分析细致
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，threat model 定义严谨，motivation 有说服力
- 价值: ⭐⭐⭐⭐⭐ — 揭示了神经音频水印的系统性安全缺陷，对领域有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[AAAI 2026\] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service](regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)

</div>

<!-- RELATED:END -->
