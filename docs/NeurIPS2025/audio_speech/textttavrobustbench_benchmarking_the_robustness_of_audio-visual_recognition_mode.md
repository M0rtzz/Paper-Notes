---
title: >-
  [论文解读] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time
description: >-
  [NeurIPS 2025][音频/语音][音视频鲁棒性] 提出 AVRobustBench，首个系统评估音视频模型在 **双模态共现关联腐蚀** 下测试时鲁棒性的基准，包含 4 个数据集 × 75 种腐蚀，并提出基于低熵样本筛选的 TTA 方法 AV2C。
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "音视频鲁棒性"
  - "分布偏移"
  - "测试时适应"
  - "多模态基准"
  - "共现腐蚀"
---

# AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time

**会议**: NeurIPS 2025  
**arXiv**: [2506.00358](https://arxiv.org/abs/2506.00358)  
**代码**: 有（论文中提及）  
**领域**: 音频/语音 (多模态鲁棒性)  
**关键词**: 音视频鲁棒性, 分布偏移, 测试时适应, 多模态基准, 共现腐蚀

## 一句话总结

提出 AVRobustBench，首个系统评估音视频模型在 **双模态共现关联腐蚀** 下测试时鲁棒性的基准，包含 4 个数据集 × 75 种腐蚀，并提出基于低熵样本筛选的 TTA 方法 AV2C。

## 研究背景与动机

**领域现状**: UAVM、CAV-MAE、ImageBind 等音视频模型取得了显著进展，但其对测试时分布偏移的鲁棒性研究极不充分。现有鲁棒性基准主要关注单模态（如 ImageNet-C 对图像）或不相关的多模态扰动。

**现有痛点**: (a) 现有基准仅对单模态施加腐蚀，或两个模态的腐蚀互不关联；(b) 真实场景中（如雨天自动驾驶）腐蚀同时影响音频和视频且相互关联；(c) 无人系统评估 SOTA 音视频模型在共现腐蚀下的表现。

**核心矛盾**: 音视频模型在干净数据上表现出色，但在双模态同时被腐蚀时的鲁棒性完全未知。

**本文目标**: 构建全面的音视频鲁棒性基准，系统评估最先进的监督/自监督模型和 TTA 方法。

**切入角度**: 设计 15 种真实世界启发的音视频共现关联腐蚀（5 个严重度级别），覆盖数字、环境、人为三大类别。

**核心 idea**: 真实世界的分布偏移同时影响音视频且具有关联性——现有模型和 TTA 方法在此条件下严重失效。

## 方法详解

### 整体框架

AVRobustBench 包含：
- **4 个基准数据集**: AudioSet-2C (16,742样本, 527类), VGGSound-2C (14,046样本, 309类), Kinetics-2C (3,111样本, 32类), EpicKitchens-2C (205样本, 97+300类)
- **75 种腐蚀**: 15 类 × 5 严重度级别
- **评估**: 6 个监督模型 + 3 个自监督模型 + 6 种 TTA 方法

### 关键设计

#### 腐蚀分类（15种，音视频同步施加）

| 类别 | 视频腐蚀 | 音频腐蚀 |
|-----|---------|---------|
| **数字** | Gaussian/Impulse/Shot/Speckle噪声 + JPEG压缩 | 对应噪声(SNR控制) + DCT量化 |
| **环境** | Snow/Frost/Spatter/Wind(运动模糊)/Rain/Underwater | 雪声/霜声/水滴/风声/雨声/水下音 |
| **人为** | Concert(亮度变化)/Smoke(灰雾)/Crowd(人影遮挡)/Interference(随机旋转) | 音乐噪声/警报声/人群噪声/随机静音 |

关键特点：所有腐蚀 **同时** 作用于音频和视频，且 **关联**（如雨天同时有雨滴画面和雨声）。

#### 评估指标
- **Accuracy/mAP**: 腐蚀后的分类准确率
- **绝对鲁棒性** $\alpha_{i,s} = 1 - \frac{\delta A}{100}$
- **相对鲁棒性** $\rho_{i,s} = 1 - \frac{\delta A}{A_{cl}}$，其中 $\delta A = A_{cl} - A_{i,s}$

#### AV2C — 提出的 TTA 方法
- 适应 QKV 注意力权重（类似 READ）
- 最小化加权 Shannon 熵，高权重给低熵（可靠）样本
- 基于当前预测与历史指数移动平均的相似度筛选多样样本

### 损失函数/训练策略

- 评估采用冻结预训练模型（标准鲁棒性基准方式）
- TTA 实验：batch=16，单次前向+后向
- AV2C：对 CAV-MAE 联合编码器的 QKV 权重进行在线适应

## 实验关键数据

### 主实验 — 测试时鲁棒性（严重度=5）

| 模型 | VGGSound-2C mAcc | 下降 | $\rho$ | Kinetics-2C mAcc | 下降 |
|-----|-----------------|------|--------|-----------------|------|
| UAVM | 27.41 | -38.39 | 0.42 | 48.06 | -30.06 |
| CAV-MAE | 35.54 | -29.96 | 0.54 | 58.15 | -29.95 |
| EquiAV | 33.78 | -28.12 | 0.55 | **63.73** | -22.29 |
| AudioCLIP | 11.14 | -15.64 | 0.41 | 23.57 | -27.44 |
| ImageBind | 10.25 | -17.93 | 0.36 | 26.82 | -25.64 |
| Wav2CLIP | 4.99 | -19.33 | 0.21 | 17.25 | -35.40 |

所有模型在严重度 5 下均出现 **显著性能下降**。

### TTA 实验（VGGSound-2C, 严重度=5）

| TTA 方法 | 平均准确率 | vs. Source |
|---------|----------|-----------|
| Source (CAV-MAE) | 35.54 | — |
| TENT | 19.09 | -16.45 |
| SAR | 26.07 | -9.47 |
| EATA | 40.60* | +5.06 |
| READ | 35.28 | -0.26 |
| SuMi | 32.10 | -3.44 |
| **AV2C (ours)** | **40.60** | **+5.06** |

注: EATA 和 AV2C 在 VGGSound-2C 上表现最佳。TENT 严重退化。

### 消融实验 — 腐蚀严重度影响

- 所有模型的 $\rho$ 随严重度增加单调下降
- 例外：Interference 腐蚀对鲁棒性影响较小（即使帧被重度旋转+音频静音，某些模型仍可识别）
- Digital 类腐蚀（Gaussian等）造成最大性能下降

### 关键发现

1. **监督模型**: EquiAV > CAV-MAE > UAVM，等变特征学习带来更好的鲁棒性
2. **自监督模型**: ImageBind 的零样本泛化在腐蚀下失效，$\rho$ 最低仅 0.21（Wav2CLIP on VGGSound）
3. **TTA 方法普遍失败**: TENT/RPL/SAR 的 LayerNorm 更新在双模态腐蚀下导致过拟合
4. **READ 的模态偏差**: 双模态腐蚀下，cross-attention 出现逐步加重的模态偏向（visual→audio 权重从 t=0 的 13.09 增到 t=100 的 20.04）
5. **Prompt 工程无效**: ImageBind 换用噪声感知 prompt 仅带来微乎其微的改善

## 亮点与洞察

- **首个共现关联腐蚀基准**: 填补了音视频鲁棒性评估的空白
- **全面的失败分析**: 系统暴露了监督/自监督模型及 TTA 方法在双模态腐蚀下的脆弱性
- **模态偏差的发现**: READ 中注意力权重在适应过程中的偏移是有趣的失败模式
- **AV2C 的简洁设计**: 低熵样本筛选 + QKV 适应，思路简单却有效

## 局限与展望

1. AV2C 仅在 VGGSound-2C 上有显著改善，Kinetics-2C 上仅持平
2. 腐蚀种类还可继续扩展（如网络延迟、编解码失真等）
3. 未评估大规模基础模型（如 VideoLLaMA）
4. 仅考虑在线 TTA，离线/少样本适应方案未涉及

## 相关工作与启发

- **ImageNet-C** (Hendrycks & Dietterich, 2019): 单模态视觉鲁棒性基准的开创性工作
- **READ** (2024): 首个音视频 TTA 方法，但在双模态腐蚀下失效
- **TENT** (Wang et al., 2020): 熵最小化 TTA 的基础方法
- **启发**: 多模态模型的鲁棒性不能简单从单模态推断——共现腐蚀带来了全新的挑战

## 评分

⭐⭐⭐⭐ (4/5)
基准构建扎实全面，实验覆盖面广，但提出的 AV2C 方法改进有限，更多是暴露问题而非解决问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[NeurIPS 2025\] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)
- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)
- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](../../ACL2025/audio_speech/audio_dialogue_benchmark.md)

</div>

<!-- RELATED:END -->
