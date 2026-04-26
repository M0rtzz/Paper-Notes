---
title: >-
  [论文解读] FLAM: Frame-Wise Language-Audio Modeling
description: >-
  [ICML2025][语音][开放词汇声音事件检测] 提出 FLAM，一个帧级音频-语言对比模型，通过文本依赖的 logit 偏置校正和百万级合成 SED 数据集，实现开放词汇声音事件的精确时间定位，同时保持全局检索和零样本分类性能。
tags:
  - ICML2025
  - 语音
  - 开放词汇声音事件检测
  - 帧级对比学习
  - logit调整
  - 音频语言对齐
  - 数据增强
---

# FLAM: Frame-Wise Language-Audio Modeling

**会议**: ICML2025  
**arXiv**: [2505.05335](https://arxiv.org/abs/2505.05335)  
**代码**: [flam-model.github.io](https://flam-model.github.io/)  
**领域**: 音频语言模型 / 声音事件检测  
**关键词**: 开放词汇声音事件检测, 帧级对比学习, logit调整, 音频语言对齐, 数据增强

## 一句话总结

提出 FLAM，一个帧级音频-语言对比模型，通过文本依赖的 logit 偏置校正和百万级合成 SED 数据集，实现开放词汇声音事件的精确时间定位，同时保持全局检索和零样本分类性能。

## 研究背景与动机

- **现有 ALM 的局限**：CLAP 等音频语言模型学习实例级全局嵌入，擅长文本-音频检索，但无法精确定位声音事件的时间边界
- **传统 SED 的限制**：传统声音事件检测（SED）模型能精确定位事件，但局限于预定义类别，无法处理开放词汇的分布外事件
- **标注数据稀缺**：与图像领域不同，音频帧级标注数据极其匮乏，人工标注成本高昂；现有 SED 数据集规模小、类别有限
- **自监督方法不足**：先前的自监督局部对齐方法（如 MGA-CLAP）虽在一定程度上改善了帧级能力，但缺乏细粒度标注，定位效果有限

**核心问题**：如何在保持全局检索能力的同时，让 ALM 具备开放词汇的帧级声音事件定位能力？

## 方法详解

### 整体框架

FLAM 基于 LAION-CLAP 架构（HTSAT 音频编码器 + RoBERTa 文本编码器），扩展为同时输出全局嵌入和帧级嵌入序列。10秒音频输入经 HTSAT 后产生 $L=32$ 帧的嵌入序列 $\mathbf{e}^{a,loc}(x) \in \mathbb{R}^{L \times d}$，全局表示通过对帧嵌入取平均获得。

### 帧级对比损失

将开放词汇 SED 建模为帧级二分类任务，对每个（音频帧, 文本事件描述）对判断事件是否在该帧活跃：

$$\mathcal{L}_{\text{SED}} = -\frac{1}{BKL}\sum_{i=1}^{B}\sum_{k=1}^{K}\sum_{l=1}^{L}\log\sigma\big(z_{i,k,l}\cdot h(X_i, l, \mathcal{Y}_k)\big)$$

其中 logit 函数包含**文本依赖的缩放和偏置**：

$$h(x, l, y) = \alpha^t(y)\;\mathbf{e}^{a,loc}(x)_l \cdot \mathbf{e}^t(y) + \beta^t(y)$$

- $\alpha^t(y) > 0$：文本依赖的 logit 缩放，由 $\text{MLP}^\alpha(E^t(y))$ 产生
- $\beta^t(y)$：文本依赖的 logit 偏置，由 $\text{MLP}^p(E^t(y))$ 产生

### Logit 调整 — 处理事件依赖的类别不平衡

帧级标签严重不平衡（大部分帧–文本对为负样本），且不同事件的不平衡程度不同（如"雷声"出现少且持续短，"雨声"频繁且持续长）。

**贝叶斯最优分类器**：推理时使用无偏分类器，将原始预测归一化为与事件先验无关的定位得分：

$$s(x, l, y) = \frac{p(z=1|x,l,y)}{p(z=1|x,l,y) + p(z=1|y)} \approx \sigma\!\left(\log\frac{p(y|x,l)}{p(y)}\right)$$

**偏置训练**：单独训练 $\beta^t$ 来近似最优偏置 $\beta^*(y) = \log\frac{p(z=1|y)}{p(z=-1|y)}$，使用辅助损失 $\mathcal{L}_p$ 并阻断 $\mathcal{L}_{\text{SED}}$ 对其的梯度传播：

$$\mathcal{L}_p = -\frac{1}{K}\sum_{k=1}^{K}\big[\bar{z}_k\log\sigma(\beta^t(\mathcal{Y}_k)) + (1-\bar{z}_k)\log\sigma(-\beta^t(\mathcal{Y}_k))\big]$$

### 联合训练目标

$$\mathcal{L} = \gamma^{\text{CLIP}}\mathcal{L}_{\text{CLIP}} + \gamma^{\text{SED}}\mathcal{L}_{\text{SED}} + \gamma^p\mathcal{L}_p$$

其中 $\gamma^{\text{CLIP}}=1$，$\gamma^{\text{SED}}=200$，$\gamma^p=1$。

### 内存高效训练

采用 SigLIP 风格的分块 ring 传输策略：每个 GPU 处理本地子集的帧–文本对损失，文本嵌入通过环形传递在 GPU 间流转，避免在单个 GPU 上聚集所有嵌入，支持大 batch 训练。

### 数据增强管道

从 1.1M 音频样本（授权音效库 + CC 授权通用数据集）中合成 **100 万个 10 秒混合样本**：

1. 随机选取一个背景音频（≥10秒，含"ambiance"关键词）
2. 采样 $N \sim \mathcal{U}(1, 10)$ 个事件（80% 音效数据集，20% 通用数据集）
3. 随机放置事件，最多 3 个同时重叠
4. 10% 的事件分拆为 2-3 段，10% 重复 2-3 次，模拟真实场景
5. 随机响度偏移 $\mathcal{U}(6,30)$ dB，10ms 淡入淡出
6. 基于 A 加权 RMS 响度做边界校正（<-70 dB 标记为非活跃）

使用 Mixtral 为音效生成 2-13 词的 caption。

## 实验关键数据

### 声音事件检测（Table 1）

| 模型 | Held-out AUROC | ASFX-SED AUROC | DESED AUROC | MAESTRO MPAUC | AudioSet-S AUROC | UrbanSED AUROC |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| FLAM-Global | 67.76 | 65.14 | 85.52 | 51.13 | 82.54 | 67.39 |
| **FLAM** | **91.00** | **81.23** | **91.66** | **56.97** | **94.76** | **93.62** |
| MGA-CLAP* | 74.17 | 69.56 | 89.28 | 52.50 | 79.12 | 78.22 |

FLAM 在几乎所有指标上大幅超越基线，开放词汇 SED（Held-out）AUROC 从 67.76 提升至 91.00。

### 零样本分类（Table 3）

| 模型 | ESC-50 | US8K | VGGSound |
|------|:---:|:---:|:---:|
| FLAM-Global | 81.6 | 65.4 | 38.9 |
| **FLAM** | **86.9** | **75.6** | **39.3** |
| MGA-CLAP* | 72.6 | 69.9 | 38.6 |

帧级监督不仅不损害全局表示，反而提升了零样本分类准确率。

### 检索性能（Table 2）

FLAM 在检索任务上与 FLAM-Global 性能接近（AudioCaps T2A R@1: 32.1 vs 36.0），帧级训练对全局检索的影响极小。

## 亮点与洞察

1. **开放词汇 SED 的清晰建模**：将帧级事件定位建模为帧–文本对的二分类，继承对比学习框架，推理时预计算音频嵌入后只需编码新文本 query，效率高
2. **文本依赖的 logit 调整**：引入事件特定的缩放和偏置来处理事件级类别不平衡，使输出从原始余弦相似度转化为校准概率，理论推导严谨
3. **大规模合成数据管道**：百万级混合样本 + 精确边界标注，巧妙解决帧级标注稀缺问题，同时通过分拆/重复策略模拟真实场景
4. **帧级监督反哺全局表示**：零样本分类性能提升（ESC-50 81.6→86.9），证明细粒度对齐也能增强判别能力
5. **内存高效训练**：ring 传输避免中心化收集，使大规模帧级对比训练可行

## 局限与展望

1. **固定输入长度**：仅支持 10 秒音频，粗帧分辨率（32帧/10秒），难以处理长音频或需要更精细时间粒度的场景
2. **合成数据 vs 真实数据鸿沟**：训练数据为合成混合，真实场景中事件有更复杂的混响、遮掩和共现模式
3. **模型规模较轻**：HTSAT + RoBERTa 相对轻量，更大的编码器 / 更具表达力的架构可能带来进一步提升
4. **DESED 上 PSDS 不理想**：仅 692 个真实标注样本导致高方差，暗示在小规模真实标注场景下仍有不足
5. **检索性能有所下降**：AudioCaps T2A R@1 从 36.0 降至 32.1，帧级训练与全局检索目标存在一定 trade-off

## 相关工作与启发

- **CLAP / LAION-CLAP**：FLAM 的基础架构来源，扩展其从实例级到帧级对齐
- **SigLIP**：二元对比损失和 logit bias 的思想直接启发了 FLAM 的帧级目标设计
- **MGA-CLAP**：自监督局部对齐的代表，作为主要对比基线，但缺乏显式帧级监督
- **Scaper / 合成 SED 数据**：传统 SED 合成方法，FLAM 将其扩展到开放词汇场景
- **GLIP / PACL（视觉领域）**：图像域的开放词汇检测/分割工作，为音频域提供了类似思路

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将帧级对比学习与文本依赖 logit 调整结合用于开放词汇 SED，思路清晰且原创
- 实验充分度: ⭐⭐⭐⭐ — 覆盖开放/封闭 SED、检索、零样本分类多个维度，消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，符号体系一致，图示直观
- 价值: ⭐⭐⭐⭐ — 开放词汇帧级音频定位是实用且重要的方向，方法可推广至更广泛的音频理解任务

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](../../NeurIPS2025/audio_speech/efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ACL 2025\] Towards Reliable Large Audio Language Model](../../ACL2025/audio_speech/towards_reliable_large_audio_language_model.md)
- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](../../ACL2025/audio_speech/audio_dialogue_benchmark.md)

<!-- RELATED:END -->
