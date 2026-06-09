---
title: >-
  [论文解读] Object-aware Sound Source Localization via Audio-Visual Scene Understanding
description: >-
  [CVPR 2025][音频/语音][声源定位] 本文提出 OA-SSL：在训练阶段用 MLLM 为每张图生成"K 个发声物 + 1 个静音物"的细粒度描述作为额外监督锚点，再用 OCA (object-aware contrastive alignment) 和 ORI (object region iso…
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "声源定位"
  - "MLLM 监督"
  - "Object-aware contrastive"
  - "Wasserstein 区域分离"
---

# Object-aware Sound Source Localization via Audio-Visual Scene Understanding

**会议**: CVPR 2025  
**arXiv**: [2506.18557](https://arxiv.org/abs/2506.18557)  
**代码**: [https://github.com/VisualAIKHU/OA-SSL](https://github.com/VisualAIKHU/OA-SSL)  
**领域**: 音频视觉学习 / 声源定位 / 多模态对比  
**关键词**: 声源定位、MLLM 监督、Object-aware contrastive、Wasserstein 区域分离

## 一句话总结
本文提出 OA-SSL：在训练阶段用 MLLM 为每张图生成"K 个发声物 + 1 个静音物"的细粒度描述作为额外监督锚点，再用 OCA (object-aware contrastive alignment) 和 ORI (object region isolation) 两个损失，让模型即使在画面里有多把吉他、只有一把在弹的复杂场景下也能只定位真正在发声的物体。

## 研究背景与动机
1. **领域现状**：音频-视觉声源定位 (AVSL) 通过自监督对比学习把音频和视觉特征对齐，定位画面中"在发声的物体"。
2. **现有痛点**：现有方法只做"音 ↔ 像素"相似度，无法区分**视觉上相似但声学上静音**的物体——例如画面里有 3 把吉他，只有 1 把在弹奏，模型会把所有吉他都高亮。
3. **核心矛盾**：自监督的 audio-visual correspondence 损失只能学到"哪类物体发了声"，学不到"这一刻具体哪个 instance 在发声"。
4. **本文目标**：
    - 引入额外的细粒度语义信号区分 sound-making vs silent；
    - 在多声源场景下让不同 instance 的定位区域之间互相分离；
    - 推理时仍保持音频+图像两支输入，不引入 MLLM 推理开销。
5. **切入角度**：MLLM (如 GPT-4V) 在多模态场景理解上具备"判断动作 / 状态"的能力，可以在训练时离线生成"playing guitar"vs"non-playing guitars and drum set"这样的描述。
6. **核心 idea**：把 MLLM 生成的"发声物 / 静音物"双向描述编码成 reference anchor，用对比 + Wasserstein 两路损失指导音视觉特征学习。

## 方法详解

### 整体框架
- **训练时**：图像 + 音频类别 → MLLM → 生成 $K$ 条 foreground caption (每个声源一条) + 1 条 background caption (静音物) → 文本编码器得到 reference 特征 $\mathbf{F}_r^p$ (前景, $B \times K \times c$) 与 $l_r^n$ (背景, $B \times c$)。
- **同时**：图像过视觉编码器得 $\mathbf{F}_v$，音频过音频编码器得 $l_a$，按余弦相似度生成 sound-associated map $\mathbf{S}_a$。
- **训练损失**：OCA loss + ORI loss + 原有自监督对比损失。
- **推理时**：完全去掉 MLLM 与文本分支，只用音视觉两路 + 学好的特征空间，所以推理代价不变。

### 关键设计

1. **Audio-Visual Scene Understanding (MLLM 生成监督)**

    - 功能：用 MLLM 生成发声物与静音物的描述作为 anchor。
    - 核心思路：精心设计 prompt，让 MLLM 输出 K 段 foreground caption (例 "a person is playing the leftmost guitar") 和 1 段 background caption (例 "two non-playing guitars and a drum set in the background")。这些 caption 通过文本编码器变成参考特征 $\mathbf{F}_r^p, l_r^n$。
    - 设计动机：MLLM 的常识能让它判断"playing"vs"holding"，弥补自监督方法在动作语义上的缺失。

2. **Object-aware Contrastive Alignment (OCA) Loss**

    - 功能：把"对应发声物"的视觉区域和 foreground anchor 拉近，把背景区域推远；同时反过来对静音物也做对称约束。
    - 核心思路：先按 $\mathbf{S}_a$ 用 sigmoid 阈值得到前景 mask $M^p$、背景 mask $M^n$，再 GAP 得到前景视觉特征 $l_v^p$ 和背景视觉特征 $l_v^n$。前景损失：
      $\mathcal{L}_{frg} = -\frac{1}{B}\sum \log \frac{p_i}{p_i + n_i^{hard} + n_i^{soft}}$
      其中 $p_i = \exp(\text{Sim}(l_v^p, l_r^p))$ 是正对，$n^{hard}$ 是同 batch 内 background-foreground 的 hard negative，$n^{soft}$ 是其他 batch 内不相似 ($\text{Sim}(l_{r_j}^p, l_{r_i}^p) \le \tau$) 的样本。背景损失 $\mathcal{L}_{bkg}$ 对称定义。最终 $\mathcal{L}_{oca} = (\mathcal{L}_{frg} + \mathcal{L}_{bkg})/2$。
    - 设计动机：单纯把"音-像素"拉近不够，要让模型"看到"silent 也是一种独立类别，且通过 false-negative 过滤避免同类发声物被错当负样本。

3. **Object Region Isolation (ORI) Loss**

    - 功能：在多声源场景中让不同声源的定位区域空间互斥。
    - 核心思路：把 K 个前景 ref 和 1 个背景 ref 拼成 $\mathbf{F}_r \in \mathbb{R}^{B \times (K+1) \times c}$，对每个 ref 与视觉特征算相似度图 $S_{r_k}$，然后用一阶 Wasserstein (Earth-Mover) 距离 + Sinkhorn 算法度量 $S_{r_n}$ 与 $1 - S_{r_m}$ 之间的距离：
      $\mathcal{L}_{ori} = \sum_i \sum_{n \neq m} D_W(\bar S_{r_n}^i, 1 - \bar S_{r_m}^i)$
    - 设计动机：单用 contrastive 不能保证两个发声物在像素空间不重叠 (例如同框小提琴 + 大提琴)；Wasserstein 距离自然刻画"区域分布的物理距离"，比 IoU 更平滑可微。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{base} + \lambda_1 \mathcal{L}_{oca} + \lambda_2 \mathcal{L}_{ori}$。MLLM 离线一次性生成所有训练 caption，缓存为文本特征，避免训练时反复调用 LLM。

## 实验关键数据

### 主实验
**MUSIC-Duet 多声源**：

| 方法 | Backbone | CAP(%) | CIoU@0.3 | AUC |
|------|----------|--------|----------|-----|
| Mix-and-Localize (CVPR22) | RN18 | 47.5 | 26.5 | 21.5 |
| AVGN (CVPR23) | RN18 | 50.6 | 32.5 | 24.6 |
| NoPrior (CVPR24) | RN18 | 52.1 | 38.6 | 30.1 |
| **OA-SSL (本文)** | RN18 | **61.4** | **45.9** | **36.1** |
| T-VSL (CVPR24) | AudioCLIP | 62.9 | 43.2 | 35.9 |
| **OA-SSL (本文)** | AudioCLIP | **更高** | **更高** | **更高** |

**VGGSound-Duet**：CIoU@0.3 从 NoPrior 的 46.9 → 本文 55.2 (+8.3)；AUC 从 29.2 → 44.8 (+15.6)。
**单声源 (MUSIC / VGGSound-Single)**：与最强 baseline 持平或略优。

### 消融实验

| 配置 | CIoU@0.3 (Duet) |
|------|-----------------|
| Baseline (no MLLM, no OCA, no ORI) | 38.6 |
| + OCA only | ~44 |
| + OCA + ORI (full) | 45.9 |
| Full but MLLM 给随机 caption | 大幅下降 |

### 关键发现
- 多声源场景增益 (+8 CIoU) 远大于单声源，证明本方法主要解决 "instance-level" 区分问题。
- OCA 提供"区分发声/静音"的能力，ORI 提供"分离不同声源"的能力，二者互补。
- MLLM caption 必须有"动作/状态"语义 (playing vs holding) 才有效，仅用类别名几乎无增益。

## 亮点与洞察
- **MLLM 作为离线监督生成器** — 把昂贵的 MLLM 用在训练阶段、推理阶段完全去掉，是非常实用的"知识蒸馏"策略，可推广到任何"自监督 + 缺细粒度标签"的任务。
- **同时建模发声 + 静音两类** — 大多数声源定位工作只关心"哪里发声"，本文显式让模型理解"哪里有相似但不发声的物体"，思路与"hard negative engineering"非常契合。
- **Wasserstein 区域分离损失** — 把 OT 距离用作多 instance 区域互斥约束，比传统 mask 互斥 / NMS 平滑可微，可迁移到多目标分割、多 referring expression 等任务。
- **对 false negative 的 soft 处理**：用 batch 内 ref 相似度做 thresholding，避免相似类别被误打成 negative。

## 局限与展望
- 严重依赖 MLLM 生成质量；当画面里物体类别罕见时 MLLM 可能输出错误前景/背景描述。
- 需要事先知道声源数量 $K$ (来自数据集标注)，对真实环境下未知声源数仍需迭代 / 估计模块。
- 未在 wild 视频 (含背景噪声、混响) 上充分评估。
- 文本编码器与 MLLM 的对齐度会显著影响效果，未来可联合训练或换用音频-文本对齐更好的 encoder (CLAP)。
- 改进方向：把 MLLM 用作 RL reward，让定位网络在 MLLM 反馈下迭代调整。

## 相关工作与启发
- **vs NoPrior (CVPR 24)**：NoPrior 用迭代识别处理多源但无细粒度语义；本文显式注入"动作语义"，定位精度更高。
- **vs T-VSL (CVPR 24)**：T-VSL 用 AudioCLIP 文本-音对齐做监督，本文进一步生成 instance-级别动作描述，监督更细。
- **vs Mix-and-Localize**：不再依赖人工设计的混合策略，监督信号来自 MLLM 自动生成。
- 启发：任何 fine-grained 的视觉/多模态任务，如果存在"易区分类别 + 难区分动作/状态"的 gap，都可以借鉴"MLLM 离线生成监督"的范式。

## 评分
- 新颖性: ⭐⭐⭐⭐ MLLM 生成 silent vs sound-making 的双向监督是新思路
- 实验充分度: ⭐⭐⭐⭐ MUSIC + VGGSound 单源/双源多 backbone 评测充分
- 写作质量: ⭐⭐⭐⭐ 损失定义清晰，公式化推导完整
- 价值: ⭐⭐⭐⭐ 推理零开销，效果显著，可作为多源 AVSL 新 SOTA baseline

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [\[CVPR 2025\] Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation](crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)
- [\[ICML 2025\] Sounding that Object: Interactive Object-Aware Image to Audio Generation](../../ICML2025/audio_speech/sounding_that_object_interactive_object-aware_image_to_audio_generation.md)
- [\[NeurIPS 2025\] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](../../NeurIPS2025/audio_speech/seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)

</div>

<!-- RELATED:END -->
