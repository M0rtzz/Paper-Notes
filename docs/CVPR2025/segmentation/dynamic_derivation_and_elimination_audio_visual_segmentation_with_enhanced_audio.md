---
title: >-
  [论文解读] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics
description: >-
  [CVPR 2025][图像分割][音频视觉分割] DDESeg 从音频的本质特性出发，针对混合音频的特征混淆和同物体不同声音的类内变异两大问题，提出动态推导模块从混合信号中衍生独立声源表征并增强判别性，再通过动态消除模块过滤掉画外音等无关音频语义，在 AVS 所有基准上取得 SOTA。
tags:
  - CVPR 2025
  - 图像分割
  - 音频视觉分割
  - 声源语义推导
  - 动态消除
  - 类内判别增强
  - 多模态对齐
---

# Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics

**会议**: CVPR 2025  
**arXiv**: [2503.12840](https://arxiv.org/abs/2503.12840)  
**代码**: 有（GitHub 链接见论文）  
**领域**: 分割 / 多模态VLM  
**关键词**: 音频视觉分割, 声源语义推导, 动态消除, 类内判别增强, 多模态对齐

## 一句话总结

DDESeg 从音频的本质特性出发，针对混合音频的特征混淆和同物体不同声音的类内变异两大问题，提出动态推导模块从混合信号中衍生独立声源表征并增强判别性，再通过动态消除模块过滤掉画外音等无关音频语义，在 AVS 所有基准上取得 SOTA。

## 研究背景与动机

**领域现状**：音频视觉分割（AVS）旨在根据音频信号定位并分割图像/视频中发出声音的物体。与文本或图像引导的分割不同，音频具有独特的性质——多个声源在时频域上重叠、同一物体可能发出差异巨大的声音。现有方法主要通过设计复杂的音视频交互架构来处理，较少关注音频本身的挑战。

**现有痛点**：(1) **特征混淆**：多个声源同时存在时，音频信号在频率、音色、时间上高度重叠（如马嘶与狗吠的频谱灰色区域大量重叠），提取的音频语义无法准确区分各声源；(2) **音视匹配困难**：同一物体的声音存在巨大类内变异（如猫可以发出嚎叫、咆哮、嘶嘶、喵喵等差异巨大的声音），导致同类声音的特征分布极为分散，模型难以将多样的声音一致地与同一视觉对象关联。

**核心矛盾**：现有方法（如 QDFormer 的量化分解、CPM 的音频分离）假设声音事件独立，试图从混合信号中直接分解或分离音频语义。但声源间共享频率，这些方法容易丢失关键语义细节或产生不完整的表征。

**本文目标** (1) 如何从混合音频中获取各声源的独立语义表征而不损失信息？(2) 如何增强音频表征的判别性以应对类内变异？(3) 如何过滤画外音等无法在视觉中匹配的音频语义？

**切入角度**：不分离音频信号，而是"推导"——利用预构建的语义记忆库，找到混合音频最近的 K 个类别中心，通过补偿差异来推导出各声源的独立语义表征。然后利用类内判别特征缩放增强表征的区分度。

**核心 idea**：推导而非分离——从混合音频中推导声源语义并增强判别性，再动态消除视觉无关的音频。

## 方法详解

### 整体框架

DDESeg 采用双分支框架进行多阶段音视频特征对齐。核心流程：(1) 音频编码器提取音频特征 $F_a$；(2) **动态推导模块 (DDM)** 从 $F_a$ 推导出 K 个独立声源语义表征 $\hat{A}$；(3) 在后续阶段，**动态消除模块 (DEM)** 根据视觉特征评估每个音频表征的相关性得分，抑制无关音频；(4) **特征融合模块**逐层对齐精炼后的音频特征与视觉特征；(5) 分割头输出最终预测。

### 关键设计

1. **动态推导模块 (Dynamic Derivation Module, DDM)**:

    - 功能：从混合音频特征推导出 K 个具有独立声源语义的音频表征
    - 核心思路：分三步——

        **Step 1 - 语义记忆构建**：用预训练音频模型提取单声源音频的特征，对每个类别做层次聚类，得到类别全局中心 $\mu^c$ 和子聚类判别特征 $x_{rep_j}^c$

        **Step 2 - 音频原型推导**：计算输入 $F_a$ 与各类别中心的距离，找 K 个最近的中心 $\{\mu_i\}_{i=1}^K$。受广义拉普拉斯算子启发，计算边特征 $e_{u_i} = \phi_{GELU}(W_e(\mu_i - F_a) + b_e)$，加权后得到补偿量 $\Delta a_{u_i}$，最终 $a_i = F_a + \Delta a_{u_i}$

        **Step 3 - 判别性增强**：利用类内判别特征 $x_{rep_j}^c$ 与推导表征 $a_i$ 的差异来学习缩放偏移 $\Delta a_{c_i,j}$，通过 $\hat{a}_i = a_i \odot (1 + \Delta a_{c_i,j})$ 增强判别性。注意 Step2 用加法做类间调整（跨类别的语义偏移），Step3 用乘法做类内增强（保持同类语义空间不变形）

    - 设计动机：不是分离信号（会丢失共享频率上的信息），而是从"知道世界上有哪些声音"的语义记忆出发，通过计算差异补偿来推导表征。类内判别增强解决了猫叫/猫嚎等大类内变异的问题

2. **动态消除模块 (Dynamic Elimination Module, DEM)**:

    - 功能：过滤掉与当前视觉画面不匹配的音频表征（如画外音）
    - 核心思路：首先用 Gumbel-Softmax 对视觉特征 V 做软聚类得到 K 个视觉语义中心 $C_v$。然后将音频表征 $\hat{A}$ 与视觉中心 $C_v$ 做多头交叉注意力（音频作 query）得到融合特征 $F_{av}$。通过 MLP + sigmoid 计算每个音频表征的相关性得分 $S \in [0,1]^k$，最后 $\hat{A} = S \cdot \hat{A}$ 来抑制不相关的音频
    - 设计动机：不是硬阈值过滤（会导致梯度截断），而是用可微的得分加权来软消除

3. **特征融合模块 (Feature Fusion)**:

    - 功能：逐层对齐音频和视觉特征
    - 核心思路：每个融合块用交叉注意力（音频 query + 视觉 key/value），卷积下采样，自注意力+FFN 捕获全局上下文
    - 设计动机：多阶段渐进对齐比一次性融合更精确

### 损失函数 / 训练策略

- 加权组合三项损失：$\mathcal{L} = \lambda_{dice}\mathcal{L}_{dice} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{iou}\mathcal{L}_{iou}$
- 权重分别为 5, 5, 2
- 使用 PVT-V2-B5 作为视觉骨干，VGGish 或 HTSAT 作为音频骨干

## 实验关键数据

### 主实验

AVS 数据集主结果（PVT-V2-B5 + VGGish/HTSAT）：

| 方法 | AVS-Objects-S4 ($\mathcal{J}\&\mathcal{F}$) | AVS-Objects-MS3 ($\mathcal{J}\&\mathcal{F}$) | AVS-Semantic ($\mathcal{J}\&\mathcal{F}$) |
|------|------|------|------|
| CAVP [CVPR24] | 90.5 | 72.7 | 55.3 |
| AVSStone [ECCV24] | 87.3 | 72.5 | 61.5 |
| BiasAVS [MM24] | 88.2 | 74.0 | 47.2 |
| **DDESeg (VGGish)** | **92.0** | **74.7** | **65.7** |
| **DDESeg (HTSAT)** | **94.2** | **77.9** | **67.9** |

DDESeg (HTSAT) vs 次优方法：S4 +3.7, MS3 +3.9, **Semantic +6.4**

### 消融实验

| 配置 | S4 $\mathcal{J}\&\mathcal{F}$ | MS3 $\mathcal{J}\&\mathcal{F}$ | 说明 |
|------|------|------|------|
| Baseline (无 DDM/DEM) | 88.1 | 70.2 | 直接音视交互 |
| + DDM Step1+2 (仅推导) | 90.3 | 72.8 | 推导多声源表征 |
| + DDM Step3 (加判别增强) | 91.4 | 74.0 | 类内判别增强有效 |
| + DEM (完整模型) | 92.0 | 74.7 | 消除无关音频进一步提升 |

### 关键发现

- DDM 的 Step2（推导）贡献最大（+2.2/+2.6），Step3（判别增强）进一步提升 +1.1/+1.2
- DEM 在 S4 上提升 +0.6，在需要区分多声源的 MS3 上提升更多
- 使用 HTSAT 替换 VGGish 作为音频编码器带来显著提升（Semantic +2.2），说明更强的音频特征对 AVS 至关重要
- 在 AVS-Semantic（最具挑战性的语义分割设置）上提升最为显著（+6.4），说明方法在需要精细语义区分时优势最大
- VPO 数据集上同样取得 SOTA，验证了方法的泛化性

## 亮点与洞察

- **推导而非分离**：跳出"分离混合音频"的范式，转为利用语义记忆"推导"各声源表征。这避免了分离方法在共享频率上的信息丢失，概念上更优雅也更稳健
- **类间加法 + 类内乘法**：Step2 用加法补偿做类间语义偏移（方便跳到不同类别的语义空间），Step3 用乘法缩放做类内增强（保持在同类语义空间内微调）。这种双重调整策略设计精巧
- **软消除机制**：用可微的相关性得分软加权替代硬阈值过滤，既能有效抑制画外音等干扰，又保持了端到端训练的梯度流
- **语义记忆的层次化构建**：对每个类别做子聚类并取最近质心的代表特征，有效捕获了类内变异模式

## 局限与展望

- 语义记忆库需要预构建，依赖单声源音频数据的可用性和覆盖度
- K（推导的声源数）是固定超参数，实际场景中声源数量是动态变化的
- 多阶段的 DDM→DEM→Fusion 流程增加了模型复杂度和推理开销
- 当画面中有声源但声音极微弱或被掩盖时，推导模块可能无法为其找到正确的语义中心
- 判别增强依赖语义记忆中的子聚类质量，对长尾类别可能效果有限

## 相关工作与启发

- **vs QDFormer**: QDFormer 假设声音事件独立，用量化分解来分离音频语义；DDESeg 不假设独立性，通过推导而非分解来处理重叠信号
- **vs CPM**: CPM 将音视融合特征解码为频谱图并用分离损失监督；DDESeg 在语义层面操作，不回到信号层面，避免了重建误差
- **vs CAVP**: CAVP 在 S4 上分数很高但 Semantic 上远低于 DDESeg，说明 CAVP 擅长二值定位但在语义区分上不足

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "推导+消除"的范式区别于现有分离/分解方法，从音频本质出发的问题分析深入
- 实验充分度: ⭐⭐⭐⭐ 覆盖 AVS 和 VPO 两大基准，消融清晰展示各模块贡献
- 写作质量: ⭐⭐⭐⭐ 问题动机图示非常清楚，方法描述详尽
- 价值: ⭐⭐⭐⭐ AVS-Semantic 上 6.4% 的提升说明方法实质性地推进了该领域

<!-- RELATED:START -->

## 相关论文

- [Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)
- [SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](../../ICCV2025/segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)

<!-- RELATED:END -->
