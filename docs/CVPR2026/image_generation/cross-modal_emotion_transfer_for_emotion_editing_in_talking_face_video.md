---
title: >-
  [论文解读] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video
description: >-
  [CVPR 2026][图像生成][情感编辑] 提出 C-MET（Cross-Modal Emotion Transfer），通过建模语音和面部表情空间之间的情感语义向量映射，首次实现了基于语音驱动的扩展情感（如讽刺、魅力）说话人脸视频生成，情感准确率超越 SOTA 14%。
tags:
  - CVPR 2026
  - 图像生成
  - 情感编辑
  - 跨模态迁移
  - 说话人脸生成
  - 情感语义向量
  - 扩展情感
---

# Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video

**会议**: CVPR 2026  
**arXiv**: [2604.07786](https://arxiv.org/abs/2604.07786)  
**代码**: [https://chanhyeok-choi.github.io/C-MET/](https://chanhyeok-choi.github.io/C-MET/) (有)  
**领域**: 图像生成 / 说话人脸  
**关键词**: 情感编辑, 跨模态迁移, 说话人脸生成, 情感语义向量, 扩展情感

## 一句话总结
提出 C-MET（Cross-Modal Emotion Transfer），通过建模语音和面部表情空间之间的情感语义向量映射，首次实现了基于语音驱动的扩展情感（如讽刺、魅力）说话人脸视频生成，情感准确率超越 SOTA 14%。

## 研究背景与动机

**领域现状**：情感说话人脸生成是生成模型的核心应用，目标是将中性说话视频转换为带有目标情感的视频。现有方法按情感源分为三类：标签驱动、语音驱动、图像驱动。

**现有痛点**：(1) 标签驱动方法仅支持预定义的离散情感类别（如 8 种基础情感），无法表示复杂/微妙的情感；(2) 语音驱动方法中情感与语言内容纠缠，无法分离；(3) 图像驱动方法需要高质量正面参考图，扩展情感（如讽刺）的参考数据难以获取。

**核心矛盾**：如何在不收集额外标注数据的情况下，利用丰富的语音情感信息来驱动面部表情生成，特别是训练中未见过的扩展情感？

**本文要解决**：跨模态（语音→视觉）的情感迁移，同时支持扩展情感的零样本生成。

**切入角度**：不直接预测面部表情，而是学习"情感语义向量"——即两种不同情感嵌入之间的差值——在语音空间和视觉空间之间的映射关系。

**核心 idea**：情感语义向量 = 目标情感嵌入 − 输入情感嵌入，通过跨模态 Transformer 学习从语音语义向量到视觉语义向量的映射。

## 方法详解

### 整体框架
C-MET 由三个部分组成：(a) 预训练编码器提取语音/视觉嵌入并计算语义向量；(b) 多模态 token 对比学习对齐表征空间；(c) Transformer 编码器回归目标视觉语义向量，解码器重建情感视频。

### 关键设计

1. **情感语义向量（Emotion Semantic Vector）**：

    - **功能**：给定输入情感 $i$ 和目标情感 $j$，在语音空间计算 $f_a^{i \to j} = f_a^j - f_a^i$，在视觉空间计算 $f_v^{i \to j} = f_v^j - f_v^i$。
    - **核心思路**：用差值向量表示情感变化方向，而非直接建模绝对情感——这使模型学到的是"情感迁移"而非"情感识别"。
    - **设计动机**：灵感来自 EmoKnob 的语音情感控制。差值向量在连续空间中具有良好的可组合性，使得训练时只需基础情感，推理时可泛化到未见过的扩展情感。

2. **多模态 Token 对比学习**：

    - **功能**：用 1D 卷积构建视觉 tokenizer $T_v$，投影层构建音频 tokenizer $T_a$，通过双向对比损失对齐两个模态的 token 表征。
    - **核心思路**：
    $L_{\text{cnt}} = \frac{L_{v \to a} + L_{a \to v}}{2}$
    - **设计动机**：缩小语音和面部表情两个模态之间的表征差距，使跨模态回归更准确。

3. **跨模态 Transformer 编码器**：

    - **功能**：将参考视觉语义向量 $z_r$、目标语音语义向量 $z_a$、输入视觉嵌入 $z_v$ 拼接为 token 序列，送入 Transformer 编码器，预测目标视觉语义向量：
    $\hat{f}_{v,t}^{i \to j} = P_v(TE(\{z_{r,t'}\} \| \{z_a\} \| \{z_{v,t}\}))$
    - **核心思路**：预测的语义向量加上输入视觉嵌入即得目标嵌入，送入预训练解码器生成情感视频。
    - **设计动机**：分别引入三种类型嵌入（type embedding）区分不同来源，使 Transformer 能有效建模跨模态依赖。

### 损失函数 / 训练策略
- 重建损失（双向）：$L_{\text{recon}} = L_{i \to j} + L_{j \to i}$
- 方向损失：$L_{\text{dir}} = 1 + \frac{\langle \hat{f}_v^{i \to j}, \hat{f}_v^{j \to i} \rangle}{\|\hat{f}_v^{i \to j}\| \|\hat{f}_v^{j \to i}\|}$（确保正反向向量方向相反）
- 总损失：$L = L_{\text{recon}} + \lambda_{\text{cnt}} \cdot L_{\text{cnt}} + \lambda_{\text{dir}} \cdot L_{\text{dir}}$
- $\lambda_{\text{cnt}} = 0.1$, $\lambda_{\text{dir}} = 0.05$

## 实验关键数据

### 主实验

| 方法 | 情感源类型 | Acc_emo↑ (MEAD) | Acc_emo↑ (CREMA-D) | FID↓ | AITV↓ |
|------|----------|----------------|-------------------|------|-------|
| EAMM | 图像 | 18.81 | 19.15 | 161.6 | 3.745 |
| EAT | 标签 | 41.56 | 39.97 | 91.0 | 12.575 |
| EDTalk | 图像 | 41.99 | 29.69 | **76.4** | 2.827 |
| FLOAT | 语音 | 13.21 | 29.11 | 92.8 | 1.434 |
| **C-MET** | **语音** | **55.91** | **43.47** | 90.8 | 2.643 |

### 消融实验

| 损失配置 | Acc_emo↑ (MEAD) | 说明 |
|---------|----------------|------|
| $L_{\text{recon}}$ only | 49.43 | 基线 |
| + $L_{\text{cnt}}$ | 53.46 | 对比学习贡献 +4% |
| + $L_{\text{cnt}}$ + $L_{\text{dir}}$ | **55.91** | 方向损失进一步 +2.4% |

即插即用验证：

| 骨干网络 | 原始 Acc_emo | + C-MET Acc_emo | AITV 变化 |
|---------|------------|----------------|----------|
| PD-FGC | 33.36 | 36.82 (+3.46) | 1.247→1.180（更快） |
| EDTalk | 41.99 | **55.91 (+13.92)** | 2.827→2.643（更快） |

### 关键发现
- 情感准确率提升显著（比 SOTA 高 14%），但在 FID/FVD 等视觉质量指标上略有让步——这反映了情感表达强度和视觉保真度之间的固有 trade-off
- 用户研究中 C-MET 在基础情感和扩展情感设定下均获得压倒性偏好（>75%）
- C-MET 可作为即插即用模块替换重的表情编码器，同时降低推理时间

## 亮点与洞察
- **首创性**：首个显式建模语音-视觉情感语义向量映射的方法
- **扩展情感零样本**：训练仅用 MEAD 的 8 种基础情感，推理时可生成讽刺、魅力等从未见过的扩展情感
- **即插即用**：可无缝集成到现有解耦网络（EDTalk、PD-FGC）中，替换重量级表情编码器

## 局限与展望
- 视觉质量指标（FID、FVD）略逊于图像驱动方法，强烈的情感表达导致较大的面部运动偏差
- 依赖预训练的 emotion2vec+large 和 EDTalk 编码器/解码器，模型的天花板受限于这些组件的能力
- 扩展情感的定量评估缺乏标准 benchmark，目前只能通过用户研究验证

## 相关工作与启发
- 与 FLOAT（语音驱动但情感-内容纠缠）的对比验证了解耦设计的必要性
- EmoKnob 的语音情感控制思路被巧妙推广到视觉生成领域
- 对比学习用于模态对齐的策略可推广到其他跨模态迁移任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 情感语义向量映射的思路新颖，扩展情感零样本生成具有开创性
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+用户研究+消融完整，但扩展情感缺乏标准评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但符号略多
- 价值: ⭐⭐⭐⭐⭐ 具有实际应用价值，解决了情感说话人脸生成的关键瓶颈

<!-- RELATED:START -->

## 相关论文

- [DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)
- [EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](../../CVPR2025/image_generation/emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)
- [Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [Multi-Aspect Cross-modal Quantization for Generative Recommendation](../../AAAI2026/image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)

<!-- RELATED:END -->
