---
title: >-
  [论文解读] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos
description: >-
  [CVPR 2026][视频生成][视频篡改检测] 首次提出活动级视频伪造定位任务和ActivityForensics大规模基准数据集（6K+伪造片段），通过grounding辅助的自动化数据构造管线制造高度逼真的活动篡改，并提出Temporal Artifact Diffuser (TADiff)基线方法…
tags:
  - "CVPR 2026"
  - "视频生成"
  - "视频篡改检测"
  - "活动级伪造"
  - "时间定位"
  - "扩散模型特征正则化"
  - "视频取证"
---

# ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos

**会议**: CVPR 2026  
**arXiv**: [2604.03819](https://arxiv.org/abs/2604.03819)  
**代码**: [https://activityforensics.github.io](https://activityforensics.github.io)  
**领域**: 视频生成  
**关键词**: 视频篡改检测, 活动级伪造, 时间定位, 扩散模型特征正则化, 视频取证

## 一句话总结
首次提出活动级视频伪造定位任务和ActivityForensics大规模基准数据集（6K+伪造片段），通过grounding辅助的自动化数据构造管线制造高度逼真的活动篡改，并提出Temporal Artifact Diffuser (TADiff)基线方法，通过扩散式特征正则化放大伪造线索。

## 研究背景与动机

**领域现状**：视频篡改定位旨在识别未裁剪视频中的篡改片段。现有基准（ForgeryNet、Lav-DF、AV-Deepfake1M、TVIL）主要关注外观级伪造（人脸替换、目标移除）。

**现有痛点**：随着视频生成技术（Wan、Sora、VACE等）的飞速进步，活动级伪造已成为新兴威胁——修改人物动作以扭曲事件语义（如将政客的中性站姿篡改为不当行为），高度逼真且极具欺骗性，严重威胁媒体真实性和公信力。但目前没有针对活动级伪造定位的基准。

**核心矛盾**：外观级伪造和活动级伪造的检测逻辑根本不同——前者依赖像素层面的纹理异常，后者需要理解动作语义变化与时间一致性。直接将动作定位模型迁移到伪造定位上会导致过度依赖语义信息。

**切入角度**：构建首个活动级伪造定位基准，利用视频描述和时间grounding自动化数据构造（避免高成本人工操作），同时提出针对性基线方法。

**核心idea**：(1) grounding辅助的自动化数据管线将篡改片段无缝嵌入原始视频；(2) TADiff通过注入噪声扰动抑制语义偏差，再通过扩散去噪放大伪造伪影线索。

## 方法详解

### 整体框架

**数据构造管线**：原始视频→视频描述+时间grounding（获取活动描述和时间段）→LLM修改描述（语义篡改）→视频生成/编辑模型合成篡改片段→无缝替换原始片段→产出精确时间标注。

**TADiff方法**：帧级特征提取→ActionFormer多尺度Transformer编码器→TADiff扩散式特征正则化→伪造置信度头+边界回归头。

### 关键设计

1. **Grounding辅助的数据构造**:

    - 功能：自动化生成活动级伪造视频和精确时间标注
    - 核心思路：利用视频描述和时间grounding模型自动定位活动片段，LLM将描述修改为语义篡改版本（如"挥手"→"竖拇指"），再用视频生成/编辑模型（Wan、Scifi、FCVG、Vidu、VACE、LTX）合成篡改片段并无缝融入
    - 设计动机：解决人工构造活动级伪造数据的高成本问题，同时确保高视觉一致性

2. **Temporal Artifact Diffuser (TADiff)**:

    - 功能：在时间特征空间中注入噪声再去噪，放大伪造伪影线索
    - 核心思路：
        - **前向过程**：向特征序列注入高斯噪声 $x_s = \sqrt{\bar{\alpha}_s} f + \sqrt{1-\bar{\alpha}_s} \epsilon$，扰动偏离语义流形
        - **反向过程**：轻量时间卷积去噪器（FiLM条件化），DDIM式更新 $x_{s-1} = \sqrt{\bar{\alpha}_{s-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{s-1}-\sigma_s^2}\hat{\epsilon} + \sigma_s z$
        - 去噪步数设为3步
    - 设计动机：动作定位模型的特征过度编码高层语义，对伪造检测所需的低层伪影线索（纹理不一致、运动不连续）不敏感。扩散过程通过注噪抑制语义偏差，去噪过程放大伪造敏感信号

3. **数据集统计**:

    - 6个篡改方法（4个视频生成+2个视频编辑）
    - 6K+伪造片段，分布均匀
    - 60%+的篡改片段占视频总时长<30%（定位挑战性高）
    - 三种评估设置：域内、跨域、开放世界

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{reg}$：focal loss（伪造置信度）+ smooth L1 loss（边界回归）。端到端训练，AdamW优化器，batch_size=16，lr=0.001。

## 实验关键数据

### 主实验（域内和开放世界）

| 设置 | 方法 | AP@0.75 | AP@0.95 | avg AP | avg AR |
|------|------|---------|---------|--------|--------|
| 域内 | ActionFormer | 86.29 | 46.79 | 70.67 | 74.31 |
| | UMMAFormer | 87.02 | 48.55 | 71.94 | 75.74 |
| | DiGIT | 78.61 | 44.92 | 64.69 | 70.43 |
| | **TADiff (Ours)** | **87.52** | **56.57** | **75.05** | **77.15** |
| 开放世界 | ActionFormer | 89.81 | 57.08 | 77.82 | 83.31 |
| | UMMAFormer | 91.13 | 57.57 | 78.79 | 84.15 |
| | **TADiff (Ours)** | **92.35** | **69.06** | **83.64** | **87.92** |

### 跨域实验（不同篡改方法之间迁移）

| 方向 | 方法 | avg AP | avg AR |
|------|------|--------|--------|
| A→B | ActionFormer | 67.18 | 72.14 |
| | **TADiff (Ours)** | **69.63 (+2.45)** | **74.91 (+2.77)** |
| B→A | ActionFormer | 37.14 | 51.03 |
| | **TADiff (Ours)** | **40.89 (+3.75)** | **52.56 (+1.53)** |

### 关键发现
- TADiff在高IoU阈值（AP@0.95）上改进最显著：域内+9.78，开放世界+11.98，说明扩散正则化特别有助于精确定位边界
- 跨域设置下B→A比A→B困难得多（avg AP仅40 vs 70），说明不同篡改方法之间的泛化是关键挑战
- 开放世界设置（训练集混合多种方法）性能最好，说明多样化训练对泛化有益
- DiGIT（原始视频取证方法）在活动级伪造上表现不佳，验证了活动级和外观级伪造检测的根本差异

## 亮点与洞察
- **新任务定义**：首次形式化活动级伪造定位任务，与外观级伪造形成互补。随着视频生成模型快速发展，这一任务的现实意义越来越大
- **自动化数据管线**：grounding辅助的数据构造方法避免了人工标注的高成本，且确保了篡改片段与上下文的视觉一致性，可扩展到更多视频生成模型
- **扩散式特征正则化**：TADiff的核心洞察——通过注噪打断语义编码、去噪放大伪影信号——简洁且有效，可迁移到其他低层线索敏感的检测任务

## 局限与展望
- 当前篡改方法限于6种，但Sora等未支持受控起止帧的模型未纳入，未来需持续扩展
- TADiff是在ActionFormer基础上的轻量改进，更深层的取证架构设计（如结合光流、频域分析）值得探索
- 活动级伪造定位依赖视觉伪影线索，当视频生成质量进一步提升后，这些线索可能消失，需要向更高层的时间-语义一致性检测发展
- 跨域泛化能力仍有限（B→A方向avg AP仅40），需要更强的域不变特征学习

## 相关工作与启发
- **vs ForgeryNet/Lav-DF**: 它们关注人脸伪造（外观级），本文关注活动伪造（语义级），检测逻辑根本不同
- **vs TVIL**: TVIL关注时间视频修复定位（目标移除），本文关注活动修改，伪造类型更隐蔽
- **vs ActionFormer**: 动作定位架构被直接用于伪造定位，但语义偏差导致性能受限，TADiff通过特征正则化解决了这一问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个活动级伪造定位基准，任务定义有前瞻性
- 实验充分度: ⭐⭐⭐⭐⭐ 三种评估协议、多种SOTA基线、跨域迁移分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，数据构造管线和方法设计描述完整
- 价值: ⭐⭐⭐⭐⭐ 高度时效性，随视频AI生成能力增长，该任务的重要性将持续提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](../../ICML2026/video_generation/explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[ACL 2025\] VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation](../../ACL2025/video_generation/vidcapbench_a_comprehensive_benchmark_of_video_captioning_for_controllable_text-.md)
- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)

</div>

<!-- RELATED:END -->
