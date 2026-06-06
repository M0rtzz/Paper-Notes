---
title: >-
  [论文解读] CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation
description: >-
  [ECCV 2024][语义分割][图像分割] 提出 CPM（Class-conditional Prompting Machine），通过结合类无关查询与基于 GMM 采样的类条件查询来增强 Mask2Former 在音视频分割中的二部图匹配稳定性和跨模态注意力效力…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "图像分割"
  - "类条件提示"
  - "二部图匹配"
  - "Mask2Former"
  - "对比学习"
---

# CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.05358](https://arxiv.org/abs/2407.05358)  
**代码**: 无  
**领域**: 音视频分割 / 多模态学习  
**关键词**: Audio-Visual Segmentation, 类条件提示, 二部图匹配, Mask2Former, 对比学习

## 一句话总结

提出 CPM（Class-conditional Prompting Machine），通过结合类无关查询与基于 GMM 采样的类条件查询来增强 Mask2Former 在音视频分割中的二部图匹配稳定性和跨模态注意力效力，同时设计音频条件提示（ACP）、视觉条件提示（VCP）和提示对比学习（PCL）三个辅助任务，在 AVSBench 和 VPO 基准上达到 SOTA。

## 研究背景与动机

音视频分割（AVS）旨在根据音视频线索定位并分割发声物体，核心挑战在于有效的跨模态交互。

**现有方法的问题**：

**基于像素分类的方法**（TPAVI, CAVP 等）：使用早期融合 + FCN 解码器，因音频信息量低于视觉而低效利用音频；无法捕获实例级信息，导致帧间分割不一致

**基于 Transformer 的方法**（Mask2Former 类）：理论上更适合多模态，但面临两大训练困难：
   - **跨注意力效力低**：混合音源的全局音频特征语义不清，导致注意力学习困难
   - **二部图匹配不稳定**：类无关查询缺乏语义引导，匹配过程振荡

**核心洞察**：如果查询本身携带类别语义信息，既能稳定二部图匹配（因为查询已经知道要找什么），又能提供更清晰的跨模态注意力信号。

## 方法详解

### 整体框架

CPM 基于 Mask2Former 架构，训练时交替使用两条路径：
- **类无关路径（推理时使用）**：标准的类无关查询经 Transformer 解码器、匈牙利匹配到掩码预测
- **CPM 路径（仅训练时）**：从 GMM 采样的类条件查询经 ACP/VCP/PCL 三个辅助任务

### 关键设计 1：类条件分布建模（CCDM）

用高斯混合模型（GMM）对每个类别 c 的 mask embedding 分布进行建模：

p(q_tilde | c) = sum_m pi_cm * N(q_tilde; mu_cm, Sigma_cm)

- 训练过程中收集匈牙利匹配后的 mask embedding 及其对应标签
- 用 EM 算法优化 GMM 参数，动量更新保证稳定性
- 推理时用贝叶斯规则替代 Softmax 分类器

**作用**：从 GMM 采样类条件查询 z^k，这些查询天然携带类别语义，绕过二部图匹配中的不稳定性。

### 关键设计 2：音频条件提示（ACP）

受 mix-and-separate 思想启发，设计音频去噪任务：

1. 取训练音频 a_i，混入非画面噪声 a_j：a_p = a_i + a_j
2. 用类条件查询作为解码器输入，在混合音频特征图上检索语义相似的声源
3. 预测声谱图掩码并与真实频谱比率对齐

L_ACP = || sigma(sum_k m_k^a) - a_i / a_p ||_2

**作用**：迫使类条件查询学会在频域中区分不同声源，增强音频模态的细粒度理解。

### 关键设计 3：视觉条件提示（VCP）

直接用采样的类条件查询替代类无关查询输入 Transformer 解码器：
- 已知查询对应哪个类，无需二部图匹配
- 训练目标：正确分割对应图像区域 + 正确分类

L_VCP = L_ce + L_mask

**作用**：提供稳定的 per-class 训练信号，缓解类无关查询的匹配振荡。

### 关键设计 4：提示对比学习（PCL）

利用 ACP 产生的声谱图显著性掩码提取类级别音频特征，然后以该特征为锚点，与视觉特征图中同类像素（正样本）和异类像素（负样本）做 InfoNCE 对比学习。

**突破**：相比前人仅用全局音频做对比学习，CPM 首次实现类级别的音视频密集对比学习。

### 损失函数 / 训练策略

总损失：L = L_agn + lambda * L_CPM，其中 L_CPM = L_ACP + L_VCP + L_PCL

- L_agn：标准 Mask2Former 损失（匈牙利匹配 + 交叉熵 + focal + dice）
- CPM 分支仅在训练时激活；推理时只走类无关路径，不增加推理开销
- GMM 通过动量更新保持训练稳定

## 实验关键数据

### 主实验：AVSBench 测试集（224x224, ResNet-50）

| 方法 | 类型 | SS mIoU | MS mIoU | AVSS mIoU |
|:---:|:---:|:---:|:---:|:---:|
| TPAVI | Per-pixel | 78.80 | 52.84 | 22.69 |
| CAVP | Per-pixel | 83.84 | 61.48 | 32.83 |
| AVSegFormer | Transformer | 80.67 | 56.17 | 27.12 |
| COMBO | Transformer | 85.90 | 60.55 | 35.30 |
| **CPM** | **Transformer** | **85.92** | **65.40** | **37.05** |

CPM 在多源场景（MS）和语义场景（AVSS）上优势更大：MS +2.83%, AVSS +1.79%。

### 原始分辨率 AVSBench-Semantics

| 方法 | SS mIoU | MS mIoU | Total mIoU |
|:---:|:---:|:---:|:---:|
| CAVP | 56.91 | 38.61 | 50.75 |
| AVSegFormer* | 50.52 | 31.40 | 45.80 |
| **CPM** | **61.71** | **43.11** | **57.25** |

CPM 提升显著：SS +4.80, MS +4.50, Total +6.50 mIoU。

### 消融实验

| 组件 | 贡献说明 |
|:---:|:---|
| CCDM（GMM 建模） | 为 ACP/VCP/PCL 提供类条件查询的基础 |
| ACP | 增强音频模态的细粒度声源分离能力 |
| VCP | 绕过匈牙利匹配，提供稳定训练信号 |
| PCL | 密集的类级别音视频对比学习，增强跨模态对齐 |

### 关键发现

1. **CPM 对多源场景增益最大**：多声源时全局音频特征不可靠，类条件提示有效分解混合声源
2. **ACP 的 mix-and-separate 策略促进音频理解**：比直接用全局音频做查询更有效
3. **VCP 稳定训练**：类条件查询天然绕过匈牙利匹配的不稳定性
4. **PCL 实现首个类级别音视频对比学习**：突破了前人只能用全局音频做对比的限制

## 亮点与洞察

1. **训练-推理解耦**：CPM 分支仅参与训练，推理时零开销增加
2. **GMM 生成式分类器**：用生成式模型替代判别式 Softmax，更好地捕获类内变异性
3. **三重辅助任务设计精妙**：ACP/VCP/PCL 分别从音频、视觉、跨模态三个维度强化学习
4. **可扩展性**：CPM 可插入任何基于 Mask2Former 的分割框架

## 局限与展望

1. GMM 的 EM 更新增加训练复杂度，需维护外部记忆库
2. ACP 依赖非画面噪声数据集，数据获取有额外成本
3. 未在更强的 backbone（如 Swin-L）上充分验证
4. 声谱图掩码预测质量直接影响 PCL，但缺乏对这一中间结果质量的分析
5. 未探索视频时序信息的利用

## 相关工作与启发

- **Mask2Former / DETR**：CPM 基于 Mask2Former 架构并解决其训练不稳定问题
- **CAVP**：前一代 AVS SOTA，使用全局音频对比学习，CPM 将其推至类级别
- **DN-DETR**：通过去噪稳定二部图匹配的先驱，启发了 CPM 绕过匹配的思路
- **Mix-and-Separate**：音频领域的经典声源分离范式，被 ACP 巧妙引入 AVS

## 评分

- 新颖性: 4/5 - GMM + 类条件提示的组合在 AVS 中原创性强
- 实验充分度: 4/5 - 多基准评测全面，但消融细节可更充分
- 写作质量: 4/5 - 结构清晰，图 2 信息量大
- 价值: 4/5 - 为 Transformer-based AVS 的训练难题提供了系统性解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](../../CVPR2025/segmentation/robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[ECCV 2024\] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](../../ICCV2025/segmentation/implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ECCV 2024\] ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](controlnet_improving_conditional_controls_with_efficient_consistency_feedback.md)
- [\[ICML 2026\] LightAVSeg: Lightweight Audio-Visual Segmentation](../../ICML2026/segmentation/lightavseg_lightweight_audio-visual_segmentation.md)

</div>

<!-- RELATED:END -->
