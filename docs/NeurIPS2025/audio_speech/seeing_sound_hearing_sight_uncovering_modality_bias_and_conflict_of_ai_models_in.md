---
title: >-
  [论文解读] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization
description: >-
  [NeurIPS 2025][语音][声源定位] 系统性地揭示了AI声源定位(SSL)模型存在严重视觉偏见——在视听冲突时降到随机水平，提出神经科学启发的EchoPin模型（HRTF滤波+耳蜗图+立体声），在AudioCOCO数据集上大幅超越现有方法并展现出类人的水平>垂直定位精度偏差。
tags:
  - NeurIPS 2025
  - 语音
  - 声源定位
  - 模态偏见
  - 跨模态冲突
  - 神经科学启发
  - HRTF
  - 耳蜗图
---

# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization

**会议**: NeurIPS 2025  
**arXiv**: [2505.11217](https://arxiv.org/abs/2505.11217)  
**代码**: 论文声明公开（GitHub）  
**领域**: audio_speech / multimodal  
**关键词**: Sound Source Localization, Modality Bias, Cross-modal Conflict, HRTF, Cochleagram, AudioCOCO

## 一句话总结

通过6种受控视听条件和人类心理物理实验，系统揭示现有AI声源定位模型存在严重视觉偏见（视听冲突时降至随机水平），并提出神经科学启发的EchoPin模型——HRTF滤波+ERB耳蜗图+立体声，在自建AudioCOCO数据集上大幅超越现有方法，且无需人类行为监督即涌现出类人的水平>垂直定位精度不对称性。

## 研究背景与动机

**领域现状**：声源定位（SSL）是将声音与视觉场景中的空间位置对应的基础多模态任务。近年来对比学习（DenseAV等）和跨模态注意力（Transformer架构）等方法在标准一致性条件下取得了不错效果，但几乎所有研究都在"音视频语义和空间一致"的理想条件下评测。

**现有痛点**：(1) 现有数据集严重偏置——单一大物体居中，模型可通过视觉捷径学习（不需要真正整合音频信息）就达到不错效果；(2) 绝大多数方法使用单声道（mono）音频，丧失了双耳空间线索；(3) 缺乏对模型在视听冲突、模态缺失等非理想条件下行为的系统研究。

**核心矛盾**：当视听信号冲突时（如听到的是狗叫但看到的声源位置是一辆车），人类可以灵活地优先信任听觉来准确定位，但AI模型是否也能做到？还是会被视觉误导？

**本文目标**：(1) 量化AI模型在各种视听条件下的模态偏见；(2) 提供人类基线对照；(3) 构建无偏数据集和生物启发模型来弥合人-机差距。

**切入角度**：从神经科学角度切入——模仿人类听觉外周处理链路（耳廓HRTF→耳蜗频率分解→双耳空间线索），设计数据合成管线和模型前端。

**核心 idea**：用HRTF空间滤波+耳蜗图频率分解+立体声配置来模拟人类听觉外周，消除数据偏置的同时赋予模型真正利用空间听觉线索的能力。

## 方法详解

### 整体框架

EchoPin系统包含三个层面：(1) **AudioCOCO数据集**——基于Unity 3D模拟器合成深度感知的空间化立体声，配合6种受控实验条件；(2) **人类心理物理实验**——14名被试在相同条件下提供人类基线；(3) **EchoPin模型**——HRTF滤波→ERB耳蜗图→双编码器+对比学习。输入为静态图像+双通道立体声音频，输出为图像上的声源位置预测。

### 关键设计

1. **AudioCOCO数据集与深度感知立体声合成**
    - 功能：提供高质量、空间平衡、无偏置的音视频训练/测试数据
    - 核心思路：从MSCOCO选取12类可发声物体，按目标面积占比分3个尺寸档（Size1: 0–5%, Size2: 5–15%, Size3: 15–30%），排除>30%的trivial样本。使用DepthAnything估计深度后，在Unity中按物体像素位置+深度构建3D空间，模拟0.17m双耳间距的听者，基于物理声传播合成空间化立体声。测试集设计6种条件：一致(Congruent)、视觉冲突(ConflictVCue)、视觉缺失(AbsVCue)、仅音频(AOnly)、仅视觉(VOnly)、多实例(MultiInstLoc)
    - 设计动机：现有数据集（FlickrSoundNet, VGGSound等）存在大物体居中偏置，导致模型学到视觉捷径而非真正的音视频对齐；同时大多使用单声道，无法提供空间定位线索

2. **HRTF滤波 + ERB耳蜗图前端**
    - 功能：将原始立体声波形转换为忠于人类听觉外周的表示
    - 核心思路：首先用KEMAR假人头数据集的HRTF对立体声进行方向相关的频谱滤波（编码双耳时间差ITD和水平差ILD），然后用66个ERB等效矩形带宽滤波器将HRTF滤波后的16kHz立体声转换为耳蜗图（66×160,000×2张量），保留音高、音色和空间线索
    - 设计动机：传统mel-spectrogram丢失了HRTF引入的细粒度频谱特征和双耳空间线索，ERB滤波器组更忠实地模拟耳蜗的频率选择性和时间动态

3. **双编码器对比学习架构**
    - 功能：将听觉和视觉特征分别编码后进行语义对齐和空间对齐
    - 核心思路：基于IS3的双流2D CNN架构，用1D卷积核先整合双耳通道信息，再分别通过视觉/音频编码器提取特征图，计算余弦相似度热力图定位声源。训练用Triplet Loss（拉近匹配的音视频嵌入）+ CIoU Loss（惩罚预测框与真实框的空间偏差）
    - 设计动机：分离编码允许模型独立学习各模态特征，对比学习天然适合跨模态对齐任务；CIoU Loss直接监督空间定位精度

### 损失函数 / 训练策略

- **Triplet Loss**：语义对齐——拉近匹配音视频对的嵌入距离，推远不匹配对
- **CIoU Loss**：空间对齐——惩罚预测声源位置边界框与真实标注的空间偏差
- 从预训练IS3模型初始化除第一层1D卷积外的所有权重，端到端微调
- 仅在Congruent条件下训练，6种条件均用于测试，评估泛化能力

## 实验关键数据

### 主实验

多实例条件下的音频定位精度 A-Acc：

| 模型 | Size1 | Size2 | Size3 |
|------|-------|-------|-------|
| Random | 1.6% | 9.1% | 21.3% |
| IS3 | 4.8% | 7.9% | 22.4% |
| CAVP | 2.9% | 7.5% | 20.4% |
| AVSegformer | 2.5% | 7.3% | 20.2% |
| **EchoPin** | **4.5%** | **24.1%** | **47.1%** |
| Human | 25.7% | 36.4% | 38.6% |

### 消融实验

Mono vs Stereo + 耳蜗图 vs Mel 的A-Acc对比（Congruent+ConflictVCue+AbsVCue+AOnly平均）：

| 配置 | Size1 | Size2 | Size3 |
|------|-------|-------|-------|
| IS3 (mono, mel, 标准数据集) | 3.0% | 13.9% | 28.7% |
| EchoPin-M (mono, 耳蜗图, AudioCOCO) | 3.6% | 15.8% | 31.4% |
| EchoPin-S (stereo, mel, AudioCOCO) | 5.3% | 17.0% | 35.2% |
| **EchoPin (stereo, 耳蜗图, AudioCOCO)** | **9.7%** | **31.3%** | **47.6%** |

### 关键发现

- **视觉偏见严重**：IS3/CAVP等在ConflictVCue条件下A-Acc降至接近随机，而人类和EchoPin仍显著高于随机
- **AI不能仅靠听觉定位**：AOnly条件下IS3基本失败，EchoPin保持一定能力但仍远不如人类
- **视觉捷径暴露**：VOnly（无声音）条件下AI模型V-Acc仍高于随机——模型倾向于定位"看起来会发声的物体"（人、动物），而非真正利用音频
- **立体声是关键**：EchoPin比EchoPin-M（mono）在Size3上高16.2个百分点
- **耳蜗图优于Mel**：EchoPin比EchoPin-S（mel）在Size3上高12.4个百分点
- **涌现类人不对称性**：EchoPin展现出水平定位精度（86.1%试次在6°内）>垂直定位精度的类人模式，源于HRTF编码的双耳水平面空间线索更强；旋转耳轴90°（EchoPin-Ro）后不对称性反转，验证了结构成因

## 亮点与洞察

- **首次系统量化SSL模态偏见**：6种受控条件×3个物体尺寸×8+模型+人类基线，实验设计极其完备
- **神经科学→工程的优雅闭环**：从生物听觉外周（耳廓HRTF→耳蜗ERB）出发设计模型前端，不仅提升性能还涌现出无需人类行为监督的类人行为——这是当前AI多模态模型中罕见的
- **数据偏置的深刻诊断**：证明了现有SSL数据集的"大物体居中"偏置是视觉捷径学习的根源，AudioCOCO通过尺寸分档和空间多样化系统性消除了这一问题
- **EchoPin-Ro实验**：通过旋转双耳轴向验证水平-垂直不对称性的结构成因，实验设计巧妙，增强了因果推断的可信度

## 局限与展望

- **小目标差距大**：Size1上EchoPin仅4.5–9.7%，人类25.7%，差距4-5倍，说明模型对微弱空间线索的利用仍严重不足
- **冲突条件仍受误导**：EchoPin在ConflictVCue下性能下降比AbsVCue更严重，说明模型尚未学会像人类一样在冲突时主动抑制视觉
- **合成数据局限**：Unity合成的立体声与真实录制仍有差距，未考虑声音折射、环境混响等复杂声学效应
- **静态场景**：仅处理静态图像+合成音频，未涉及视频中的动态声源和时序线索
- **物体类别有限**：仅12类可发声物体，开放世界场景的泛化能力待验证

## 相关工作与启发

- **vs IS3/CAVP/AVSegformer**：这些方法在标准数据集上表现尚可，但都严重依赖视觉捷径，视听冲突时暴露出本质缺陷；EchoPin通过数据+模型双管齐下解决
- **vs ImageBind/LanguageBind**：大规模预训练多模态模型在SSL任务上也未展现优势，说明规模不能替代任务针对性的感官建模
- **对VLM研究的启示**：多模态大模型是否也存在类似的模态偏见？本文的受控实验范式可直接迁移到其他多模态任务
- **听觉前端设计的迁移价值**：HRTF+ERB耳蜗图管线可应用于音视频分离、空间音频生成、机器人听觉感知等任务
- **AudioCOCO的方法论价值**：基于3D模拟器+深度估计的空间化音频合成管线，可推广到任何需要音视频空间对齐的数据集构建

## 评分

- **新颖性**: ⭐⭐⭐⭐ 理由：神经科学启发的SSL模型+系统性模态偏见分析框架具有很高原创性，但核心模型架构基于IS3改造，增量性质
- **实验充分度**: ⭐⭐⭐⭐⭐ 理由：6种受控条件×3个尺寸×8+模型+人类心理物理实验+多种消融+EchoPin-Ro验证实验，实验设计堪称教科书级别
- **写作质量**: ⭐⭐⭐⭐ 理由：结构清晰、图表丰富、实验条件描述严谨，但论文整体偏长
- **价值**: ⭐⭐⭐⭐ 理由：揭示了多模态模型的深层缺陷并提供了兼具科学洞察和工程价值的解决方案，对多模态AI研究有广泛启发意义
---
title: >-
  [论文解读] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization
description: >-
  [NeurIPS 2025][语音][声源定位] 系统性地揭示了AI声源定位(SSL)模型存在严重视觉偏见——在视听冲突时降到随机水平，提出神经科学启发的EchoPin模型（HRTF滤波+耳蜗图+立体声），在AudioCOCO数据集上大幅超越现有方法并展现出类人的水平>垂直定位精度偏差。
tags:
  - NeurIPS 2025
  - 语音
  - 声源定位
  - 模态偏见
  - 跨模态冲突
  - 神经科学启发
  - HRTF
  - 耳蜗图
---

# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization

**会议**: NeurIPS 2025  
**arXiv**: [2505.11217](https://arxiv.org/abs/2505.11217)  
**代码**: [GitHub](https://github.com/) (论文声明公开)  
**领域**: audio_speech / multimodal  
**关键词**: 声源定位, 模态偏见, 跨模态冲突, 神经科学启发, HRTF, 耳蜗图  

## 一句话总结
系统性地揭示了AI声源定位(SSL)模型存在严重视觉偏见——在视听冲突时降到随机水平，提出神经科学启发的EchoPin模型（HRTF滤波+耳蜗图+立体声），在AudioCOCO数据集上大幅超越现有方法并展现出类人的水平>垂直定位精度偏差。

## 背景与动机

1. **领域现状**：声源定位(SSL)是多模态感知的基础任务，将听到的声音与视觉场景中的声源位置对应。现有多模态模型（对比学习、跨模态注意力等）在标准一致性条件下表现不错。
2. **核心问题**：几乎没有研究系统性地检验这些AI模型在视听冲突时的行为——是否会像人类一样优先信任听觉？还是存在视觉偏见？
3. **关键发现**：人类在视听冲突甚至无视觉条件下仍能稳健定位声源（优先听觉），而AI模型严重偏向视觉，冲突时性能暴跌至随机水平。

## 方法详解

### 整体框架
三大贡献：(1) AudioCOCO数据集 + 6种实验条件；(2) 人类心理物理实验基线；(3) EchoPin神经科学启发模型。

### 关键设计1: AudioCOCO数据集
- 从MSCOCO选12类可发声物体，按目标占比分3个尺寸档（Size1: 0-5%, Size2: 5-15%, Size3: 15-30%）
- 用Unity 3D模拟器 + DepthAnything深度估计，根据物体在图像中的像素位置和深度，合成**空间化立体声音频**
- 训练集4,953图 → 9,360音-图对；测试集5,500图 → 18,864对
- **6种实验条件**：一致(Congruent)、视觉冲突(ConflictVCue)、视觉缺失(AbsVCue)、仅音频(AOnly)、仅视觉(VOnly)、多实例(MultiInstLoc)

### 关键设计2: EchoPin模型
- **HRTF滤波**：用KEMAR假人头数据集的头相关传递函数，模拟耳廓/头颅/躯干对声波的方向性滤波
- **耳蜗图(Cochleagram)**：用ERB等效矩形带宽滤波器组将HRTF滤波后的立体声波形转换为耳蜗图表示（66通道×160k时间步×2双耳），比mel-spectrogram更忠于听觉外周处理
- **双编码器架构**：基于IS3的2D CNN双流架构，视觉和听觉流独立编码后融合
- **训练损失**：Triplet Loss（语义对齐）+ CIoU Loss（空间对齐）

### 关键设计3: 人类心理物理实验
- 14名被试，2,100次试验，实验室内佩戴立体声耳机
- 与AI在完全相同的6种条件下对比

## 实验关键数据

### 一致条件下A-Acc（Size2, 声源在目标框内）

| 模型 | Size1 | Size2 | Size3 |
|------|-------|-------|-------|
| Random | 1.6% | 9.1% | 19.8% |
| IS3 | 4.8% | 7.9% | 22.4% |
| **EchoPin** | **4.5%** | **24.1%** | **47.1%** |
| Human | 25.7% | 36.4% | 38.6% |

### 关键对比发现
- **视觉冲突(ConflictVCue)**：IS3降到接近随机水平，EchoPin仍显著高于随机
- **仅音频(AOnly)**：人类仍能定位，IS3基本失败，EchoPin保持一定能力
- **仅视觉(VOnly)**：AI模型在无声条件下V-Acc仍高于随机 → 暴露了视觉偏见
- **Mono vs Stereo**：EchoPin立体声比单声道在Size3上A-Acc高16.2%（47.6% vs 31.4%）
- **耳蜗图 vs Mel**：EchoPin(耳蜗图)比EchoPin-S(Mel) Size3上高12.4%

### 类人偏差
EchoPin展现出类人的水平定位精度>垂直定位精度的不对称性——源于双耳立体声+HRTF的水平面空间线索更强。

## 亮点
1. **系统性模态偏见分析**：首次用6种受控条件定量揭示SSL模型的视觉偏见
2. **神经科学驱动设计**：HRTF+耳蜗图组合忠实模拟人类听觉外周，涌现出类人行为
3. **人-机对照实验**：心理物理实验提供了可靠的人类基线
4. **AudioCOCO数据集**：规避了现有数据集的快捷学习问题（大物体居中等偏见）

## 局限与展望
1. 小目标(Size1)上EchoPin与人类差距仍很大（4.5% vs 25.7%），说明模型对微弱空间线索的利用不足
2. 视觉冲突条件下EchoPin仍受误导——不如人类的听觉优先策略稳健
3. 仅使用静态图像+合成音频，未涉及真实世界视频中的动态声源

## 与相关工作的对比
- **vs IS3**：IS3用单声道+标准训练集，重度视觉偏见；EchoPin用立体声+HRTF+耳蜗图+空间平衡数据集
- **vs CAVP/AVSegformer**：这些方法也存在严重视觉偏见，多实例条件下与IS3接近
- **vs ImageBind/LanguageBind**：大规模预训练模型在SSL上也未展现优势

## 启发与关联
- 对VLM领域有启示意义：多模态模型是否也存在类似的模态偏见？
- HRTF+耳蜗图的听觉前端处理思路可迁移到其他音频-视觉任务（音视频分离、空间音频）
- AudioCOCO的受控数据集构建方法值得在其他多模态领域借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ 神经科学启发式SSL模型+系统性模态偏见分析
- 实验充分度: ⭐⭐⭐⭐⭐ 人类实验+多模型对比+6条件+消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 揭示了多模态模型的重要缺陷并提供解决思路

<!-- RELATED:START -->

## 相关论文

- [Object-aware Sound Source Localization via Audio-Visual Scene Understanding](../../CVPR2025/audio_speech/object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [Improving Sound Source Localization with Joint Slot Attention on Image and Audio](../../CVPR2025/audio_speech/improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [Generating Physically Sound Designs from Text and a Set of Physical Constraints](generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)

<!-- RELATED:END -->
