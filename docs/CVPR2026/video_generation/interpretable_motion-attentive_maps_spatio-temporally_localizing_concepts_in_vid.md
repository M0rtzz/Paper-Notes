---
title: >-
  [论文解读] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers
description: >-
  [CVPR 2026][视频生成][视频扩散模型] 提出IMAP(可解释运动注意力图)，通过GramCol空间定位和运动头选择时序定位两个无训练模块，从Video DiT中提取运动概念的时空显著性图，在运动定位和零样本视频语义分割上超越现有方法。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "视频扩散模型"
  - "可解释性"
  - "运动定位"
  - "注意力分析"
  - "显著性图"
---

# I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers

**会议**: CVPR 2026  
**arXiv**: [2603.02919](https://arxiv.org/abs/2603.02919)  
**代码**: [https://github.com/youngjun-jun/IMAP](https://github.com/youngjun-jun/IMAP)  
**领域**: 视频生成  
**关键词**: 视频扩散模型, 可解释性, 运动定位, 注意力分析, 显著性图

## 一句话总结
提出IMAP(可解释运动注意力图)，通过GramCol空间定位和运动头选择时序定位两个无训练模块，从Video DiT中提取运动概念的时空显著性图，在运动定位和零样本视频语义分割上超越现有方法。

## 研究背景与动机
1. **领域现状**: Video Diffusion Transformers(如CogVideoX/HunyuanVideo)已能生成高质量视频，但对其内部机制的理解仍不充分。现有可解释性工作主要集中在图像DiT上。
2. **现有痛点**: 已有方法ConceptAttention仅提供空间分离，不处理运动/时序；DiTFlow/DiffTrack关注帧间视觉token的动态对应，但不分析文本如何转化为运动。核心问题未解答：Video DiT真的理解并创造运动了吗？
3. **核心矛盾**: 视频的核心区别于图像的是时序运动信息，但现有显著性图方法只做空间定位，无法回答"何时、哪个物体在运动"这一关键问题。
4. **本文目标**: 为视频DiT中的运动概念构建时空定位的可解释显著性图。
5. **切入角度**: 分析Video DiT的多头注意力发现：QK匹配有强空间定位能力，帧嵌入分离度与运动可定位性相关。不同注意力头有不同角色——某些头专注时序运动特征。
6. **核心 idea**: 用GramCol做空间定位（文本代理token+Gram矩阵），用帧分离度评分选择运动头做时序定位。

## 方法详解

### 整体框架
Pipeline在Video DiT的MM-Attn模块上运行。给定概念词：(1)通过QK匹配找到最相关的视觉token作为文本代理；(2)用GramCol计算基于Gram矩阵的空间显著性图；(3)对运动概念，额外进行运动头选择——用Calinski-Harabasz指数衡量帧间分离度，只保留top-k运动头的特征来计算IMAP。全程无需梯度计算、无需参数更新。

### 关键设计
1. **GramCol空间定位**:
    - 功能：为任意文本概念生成每帧的空间显著性图。
    - 核心思路：对每帧$f_i$，通过QK匹配$s_{f_i}^c = \arg\max_p \text{row}_p(q_{f_i})k_c^\top$找到与概念c最匹配的视觉token作为文本代理。GramCol为视觉Gram矩阵$G = h_x h_x^\top \in \mathbb{R}^{P \times P}$的第$s_{f_i}^c$列，即所有视觉token与代理token的相似度向量。最终对选定时间步、层、头取平均。
    - 设计动机：相比ConceptAttention(跨模态特征相乘)，GramCol在同一模态空间内计算相似度，天然保证"正向高亮"的可解释性——与代理token相似的区域自动获得正的大值。而且不需要对概念列表做softmax，单个概念就能工作。

2. **运动头选择 (Motion Head Selection)**:
    - 功能：识别Video DiT中专门处理运动的注意力头，实现时序定位。
    - 核心思路：对每个注意力头，将视觉token按帧分为$F$个聚类，计算Calinski-Harabasz指数(CHI)衡量帧间特征分离度。CHI越高说明该头的帧间差异越大，即包含更多时序运动信息。每层选择top-5高CHI头，只用这些头的特征计算GramCol，得到IMAP。Pearson相关系数0.60验证了CHI与运动定位得分的正相关。
    - 设计动机：运动是帧间变化，包含强运动信息的头的帧间特征自然应该差异大。与直接聚合所有头相比，选择运动头消除了空间头的干扰，使得运动定位更清晰。

3. **层与时间步选择**:
    - 功能：缩小分析范围，只在信息丰富的层和时间步上提取特征。
    - 核心思路：排除早期时间步（接近噪声，语义不可解，且容易出现记忆相关伪影如水印）。层选择基于注意力矩阵的第二大特征值$\lambda_2$——在DTMC框架下，$\lambda_2$越大表示转移矩阵越informative。CogVideoX选$\lambda_2 > 0.7$的层，HunyuanVideo选$> 0.75$。
    - 设计动机：$L \times T$个层×时间步的空间太大，全聚合会稀释信号。基于$\lambda_2$的自动选择避免了手动调参。

### 损失函数 / 训练策略
IMAP是完全无训练、无梯度的方法。不需要任何额外训练或参数更新。对于真实视频，通过加噪-去噪过程提取特征。GramCol只需Gram矩阵的一列——计算复杂度$O(Pd)$的矩阵乘法加$O(P)$的索引操作。CHI的计算也是轻量级操作（帧间/帧内方差比）。整个pipeline的额外开销相对于Video DiT的推理本身可以忽略不计。实验中处理49帧视频的全部分析可在数秒内完成。实现细节：CogVideoX用$\lambda_2 > 0.7$的层，HunyuanVideo用$\lambda_2 > 0.75$的层。运动头选择固定取top-5。仅使用双流MM-DiT块（HunyuanVideo的单流块不使用）。

## 实验关键数据

### 主实验 (运动定位)

| 方法 | Backbone | SL | TL | PR | SS | OBJ | Avg |
|------|----------|-----|-----|-----|-----|-----|------|
| ViCLIP | ViT-H | 0.33 | 0.17 | 0.35 | 0.29 | 0.28 | 0.28 |
| DAAM | VideoCrafter2 | 0.36 | 0.17 | 0.38 | 0.32 | 0.35 | 0.32 |
| ConceptAttn | CogVideoX-5B | 0.50 | 0.32 | 0.51 | 0.47 | 0.47 | 0.45 |
| **IMAP** | CogVideoX-5B | **0.58** | **0.65** | **0.64** | **0.52** | **0.59** | **0.60** |
| ConceptAttn | HunyuanVideo | 0.42 | 0.26 | 0.44 | 0.35 | 0.34 | 0.36 |
| **IMAP** | HunyuanVideo | **0.60** | **0.41** | **0.62** | **0.50** | **0.62** | **0.55** |

### 消融实验

| 配置 | Avg Score | 说明 |
|------|-----------|------|
| Cross-Attention Map | 0.34 | 基础注意力图 |
| GramCol (全部头) | ~0.45 | 空间定位有效但时序不精确 |
| GramCol + 层选择 | ~0.50 | 排除低info层后提升 |
| IMAP (GramCol + 运动头) | 0.54-0.60 | 运动头选择带来时序定位突破 |

### 关键发现
- 时序定位(TL)是IMAP最大优势：在CogVideoX-2B上TL从0.56(Cross-Attn)提升到0.62，在HunyuanVideo上从0.26提升到0.41。
- GramCol比ConceptAttention更为稳定：ConceptAttention在不同头之间行为异质导致不稳定，GramCol使用同模态相似度避免了这一问题。
- 运动头选择的有效性通过CHI-MLS的正相关(r=0.60)得到验证,随机选头性能显著下降。
- IMAP在零样本视频语义分割任务上同样有效。

## 亮点与洞察
- **文本代理token的巧妙设计**：不直接用跨模态的文本token计算相似度，而是用QK匹配找到"最能代表文本概念"的视觉token，将跨模态问题转化为同模态问题。这个思路可以推广到任何需要跨模态定位的场景。
- **运动=帧间差异的简单假设**：用聚类分离度衡量运动信息含量，计算开销极低(CHI是轻量操作)，却非常有效。证明了有时候简单的统计指标比复杂的学习方法更适合做特征选择。
- **对Video DiT内部机制的洞察**：发现不同注意力头确实分工明确(空间vs运动)，$\lambda_2$大的层更语义化——这为未来Video DiT的设计和优化提供了指导。

## 局限与展望
- **评估依赖LLM评分**：使用OpenAI o3-pro进行MLS评估，虽然使用了详细的rubric，但LLM评估的可复现性和一致性仍有顾虑。缺少人类评估的对比验证。
- 对非常**微妙的运动**（如微表情变化、缓慢渐变）的定位能力未验证——CHI分离度可能无法捕捉这类细粒度帧间差异。
- 目前只在CogVideoX (2B/5B) 和HunyuanVideo上验证，对其他架构（单流DiT、跨注意力架构）的适用性需要更多实验。
- 运动头选择的top-k=5是全局固定的，不同视频/运动类型可能需要不同数量的头。自适应k值选择是自然的改进方向。
- $\lambda_2$层选择阈值（CogVideoX 0.7, HunyuanVideo 0.75）也是手动设定的，缺乏自动化的选择策略。
- IMAP是分析工具而非生成控制工具，如何将运动头发现反向用于运动生成/编辑控制是值得探索的方向。
- 目前的benchmark（504视频，150种运动类型）规模有限，大规模评估有待构建。
- 对多个物体同时运动的场景（如两人互动），各物体的运动分离能力需要进一步验证。

## 相关工作与启发
- **vs ConceptAttention**: ConceptAttention只做空间分离，且跨模态$h_x h_c^\top$的相似度在不同头之间行为异质；GramCol用同模态Gram矩阵避免了这些问题，并扩展到时序定位。ConceptAttention的softmax操作导致多概念竞争，GramCol不需要。
- **vs DAAM**: DAAM使用U-Net的交叉注意力图，适用于旧架构，不能直接用于联合注意力的DiT架构；IMAP专为DiT架构设计，利用MM-Attn的QK匹配和头级分析。
- **vs DiTFlow/DiffTrack**: 它们关注帧间视觉token对应（光流/跟踪），而IMAP关注——“特定运动文本概念对应哪些视觉区域”。角度互补，潜在可组合使用。
- **与注意力头剪枝研究的关联**: 运动头vs空间头的发现与Video DiT的推理加速研究（稀疏化不同头）相互印证，提示我们可以更智能地剪枝/稀疏化而不丢失运动信息。
- **与TokenRank的关联**: 本文借用了TokenRank的DTMC视角和$\lambda_2$的重要性指标，但将其从per-state加权扩展到per-layer选择，是一个新的应用。
- **对视频编辑/控制的启发**: IMAP发现的运动头可以反向用于运动编辑——通过操控运动头的特征来控制生成视频中的运动，而不影响空间外观。
- **对视频理解的洞察**: 本文首次展示了Video DiT内部确实存在专门处理运动的注意力头，这对理解视频生成模型的内部机制有重要意义。
- **零样本视频语义分割的潜力**: GramCol在零样本视频语义分割上也表现优异，说明Video DiT的内部表示对感知任务同样有价值，可以作为轻量级视频理解工具。
- **504视频/150种运动类型的benchmark**: 本文构建的运动定位评估基准本身也是贡献，填补了该方向的评估空白。使用Qwen3-VL标注并过滤无运动视频，保证了评估质量。

## 评分
- 新颖性: ⭐⭐⭐⭐ GramCol+运动头选择的设计新颖且优雅，首次系统研究Video DiT中的运动可解释性
- 实验充分度: ⭐⭐⭐⭐ 三个Video DiT模型验证，含消融和零样本分割，benchmark构建规范
- 写作质量: ⭐⭐⭐⭐⭐ 分析层次清晰，从时间步→层→头逐步缩小范围，每步都有理论依据和实验验证
- 价值: ⭐⭐⭐⭐ 为Video DiT可解释性研究开辟了运动维度，GramCol和IMAP都有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](disca_accelerating_video_diffusion_transformers_wi.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

</div>

<!-- RELATED:END -->
