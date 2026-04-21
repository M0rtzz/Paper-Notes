---
title: >-
  [论文解读] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding
description: >-
  [CVPR 2026][视频理解][视频时刻检索] 提出一个两阶段视频时刻检索框架：第一阶段用LLM引导字幕匹配并生成辅助短视频作为时序先验，第二阶段用多模态控制Mamba网络高效融合生成先验与长序列，在TVR数据集上超越SOTA（R@1/IoU=0.5达45.20%），同时降低计算开销。
tags:
  - CVPR 2026
  - 视频理解
  - 视频时刻检索
  - 多模态查询增强
  - 生成视频先验
  - Mamba
  - 时序定位
---

# Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding

**会议**: CVPR 2026  
**arXiv**: [2603.22121](https://arxiv.org/abs/2603.22121)  
**代码**: [https://github.com/YunzhuoSun/Manba-VMR](https://github.com/YunzhuoSun/Manba-VMR)  
**领域**: 视频理解 / 多模态VLM  
**关键词**: 视频时刻检索, 多模态查询增强, 生成视频先验, Mamba, 时序定位

## 一句话总结
提出一个两阶段视频时刻检索框架：第一阶段用LLM引导字幕匹配并生成辅助短视频作为时序先验，第二阶段用多模态控制Mamba网络高效融合生成先验与长序列，在TVR数据集上超越SOTA（R@1/IoU=0.5达45.20%），同时降低计算开销。

## 研究背景与动机

1. **领域现状**：视频时刻检索（VMR）旨在从未裁剪视频中定位与文本查询语义匹配的时间段。现有方法主要依赖自然语言查询（NLQ）或静态图像增强（如ICQ用DALL-E生成图像），并使用Transformer架构进行跨模态融合。

2. **现有痛点**：纯文本查询在处理含多动词的复杂查询时容易产生时序歧义。例如"Adams走进房间并把咖啡递给Park"需要理解"走"后接"递"的时序关系，但纯文本描述缺乏动态线索。静态图像增强虽提升了语义表达力，但无法传达动态运动信息——生成的图像忽略了动作的时序流程（如进入房间→靠近→伸手递咖啡的顺序），导致定位错误。

3. **核心矛盾**：多动词查询需要显式的时序动态线索（motion cue），但文本和静态图像都无法提供。同时，引入生成视频会拉长输入序列，使Transformer的二次复杂度成为瓶颈。

4. **本文目标**：(a) 如何为查询生成富含时序动态的辅助信息？(b) 如何高效融合生成先验与长视频序列？

5. **切入角度**：利用文本到视频扩散模型（CogVideoX）生成短视频作为时序先验，捕获隐含的运动信息；用Mamba（SSM）替代Transformer实现线性时间复杂度的长序列建模。

6. **核心 idea**：生成动态视频而非静态图像作为查询增强的时序先验，并用多模态控制的Mamba网络高效融合文本、生成视频和目标视频，实现精确的时序定位。

## 方法详解

### 整体框架
框架分为两阶段。**第一阶段**：LLM引导的字幕匹配 → 查询分解为动词子事件 → 融合查询和字幕生成短视频。**第二阶段**：文本embedding、生成视频embedding、目标视频embedding输入多模态控制Mamba网络 → 输出上下文特征 → 线性头预测起止时间戳 → NMS精炼。

### 关键设计

1. **LLM引导的字幕匹配与查询处理**:

    - 功能：分解复杂查询为动词中心的子事件，并从视频字幕中匹配相关文本线索
    - 核心思路：使用LLaMA-3.1将查询按动词分解为子事件，例如"走进房间并递咖啡"分解为"开门后走进房间"、"端着咖啡靠近Park"、"伸手递咖啡"。同时补充隐含的中间动作（如"开门"）。然后对每个字幕句子评估其与各子查询的相关性分数 $r_j = \max_i \sigma(\text{LLM}(q_i, s_j))$，选取高于阈值 $\theta$ 的top-k字幕构成精炼子集 $S'$。
    - 设计动机：Query分解将高层抽象描述拆解为细粒度的动作序列，字幕匹配引入对话线索弥补查询的模糊性，两者配合为后续视频生成提供丰富的时序语境。

2. **时序先验生成（Temporal Prior Generation）**:

    - 功能：从融合查询和字幕生成短视频，捕获隐含的动态运动信息
    - 核心思路：用LLM将查询 $q$ 和匹配字幕 $S'$ 融合为连贯的叙事提示 $p = q \oplus \text{LLM}(\{s\}_{s \in S'})$，输入CogVideoX文本到视频模型生成6秒辅助短视频 $v_g \sim \mathcal{D}(p, \theta)$。生成视频的长度远小于目标视频（$L_g \ll L_o$），提供了目标事件的动态"预览"。
    - 设计动机：静态图像（如ICQ用DALL-E生成）可以补充语义但无法表达运动序列，生成视频天然包含时序动态，弥补了静态增强在时序定位上的根本缺陷。实验证实CogVideoX优于Stable Video Diffusion，因为其运动保真度更高。

3. **多模态控制Mamba网络**:

    - 功能：高效融合文本、生成视频先验和目标视频的多模态信息，输出上下文特征用于时间戳预测
    - 核心思路：目标视频embedding $e_o$ 加上GCN提取的关系嵌入 $r_o$ 构成输入序列 $x = e_o + r_o$。核心是双向SSM，状态转移 $h_t = Ah_{t-1} + Bx_t, y_t = Ch_t$。关键创新是**视频引导门控**：$g_t = \sigma(W_g[e_q; \text{pooled}(e_g)]_t)$ 动态调制状态转移为 $h_t' = g_t \odot (Ah_{t-1} + Bx_t)$，其中 $e_q$ 是文本embedding，$\text{pooled}(e_g)$ 是生成视频的均值池化embedding。门控机制使Mamba聚焦于与运动先验对齐的视频段落，同时过滤无关噪声。
    - 设计动机：Transformer的二次复杂度在长序列上内存爆炸（序列超700时OOM），而Mamba的线性复杂度适合处理未裁剪的长视频。视频引导门控将生成先验的运动信息注入状态传播，比简单拼接更高效。

### 损失函数 / 训练策略
总损失由三部分组成：$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{bound}} + \lambda_2 \mathcal{L}_{\text{rel}} + \lambda_3 \mathcal{L}_{\text{cont}}$。边界损失 $\mathcal{L}_{\text{bound}}$ 是起止位置的BCE；相关性损失 $\mathcal{L}_{\text{rel}}$ 是clip级别的BCE评分；对比损失 $\mathcal{L}_{\text{cont}}$ 用InfoNCE最大化生成视频与正样本clip的相似度。权重 $\lambda_1=1, \lambda_2=0.5, \lambda_3=0.1$。使用AdamW优化器，4块RTX 4090训练20 epochs。

## 实验关键数据

### 主实验
TVR数据集对比：

| 方法 | R@1/IoU=0.5 | R@10/IoU=0.5 | R@1/IoU=0.7 | R@10/IoU=0.7 |
|------|-------------|--------------|-------------|--------------|
| HERO | 33.86 | 58.69 | 10.15 | 34.00 |
| SgLFT | 42.51 | 72.41 | 21.03 | 54.62 |
| ICQ | 44.13 | 75.27 | 24.08 | 59.23 |
| **Ours** | **45.20** | **76.09** | **25.10** | **60.87** |

ActivityNet也有提升：R@100/IoU=0.5从ICQ的81.20→83.59。

### 消融实验

| 配置 | R1/0.5 | R1/0.7 | 说明 |
|------|--------|--------|------|
| Full model | 45.20 | 25.10 | 完整方法 |
| w/o LLM模块 | 40.15 | 21.45 | 去掉字幕匹配+查询分解，掉5.05 |
| w/o 视频先验 | 38.76 | 20.08 | 改用静态图像，掉6.44 |
| w/o 视频门控 | 41.23 | 22.34 | 标准SSM无门控，掉3.97 |
| 用Transformer替代Mamba | 37.89 | 19.56 | 掉7.31，且长序列OOM |
| w/o 对比损失 | 41.08 | 21.56 | 多模态融合变差 |

### 关键发现
- 视频先验生成贡献最大（去掉后掉6.44），证明动态运动线索对时序定位至关重要，远超静态图像增强
- 多动词查询分析：4+动词查询上本文35.9% vs ICQ 16.8% vs SgLFT 16.3%，改善幅度高达约19%，说明生成视频对复杂时序关系建模特别有效
- Mamba内存线性增长 vs Transformer在序列长700时OOM，验证了选择Mamba的必要性
- CogVideoX > Stable Video Diffusion > DALL-E静态 > 无先验，更好的视频生成模型带来更精确的时序定位

## 亮点与洞察
- **用生成视频替代生成图像做查询增强**：这是一个范式转变——从静态语义补充升级到动态时序补充。这个思路可迁移到其他需要时序理解的视频任务（如视频问答、动作预测）
- **LLM做查询分解补充隐含动作**：利用LLM的推理能力补全查询中未明说的中间步骤（如"开门"），这是一种cheap but effective的数据增强
- **视频引导门控融入Mamba**：通过门控让生成先验引导SSM的状态传播，比简单拼接更精准更高效

## 局限与展望
- 视频生成质量是瓶颈——如果CogVideoX生成了不相关的运动，反而可能引入噪声
- 生成视频需预计算（6秒/clip），离线模式可接受但不支持实时应用
- 仅在TVR和ActivityNet上评估，缺少对其他VMR数据集（如Charades-STA）的完整对比
- 字幕依赖问题：ActivityNet无字幕时退化为纯查询生成视频，效果不如有字幕的TVR，说明字幕融合对性能贡献重要但并非所有场景都有字幕

## 相关工作与启发
- **vs ICQ**: ICQ用DALL-E生成静态图像增强查询，本文用CogVideoX生成动态视频。在多动词查询场景差距尤其大（35.9% vs 16.8%）
- **vs SgLFT**: SgLFT用语义引导Transformer融合字幕，但缺乏运动先验且受限于Transformer的二次复杂度
- **vs Motion Mamba**: 本文扩展了Motion Mamba的文本控制选择机制，增加了视频引导门控实现多模态控制

## 评分
- 新颖性: ⭐⭐⭐⭐ 用生成视频做时序先验是VMR领域的新思路，LLM字幕匹配和Mamba融合也有创新
- 实验充分度: ⭐⭐⭐⭐ 消融实验覆盖全面，多动词分析有说服力，但主数据集偏少
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对VMR社区有启发，生成视频先验的思路有潜力扩展到更多视频理解任务

<!-- RELATED:START -->

## 相关论文

- [SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](../../CVPR2025/video_understanding/decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)

<!-- RELATED:END -->
