---
title: >-
  [论文解读] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding
description: >-
  [CVPR 2026][视频理解][视频时刻检索] 提出一个两阶段视频时刻检索框架：第一阶段用LLM引导字幕匹配并生成辅助短视频作为时序先验，第二阶段用多模态控制Mamba网络高效融合生成先验与长序列，在TVR数据集上超越SOTA（R@1/IoU=0.5达45.20%），同时降低计算开销。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "视频时刻检索"
  - "多模态查询增强"
  - "生成视频先验"
  - "Mamba"
  - "时序定位"
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
这篇论文要解决的核心问题是：当文本查询里有好几个连续动词时（如"走进房间并把咖啡递给Park"），纯文本和静态图像都讲不清这些动作的先后顺序，导致模型定位错时间段。它的破局思路是先"把查询拍成一段短视频"，让生成视频自带运动节奏当时序先验，再用一个能吃长序列的网络把这个先验和真实视频对齐。

整体流程分两阶段。第一阶段是"造先验"：用 LLM 把查询拆成按动词排列的子事件、从视频字幕里捞出相关对话线索，再把两者融成一句叙事提示喂给文本到视频扩散模型，生成一段约 6 秒的辅助短视频。第二阶段是"对齐与定位"：把文本 embedding、生成视频 embedding、目标视频 embedding 一起送进多模态控制的 Mamba 网络，输出上下文特征，由线性头预测起止时间戳，最后用 NMS 精炼候选区间。

### 关键设计

**1. LLM 引导的字幕匹配与查询处理：把抽象查询拆成有顺序的动作线索**

复杂查询的麻烦在于它把一连串动作压缩成一句话，"走进房间并递咖啡"读起来是一件事，实际上要拆成开门、走进、靠近、伸手递这几个有先后的子动作，模型才知道该往视频的哪一段找。这里先用 LLaMA-3.1 把查询按动词切成子事件，并主动补上查询里没明说但物理上必然发生的中间动作（如"开门"）——这是一种很便宜但有效的增强，等于让 LLM 替查询补全常识。光有查询还不够具体，于是再去视频字幕里找佐证：对每条字幕句子算它和各子查询的相关性分数 $r_j = \max_i \sigma(\text{LLM}(q_i, s_j))$，取高于阈值 $\theta$ 的 top-k 字幕组成精炼子集 $S'$。查询分解负责把动作排成序列，字幕匹配负责用对话线索消解查询本身的模糊，两者一起给下一步的视频生成攒够时序语境。

**2. 时序先验生成：用生成的短视频替代静态图像，把"运动"显式画出来**

前一步拆出来的子事件和字幕仍然是文字，而 VMR 最缺的恰恰是动态线索——ICQ 之类的方法用 DALL-E 生成静态图像来补语义，但一张图说不出"先进门再递咖啡"这个顺序，模型还是会定位错。这里改成生成视频：先用 LLM 把查询 $q$ 和匹配字幕 $S'$ 融成一句连贯的叙事提示 $p = q \oplus \text{LLM}(\{s\}_{s \in S'})$，再交给 CogVideoX 文本到视频模型生成一段约 6 秒的短视频 $v_g \sim \mathcal{D}(p, \theta)$。这段生成视频远短于目标视频（$L_g \ll L_o$），相当于给目标事件先放了个动态"预览"，时序动态天然就编码在帧与帧之间。之所以有效，是因为它从根上补齐了静态增强缺的那一维——运动；实验里 CogVideoX 比 Stable Video Diffusion 效果更好，原因就是它的运动保真度更高，生成视频越像真的，给出的时序先验就越准。

**3. 多模态控制 Mamba 网络：用生成先验当"门"，引导长视频的状态传播**

有了生成视频先验，还得把它和动辄上千帧的目标视频高效融合——Transformer 的二次复杂度在这种长度上直接内存爆炸（序列超 700 就 OOM），所以这里换成线性复杂度的 Mamba。输入序列由目标视频 embedding $e_o$ 加上 GCN 提取的关系嵌入 $r_o$ 构成，$x = e_o + r_o$，主干是双向 SSM，状态按 $h_t = Ah_{t-1} + Bx_t,\ y_t = Ch_t$ 传播。关键创新是把生成先验做成一道**视频引导门控**，让它来决定每个时刻状态更新放多少进去：

$$g_t = \sigma(W_g[e_q; \text{pooled}(e_g)]_t), \qquad h_t' = g_t \odot (Ah_{t-1} + Bx_t)$$

其中 $e_q$ 是文本 embedding，$\text{pooled}(e_g)$ 是生成视频的均值池化 embedding。门控信号同时看文本和生成视频，于是当目标视频某段和运动先验对得上时门就开大、状态多更新，对不上时门关小、把无关噪声过滤掉。比起把三路特征简单拼起来一起算，这种门控把先验里的运动信息直接注入状态传播过程，既省算力又更聚焦。

### 一个完整示例
以查询"Adams 走进房间并把咖啡递给 Park"为例走一遍：第一步 LLaMA-3.1 把它拆成 "开门后走进房间"、"端着咖啡靠近 Park"、"伸手递咖啡" 三个子事件，并补出查询没写的"开门"；第二步去字幕里捞线索，假设某句字幕"Adams: Here's your coffee, Park"对"伸手递咖啡"子查询打出高相关分，被选进 $S'$；第三步 LLM 把查询和这条字幕融成叙事提示，CogVideoX 据此生成一段 6 秒短视频，里面真的演出了"进门→走近→递杯"的连贯动作；第四步这段生成视频经池化后变成门控信号 $\text{pooled}(e_g)$，在 Mamba 扫过几百帧目标视频时，把真正出现"递咖啡"动作的那几帧的门打开、其余压低；最后线性头在被放大的区间里预测起止时间戳，NMS 去掉重叠候选，输出最终时间段。整条链路的关键就是：文字层面模糊的时序关系，被"造一段视频"显式化，再借门控传回到对真实视频的定位上。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)
- [\[CVPR 2026\] Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking](ninja_codes_neurally_generated_fiducial_markers_for_stealthy_6-dof_tracking.md)
- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[CVPR 2026\] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)

</div>

<!-- RELATED:END -->
