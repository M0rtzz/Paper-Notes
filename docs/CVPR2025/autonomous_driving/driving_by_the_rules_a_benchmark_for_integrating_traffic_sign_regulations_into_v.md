---
title: >-
  [论文解读] Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map
description: >-
  [CVPR 2025][自动驾驶][交通标志规则] 本文首次定义了将交通标志规则集成到在线向量化高精地图的任务，构建了包含10000+视频片段和18000+车道级规则的MapDR数据集，并提出模块化（VLE-MEE）和端到端（RuleVLM）两种基线方案，其中RuleVLM在整体F1指标上达到64.2%。
tags:
  - CVPR 2025
  - 自动驾驶
  - 交通标志规则
  - 高精地图
  - 数据集基准
  - 视觉语言模型
  - 规则-车道关联推理
---

# Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map

**会议**: CVPR 2025  
**arXiv**: [2410.23780](https://arxiv.org/abs/2410.23780)  
**代码**: [MIV-XJTU/MapDR](https://github.com/MIV-XJTU/MapDR)  
**领域**: 自动驾驶  
**关键词**: 交通标志规则, 高精地图, 数据集基准, 视觉语言模型, 规则-车道关联推理

## 一句话总结
本文首次定义了将交通标志规则集成到在线向量化高精地图的任务，构建了包含10000+视频片段和18000+车道级规则的MapDR数据集，并提出模块化（VLE-MEE）和端到端（RuleVLM）两种基线方案，其中RuleVLM在整体F1指标上达到64.2%。

## 研究背景与动机

高精地图（HD Map）通常包含三个层次：几何层（车道线、分界线等向量信息）、连接层（车道拓扑关系）和交通规则层（限速、公交专用道等车道级规则）。现有的在线高精地图构建方法（如MapTR、TopoMLP等）主要关注几何层和连接层的构建，完全忽略了交通规则层。这导致自动驾驶系统仍然需要依赖离线地图来获取交通法规信息，与在线化趋势背道而驰。

虽然OpenLane-V2尝试了交通标志与车道的关联，但仅考虑方向标志，且标注仅为标志类别，缺乏符合HD地图标准的结构化规则描述。交通标志是道路上的"视觉语言"，从标志中提取结构化规则并与具体车道关联，是一个涉及视觉理解、语义推理和空间推理的复杂多模态任务。

本文的核心贡献是定义了这个新任务，构建了第一个专用数据集MapDR，并提供了模块化和端到端两类基线方案。

## 方法详解

### 整体框架
任务被形式化为：给定图像序列X和车道中心线L，输出二部图G=(R∪L, E)，其中R是从交通标志中提取的结构化规则集合，E是规则与车道的对应关系矩阵。任务可拆分为两个子任务：(1) 规则提取，(2) 规则-车道对应推理。本文提出模块化方案（VLE+MEE串联）和端到端方案（RuleVLM）。

### 关键设计

1. **MapDR数据集**:

    - 规模：10000+视频片段，400000+前视图像，18000+车道级规则
    - 采集范围：北京、上海、广州三大城市的复杂交通场景
    - 每个视频片段以交通标志为中心，覆盖100m×100m区域
    - 每个片段包含30-60帧，每2米采集一帧
    - 分辨率：1920×1240，提供相机内参和位姿
    - 规则标注格式：每条规则包含8个预定义属性的{key:value}对，参照HD地图规范
    - 提供向量化局部地图（分界线、边界线、中心线、人行横道等3D点列表）
    - 数据呈现自然长尾分布：公交车道和方向车道多，潮汐车道少
    - 隐私保护：所有图像的车牌和人脸已做脱敏处理

2. **模块化方案：VLE + MEE**:

    - **VLE（Vision-Language Encoder）**：用于规则提取
        - 视觉编码器：ViT-b16；文本编码器：L=6层Transformer
        - 两阶段流程：先聚类OCR结果中的符号和文字（通过[STC] token的余弦相似度），再提取结构化规则
        - 引入[CLS] token表示整条规则，[STC] token表示句子级表示
        - 使用实例间和实例内注意力机制增强交互
        - 利用文字布局的位置编码捕获空间语义
    - **MEE（Map Element Encoder）**：用于规则-车道对应推理
        - M=2层Transformer编码器 + N=2层交叉注意力融合层
        - 将向量点序列类比为句子中的单词序列
        - [VEC] token表示每条向量的固定长度特征
        - 引入类型嵌入（区分分界线/中心线等）、实例嵌入（区分不同向量实例）
        - 使用实例间/内注意力机制捕获向量间的关系
        - 对每个[VEC] token用二分类头判断该车道是否对应当前规则

3. **端到端方案：RuleVLM**:

    - 基于Qwen-VL (9.6B)，用LoRA微调
    - 三种向量编码方式的比较：
        - TextPrompt：将中心线坐标编码为文本输入LLM（效果最差，序列过长）
        - VisualPrompt：将中心线可视化在PV图像上作为视觉提示（规则提取好但整体一般）
        - RuleVLM：用MEE独立编码向量化地图信息，通过adapter与LLM对齐（效果最好）
    - 输出序列化的JSON格式规则，通过JSON decoder解析还原结构化数据
    - 训练时随机打乱中心线顺序防止过拟合

### 损失函数 / 训练策略
- VLE规则提取：对比损失（聚类阶段）+ 多头分类损失（理解阶段）
- MEE对应推理：二分类BCE损失
- RuleVLM：标准的next-token prediction + LoRA微调
- VLE训练50 epoch，MEE训练120 epoch
- VLE用DeiT和BERT预训练权重初始化，MEE从零训练
- 输入图像resize至256×256，特征维度768，12注意力头

## 实验关键数据

### 主实验

| 方法 | 类型 | P_RE | R_RE | P_CR | R_CR | F1 (Overall) |
|------|------|------|------|------|------|------|
| Heuristic | 模块化 | 18.01 | 11.51 | 33.05 | 17.99 | 0.035 |
| ALBEF-BERT | 模块化 | 75.78 | 57.56 | 4.14 | 17.25 | 0.003 |
| VLE-MEE | 模块化 | 76.67 | 74.54 | 78.05 | 82.16 | 0.653 |
| Qwen-VL (TextPrompt) | 端到端 | 42.21 | 41.09 | - | - | 0.083 |
| Qwen-VL (VisualPrompt) | 端到端 | 89.29 | 89.50 | - | - | 0.392 |
| RuleVLM | 端到端 | 89.28 | 89.44 | - | - | 0.642 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| VLE无注意力机制 | R_RE: 57.56% | 注意力机制是关键，加入后recall提升至71.75% |
| VLE+注意力+布局 | R_RE: 74.54% | 文字布局带来边际提升 |
| MEE无注意力 | P_CR: 4.14% | 基本不可用 |
| MEE+注意力 | P_CR: 68.91% | 注意力机制是决定性因素 |
| MEE+注意力+类型嵌入 | P_CR: 78.05% | 向量类型信息显著提升对应推理 |

### 关键发现
- 规则提取和对应推理是两个差异很大的子任务：ALBEF-BERT在规则提取上表现不错但在对应推理上几乎失败
- MEE中的注意力机制是对应推理的决定性因素，没有它性能接近随机
- 类型嵌入对MEE很重要，说明向量类型（分界线vs中心线）包含了重要的语义信息
- 端到端方案中，文本坐标输入效果最差（LLM难以进行文本空间推理），视觉提示次之，MEE向量编码最优
- RuleVLM（F1=0.642）接近模块化VLE-MEE（F1=0.653），说明端到端方案潜力巨大

## 亮点与洞察
- 填补了HD地图研究中交通规则层的空白，任务定义清晰、评价指标完备
- MapDR数据集规模大、标注精细、覆盖多样化场景，是该领域的重要基础设施
- 将规则以{key:value}结构化格式定义，参照工业HD地图标准，实用性强
- MEE将向量编码类比为语言模型的token化，设计优雅
- 模块化和端到端方案的对比实验提供了有价值的启示：两种路线各有优劣
- 数据集的长尾分布反映了真实世界，具有挑战性

## 局限与展望
- 数据集仅覆盖中国三个城市的交通标志，缺乏国际化数据（不同国家的标志差异很大）
- 当前方案假设OCR结果已知（规则提取）或向量化地图已知（对应推理），在全自动流水线中还需整合OCR和在线建图
- 最终F1仅64%左右，说明任务仍有很大提升空间
- 夜间、恶劣天气等条件下的性能未评估
- RuleVLM基于9.6B的Qwen-VL，推理效率可能不满足实时要求
- 潮汐车道等稀有规则的数据量不足，可能导致长尾类别性能不佳
- 未与自动驾驶规划系统集成验证，交通规则层对规划性能的实际影响不清楚

## 相关工作与启发
- 与OpenLane-V2的对比：OpenLane-V2仅处理方向标志的单标签分类，而MapDR支持多属性结构化规则描述
- 与MAPLM等LLM驾驶基准的关系：MAPLM侧重端到端规划，MapDR侧重精确的规则提取和推理
- VLE的设计受ALBEF等视觉语言模型启发，但针对交通标志的多文本多规则特性做了特殊适配
- MEE的向量编码思路可以应用于其他需要处理向量化地理信息的任务
- RuleVLM展示了将结构化几何输入（向量）与LLM结合的有效方式，对多模态LLM的设计有启发

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义了交通规则集成到在线HD地图的任务，数据集和基线均是全新贡献
- 实验充分度: ⭐⭐⭐⭐ 模块化和端到端方案的对比全面，但缺少跨域评估和实际应用验证
- 写作质量: ⭐⭐⭐⭐ 任务定义和数据集描述详细清晰，方法部分结构稍显复杂
- 价值: ⭐⭐⭐⭐⭐ 填补了关键研究空白，数据集将推动交通规则理解和HD地图完整性的研究

<!-- RELATED:START -->

## 相关论文

- [MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)

<!-- RELATED:END -->
