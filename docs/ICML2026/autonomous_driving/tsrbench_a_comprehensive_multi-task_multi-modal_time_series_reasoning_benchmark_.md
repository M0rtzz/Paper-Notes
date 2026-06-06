---
title: >-
  [论文解读] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models
description: >-
  [ICML 2026][自动驾驶][时间序列推理] TSRBench 构造了一个覆盖 14 个领域、4 大维度（感知/推理/预测/决策）、15 个任务、4125 道题、同时支持文本/可视化/文本+图/嵌入四种模态输入的时间序列推理基准，系统评测 30+ 主流 LLM、VLM 与 TSLLM…
tags:
  - "ICML 2026"
  - "自动驾驶"
  - "时间序列推理"
  - "多模态评测"
  - "LLM/VLM/TSLLM"
  - "4 维度 15 任务"
  - "scaling law"
---

# TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models

**会议**: ICML 2026  
**arXiv**: [2601.18744](https://arxiv.org/abs/2601.18744)  
**代码**: https://tsrbench.github.io/ (有)  
**领域**: 评测基准 / 时间序列推理 / 多模态VLM / 通用大模型  
**关键词**: 时间序列推理, 多模态评测, LLM/VLM/TSLLM, 4 维度 15 任务, scaling law

## 一句话总结
TSRBench 构造了一个覆盖 14 个领域、4 大维度（感知/推理/预测/决策）、15 个任务、4125 道题、同时支持文本/可视化/文本+图/嵌入四种模态输入的时间序列推理基准，系统评测 30+ 主流 LLM、VLM 与 TSLLM，揭示出"scaling 在感知/推理上仍成立但在预测上失效"以及"文本与可视化模态高度互补但当前模型几乎无法融合"等关键结论。

## 研究背景与动机

**领域现状**：时间序列在金融、医疗、工业、交通等高风险领域无处不在，推理时间序列被视为通用模型走向真实问题求解的核心能力。已有评测基本沿用传统时间序列分析范式（预测、分类、异常检测、插补），把序列当成孤立的数字串；近期出现的 TimeMMD / CiK / TimeSeriesExam / MTBench / TSR-SUITE 等开始引入上下文与 LLM/VLM，但要么集中在预测一个维度，要么只在一两个垂直领域里测一两类任务。

**现有痛点**：（1）任务维度极不平衡，多数 benchmark 只覆盖 1-2 个能力维度（如 TimeMMD 只测预测、TimeSeriesExam 只测感知）；（2）领域多样性有限，模型可以靠领域专有先验刷分；（3）模态单一，几乎没有 benchmark 同时支持"文本数字串 / 折线图 / 图文交错 / 时间序列 embedding"这四种输入形式；（4）缺少对"语义理解 vs 数值预测"是否解耦、"文本 vs 视觉"是否互补这类 generalist 级别的横向问题的定量回答。

**核心矛盾**：通用 LLM/VLM 的能力评测正在快速扩张到数学、科学、具身等场景，但时间序列这一与现实物理世界最紧密耦合的模态，却长期被排除在"通用推理"评测框架之外；现有时序 benchmark 又只看任务指标而不看"推理能力维度"。

**本文目标**：构造一个能同时压力测试感知 / 推理 / 预测 / 决策四大能力维度、多领域、多模态、含可靠 ground truth 的时间序列推理基准，并在其上系统比较 LLM / VLM / TSLLM 三大模型族在不同模态输入下的表现，回答 scaling、模态互补、能力解耦等关键问题。

**切入角度**：把"时间序列推理"显式拆成 4 个能力维度 × 15 个具体任务，按"高文本-时序对齐 / 领域多样 / 可验证 GT / 必要时用合成数据保证精确数值答案"四条原则收题；输入侧统一渲染为 100 PPI 折线图供 VLM，并对专有模型同时跑 T、V、T+V 三种模态以直接测量模态融合能力。

**核心 idea**：用一个能力维度齐全、模态齐全、GT 可验证的统一基准，把"通用模型在时间序列上到底缺什么能力"这个含糊问题，拆成可以画 radar、可以做 scaling 曲线、可以做模态消融的量化命题。

## 方法详解

### 整体框架
TSRBench 由三块组成：（1）题库构造，从 14 个真实领域收集时序数据 + 合成 chaotic 物理系统/算法交易等可验证场景，再按 4 大维度 / 15 任务模板生成多选题或开放题，共 4125 题、15250 条 channel；（2）输入渲染器，把同一条时间序列同时转成"纯数字文本串"供 LLM、"100 PPI 折线图（多元数据竖向堆叠子图、共享时间轴、显示网格与序列名）"供 VLM、"projector embedding"供 TSLLM，专有模型额外提供 T+V 联合输入；（3）评测协议，所有模型一律开启 reasoning，用 accuracy 作为统一指标，覆盖 6 个 proprietary + 13 LLM + 13 VLM + 4 TSLLM 共 30+ 模型。

### 关键设计

1. **四维度十五任务的能力分解**:

    - 功能：把"时间序列推理"拆成 Perception（PR/NU/AD/CA 4 任务）、Reasoning（ER/CD/AR/TR/NR/DR/IR 7 任务）、Prediction（TSF/EP 2 任务）、Decision-Making（QualDM/QuantDM 2 任务）共 4×15 的二维网格，每个任务对应一个独立的能力子空间。
    - 核心思路：感知层考察对趋势/周期/平稳性/均值/噪声/异常这类底层统计属性的识别；推理层把推理细分为病因（ER）/因果发现（CD）/溯因（AR）/时序关系（TR）/数值推理（NR）/演绎（DR）/归纳（IR）七类，其中 IR 特别强调"先抽象规则、再预测特定未来事件"而不是 curve-fitting；预测层把数值预测改成多选题以降低 generalist 模型直接输出数字的难度；决策层把定性决策与需要"对多套候选规则做定量模拟比较"的定量决策分开。
    - 设计动机：以往 benchmark 把"推理"当一个整体打分，无法定位模型的具体短板；而把推理细分到 7 个子任务后，可以直接读出"GPT-5 在 AR / TR / NR 强但在 CD 上仍弱"这种细粒度结论，为后续模型设计提供方向。

2. **多模态统一输入与模态融合测试**:

    - 功能：让同一条时序数据在 LLM/VLM/TSLLM 三类模型上"公平"评测，并定量测量"文本 + 视觉"是否互补。
    - 核心思路：对每条序列同时生成（a）数字文本序列、（b）按代码渲染的折线图（单变量一张图、多变量竖向堆叠子图共享 x 轴，始终开 grid 并标注序列名）、（c）embedding；分辨率经消融固定为 100 PPI 以平衡 token 消耗与特征可见度；对支持多模态输入的专有模型额外做 T、V、T+V 三套实验，直接比较"加了图之后到底涨没涨"。
    - 设计动机：现有结论"VLM 看图就能解决一切时序问题"或"LLM 看数字串就够"都缺少同 prompt、同问题下的对照；TSRBench 通过把三种模态嵌进同一道题，定量回答模态互补与融合失败问题。

3. **可验证 GT + 合成数据兜底的数据构造**:

    - 功能：保证 4125 道题每一道都有"可机器判分、无歧义"的 ground truth，特别是涉及精确数值推理的题。
    - 核心思路：四条收集原则——高文本-时序对齐（context 必须是推理所必需，不能是装饰）、领域多样性（14 个领域分布以防过拟合到单一领域）、可验证无歧义 GT（要么用 Python 高保真仿真直接生成答案，要么直接从序列/context 中提取）、合成数据补齐数值推理（用 chaotic 物理系统与算法交易回测等可控仿真，把"无噪声的精确答案"作为压力测试数值推理与演绎逻辑的工具）。
    - 设计动机：真实世界数据虽然复杂但常常缺少精确 GT，对数值推理这类需要 0/1 判分的任务无法可靠评测；显式引入合成数据，既保留了真实数据的复杂性，又获得了能直接评测数值精确度的子集，从源头降低了"模型答错是因为题模糊"的风险。

### 损失函数 / 训练策略
本文是评测基准而非训练方法，无训练损失。评测协议：所有模型一律开启 reasoning，proprietary 模型同时跑 T / V / T+V 三种输入；对 o4-mini / GPT-5 / GPT-5-mini 同时报 low 与 high reasoning 两档；用 accuracy 作为所有 15 个任务的统一指标，并对模型规模、可视化分辨率、是否启用工具、reasoning effort 做消融。

## 实验关键数据

### 主实验

| 模型 | 输入 | Perception (PR) | Reasoning (TR) | Prediction (EP) | Decision (QualDM) | Overall |
|------|------|------|------|------|------|------|
| GPT-5 | T | 75.7 | 68.8 | 79.7 | 31.9 | 55.5 |
| o4-mini | T | 73.1 | 34.4 | 73.3 | 30.4 | 47.7 |
| GPT-5-mini | T | 72.2 | 39.4 | 67.8 | 35.5 | 46.6 |
| DeepSeek-V3.2 | T | 67.7 | 19.4 | 47.2 | 33.1 | 39.1 |
| Qwen3-235B-A22B | T | 66.0 | 28.1 | 48.9 | 34.8 | 42.2 |
| GPT-OSS-120B | T | 66.8 | 31.3 | 59.7 | 33.7 | – |
| Qwen2.5-3B | T | 46.4 | 21.2 | 58.3 | 22.7 | 33.2 |

GPT-5 (T) 以 55.5% 的 overall 居首，但在 NR / DR / IR 等需要严格规则应用的子任务上仍远低于感知类任务；最小的 Qwen2.5-3B 整体只有 33.2，留出明显的 scaling 空间。

### 消融实验

| 维度 | 关键观察 | 说明 |
|------|---------|------|
| 模型规模 | Perception / Reasoning 上 scaling 成立 | LLM 与 VLM 在感知与推理维度上都呈现稳定的规模-精度正相关 |
| 模型规模 | Prediction 上 scaling 失效 | TSF 任务在不同规模上几乎无显著提升，与其他三维度形成断点 |
| 任务相关性 | TSF 与其他任务相关性弱 | 强推理不等于强 context-aware forecasting，语义理解与数值预测解耦 |
| 模态融合 (T+V) | 文本与视觉互补但模型融合失败 | 同一道题 T 与 V 解决不同子集，T+V 通常并未在 proprietary 模型上超过两者中的较强者 |
| 可视化分辨率 | 100 PPI 为甜点 | 更高分辨率 token 开销陡增，更低则细节丢失明显 |

### 关键发现
- **scaling 在感知/推理上成立但在预测上断裂**：感知与推理维度随模型规模平滑上升，但 TSF 任务几乎是"平台曲线"，说明当前 LLM/VLM 的预训练目标和数据并未真正改善数值预测能力。
- **TSF 与其他任务相关性弱**：一个模型在 reasoning 全套任务上很强，并不能预测它在 TSF 上的表现，这意味着时间序列预测应被作为一个独立能力族单独训练与评测，而不是顺带从通用推理里"溢出来"。
- **文本与视觉高度互补但融合失败**：T 与 V 各自解决的题目集合差异巨大，加在一起的 T+V 输入在主流多模态模型上几乎不带来 1+1>2 的效果，暴露出当前多模态注意力对"同一信号的两种视图"缺乏对齐机制。
- **Decision-Making 是普遍短板**：所有模型在 QualDM / QuantDM 上都明显低于其感知与推理分数，反映出"看懂 + 推理"到"真正基于时序做决策"之间还有大量空缺。

## 亮点与洞察
- **能力网格化拆分**：把"时间序列推理"显式拆成 4×15 的能力网格，是相比 MTBench / TimeSeriesExam 等最大的方法论升级；这种网格让 radar/scaling 曲线/模态消融都能自然展开，后续工作可以直接对接到具体的弱项子任务。
- **多模态对照设计**：同一道题三种输入（T/V/T+V）的统一渲染管线，是测量"模态融合是否真的有用"的最干净设定；它直接给出了"专有 VLM 也基本没学会融合双视图"这种业界长期感觉到但缺少定量证据的结论。
- **合成数据兜底数值推理**：用 chaotic 系统、算法交易回测等可控仿真填补"真实数据缺精确 GT"的空白，使 NR / DR 这类需要 0/1 判分的任务真正具备压力测试能力，这一思路可迁移到任何强调"数值精确性"的 benchmark 构造。

## 局限与展望
- 评测全部使用 accuracy 单一指标，对 TSF 这类本应用 MSE / MAPE 的任务，转成多选题后会损失"接近但不完全对"的信号；未来可以引入分档评分或与连续指标的相关分析。
- 当前 VLM 输入只测了折线图一种可视化形式，热力图 / 频谱图 / 极坐标图等其它表征对推理任务的影响尚未覆盖。
- 决策维度只评测离散选项中的"选最优"，没有覆盖序列决策与强化学习风格的 long-horizon 决策，QuantDM 的难度天花板还可以继续抬。
- 多语言场景下的时间序列推理（如非英文金融/医疗报告 + 时序）未纳入，跨语言的 context-时序对齐是值得追加的维度。

## 相关工作与启发
- **vs TimeMMD / CiK**：他们集中在带 context 的时序预测，只覆盖 Prediction 一个维度；本文把维度扩到 4 个并把任务数扩到 15。
- **vs TimeSeriesExam**：他们用合成数据只测 perception，本文同样使用合成数据兜底数值推理，但把任务范围扩展到推理、预测、决策三大维度。
- **vs MTBench / EngineMT-QA / SciTS / TimeMQA / TSR-SUITE**：这些 benchmark 各自只覆盖时序推理空间的一个切片（窄领域、单一模态、缺少决策维度等），TSRBench 通过 4×15 的能力网格 + 四模态输入 + 30+ 模型横向比较，提供第一个面向 generalist 模型的、完整的时间序列推理评测矩阵。

## 评分
- 新颖性: ⭐⭐⭐⭐ 能力维度的显式拆分与四模态统一输入设计在时序评测领域是少见的系统化贡献，但单点技术多为工程整合。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 30+ 模型、4 模态、15 任务、多档 reasoning effort，并对可视化分辨率与工具增强做了消融。
- 写作质量: ⭐⭐⭐⭐ 维度拆解清晰，任务定义具体，发现部分有定量支撑；但子任务定义密度高，初读容易迷失。
- 价值: ⭐⭐⭐⭐⭐ 为通用大模型在时序推理上的能力定位提供首个标准化矩阵，"scaling 在预测上断裂"和"模态融合失败"两条结论对后续基础模型与多模态架构设计有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/autonomous_driving/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](../../AAAI2026/autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](../../CVPR2026/autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](../../ICCV2025/autonomous_driving/uavscenes_a_multi-modal_dataset_for_uavs.md)

</div>

<!-- RELATED:END -->
