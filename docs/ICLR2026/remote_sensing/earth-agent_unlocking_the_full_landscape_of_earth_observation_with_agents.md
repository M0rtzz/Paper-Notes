---
title: >-
  [论文解读] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents
description: >-
  [ICLR 2026][遥感][地球观测] Earth-Agent是首个基于MCP工具生态的地球观测Agent框架，统一了RGB和光谱遥感数据，通过动态调用104个专家工具实现跨模态、多步骤、定量时空推理，配套提出的Earth-Bench基准包含248个专家任务和13,729张图像…
tags:
  - "ICLR 2026"
  - "遥感"
  - "地球观测"
  - "Agent框架"
  - "MCP工具生态"
  - "多模态遥感"
  - "benchmark"
---

# Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents

**会议**: ICLR 2026  
**arXiv**: [2509.23141](https://arxiv.org/abs/2509.23141)  
**代码**: [opendatalab/Earth-Agent](https://github.com/opendatalab/Earth-Agent)  
**领域**: 遥感 / LLM Agent  
**关键词**: 地球观测, Agent框架, MCP工具生态, 多模态遥感, benchmark

## 一句话总结
Earth-Agent是首个基于MCP工具生态的地球观测Agent框架，统一了RGB和光谱遥感数据，通过动态调用104个专家工具实现跨模态、多步骤、定量时空推理，配套提出的Earth-Bench基准包含248个专家任务和13,729张图像，实验证明Earth-Agent远超通用Agent和遥感MLLM。

## 研究背景与动机
地球观测(EO)是理解地球系统演变状态的关键任务。近年来，多模态大语言模型(MLLM)已经推动了遥感研究的进步，但仍然存在根本性的能力缺失：

现有MLLM在EO领域的痛点：
- **仅限RGB感知**: 无法处理光谱数据（多光谱、高光谱、SAR等），而这正是科学级遥感分析的核心
- **浅层推理**: 无法进行需要多步骤推理和领域特定工具调用的复杂任务
- **缺乏定量能力**: 不能执行地球物理参数反演、定量时空分析等需要精确计算的科学任务
- **无系统评估**: 缺乏覆盖全模态、兼顾推理轨迹和最终结果的评估协议

现有Agent方法的局限：
- 局限于RGB感知，不处理光谱数据
- 推理深度不足，工具调用能力初级
- 没有面向EO的系统评估基准

Earth-Agent的切入角度：将EO分析建模为基于ReAct风格的POMDP过程，LLM作为策略网络，通过MCP协议动态调用领域专家工具，打通RGB和光谱模态。

## 方法详解

### 整体框架
Earth-Agent是一个ReAct型Agent框架，把地球观测分析建模成部分可观察马尔可夫决策过程(POMDP)：以LLM作为策略网络，输入任务目标、遥感图像(RGB/光谱/产品)和交互历史，迭代地"思考-调用工具-观察结果-更新记忆"，最终输出定量分析、参数反演值或时空推理结论。关键之处在于真正的计算不由LLM内隐完成，而是委托给一个由104个领域专家工具组成的MCP工具生态，LLM只负责决定调什么工具、按什么顺序调、传什么参数。

### 关键设计

**1. MCP工具生态：把科学级计算从模型内隐知识里解耦出来**

预训练MLLM处理遥感问题时，地表温度反演、光谱指数计算这类需要精确物理模型的任务只能靠"内隐知识"硬猜，结果既不可靠也无法定量。Earth-Agent把这部分能力外包给104个专家工具，按功能分成五大套件：Index Kit负责光谱指数计算(NDVI、NDWI等)，Inversion Kit负责地球物理参数反演(叶面积指数、地表温度等)，Perception Kit负责RGB图像感知(目标检测、场景分类、语义分割等)，Analysis Kit负责时空分析(变化检测、趋势分析等)，Statistics Kit负责区域统计、直方图等运算。这些工具统一通过Model Context Protocol(MCP)注册和管理，LLM可以按需动态组合调用。这样一来，从Landsat数据反演地表温度时调用的是真正的物理模型而非模型的猜测，Earth-Agent的能力上限因此突破了底座MLLM本身，工具集也因为MCP的标准接口而可扩展、可替换。

**2. 跨模态统一处理：让一个Agent同时吃下RGB、光谱和产品数据**

现有EO Agent大多只能处理RGB可见光图像，而科学级遥感分析的核心恰恰是光谱数据。Earth-Agent原生支持三类输入：多光谱/高光谱卫星图像(如Landsat、Sentinel-2)、已处理的遥感产品(如MODIS地表温度产品)、以及常规RGB图像。面对一个任务，LLM会根据需求自动判断该走光谱工具链还是感知工具链——例如查地表温度走Inversion Kit的光谱反演，做场景识别则走Perception Kit。这种统一处理打通了此前被RGB局限割裂的两个世界，让定量光谱分析和常规视觉理解能在同一个框架里完成。

**3. ReAct-POMDP多步推理：把复杂任务拆成可观察的决策链**

很多EO任务无法一步答出，需要多个工具串联。Earth-Agent把它建模为POMDP，LLM不是一次性给答案，而是在每一轮根据当前观察决定下一个动作，逐步逼近结论。比如"分析某地区2020-2025年植被变化趋势"会被分解成：提取多时相NDVI、做时序分析、拟合趋势、再综合生成结论这样一条工具链。每一步的中间结果都进入记忆供后续推理参考，这让Agent能处理单次调用搞不定的长链路定量分析，也让整个推理过程变得可观察、可评估。

**4. Earth-Bench评估基准与双层协议：既看结果对不对，也看过程走得对不对**

为了系统评估EO Agent，论文构建了Earth-Bench：248个由领域专家人工策划的任务、13,729张图像，模态上覆盖光谱(100题)、产品(88题)和RGB(60题)。评估采用双层协议：端到端层看最终结果，包括答案正确率Accuracy和工具使用效率Efficiency(实际工具数相对ground truth的比值)；轨迹层则深入推理过程，用Tool-Any-Order衡量是否用全了所有必要工具、Tool-In-Order衡量工具顺序是否正确、Tool-Exact-Match衡量是否逐步完全匹配、Parameter Accuracy衡量传给工具的参数是否准确。只评最终答案会掩盖"蒙对"的情况，轨迹层的引入才能真正刻画Agent的行为质量。

### 损失函数 / 训练策略
Earth-Agent核心是zero-shot推理，不需要针对EO任务额外训练，LLM仅凭prompt和工具描述理解任务并完成调用。论文另外探索了Training-Free Evolution方法(思路类似training-free GRPO)，尝试在不微调模型权重的前提下优化Agent的工具调用策略。

## 实验关键数据

### 主实验
不同LLM后端在Earth-Bench上的表现：

| 模型 | Tool-Any-Order | Tool-In-Order | Tool-Exact-Match | Parameter | Accuracy | Efficiency |
|------|---------------|---------------|-------------------|-----------|----------|------------|
| DeepSeek-V3 (IF) | 0.892 | 0.876 | 0.741 | 0.572 | — | — |
| GPT-5 (AP) | 0.766 | 0.750 | 0.596 | 0.462 | 59.32% | 1.531 |
| Kimi-K2 (IF) | 0.806 | 0.799 | 0.633 | 0.522 | 62.71% | 1.410 |

### 消融实验

| 对比 | 关键指标 | 说明 |
|------|---------|------|
| Earth-Agent vs 通用Agent框架 | Accuracy | Earth-Agent显著优于LangChain等通用Agent |
| Earth-Agent vs 遥感MLLM | RGB benchmark | 在遥感基准上超越专用遥感MLLM |
| 光谱任务 vs RGB任务 | Tool-Exact-Match | 光谱任务工具链更长更复杂，精确匹配难度更大 |
| 不同LLM backbone | 综合表现 | 更强的LLM带来更好的工具调用和推理能力 |

### 关键发现
- DeepSeek-V3在工具使用准确性上表现最好（Tool-Any-Order 0.892）
- Kimi-K2在最终答案准确率上略胜GPT-5（62.71% vs 59.32%）
- 工具效率(Efficiency)普遍>1.0，说明模型倾向于使用比ground truth更多的工具
- 参数准确性(Parameter)是最大瓶颈（最高仅0.572），说明LLM对遥感领域参数的理解仍有限
- 工具顺序(Tool-In-Order)与工具存在性(Tool-Any-Order)差距不大，说明模型基本能把握正确顺序

## 亮点与洞察
- **范式转换**: 从MLLM直接回答遥感问题，转向Agent动态调用专家工具——这是EO-AI的重要方向转变
- **MCP协议的应用**: 使用MCP管理工具是工程上的良好实践，使得工具集可扩展、可替换
- **双层评估设计精妙**: 不仅评估最终结果，还评估推理过程（工具调用轨迹），这对理解Agent行为至关重要
- **实际科学价值**: 地球物理参数反演、定量时空分析等任务超越了传统CV的范畴，具有真正的科学应用价值
- **104个工具的构建**: 这本身就是一个重大的工程贡献，涵盖了EO分析的主要环节

## 局限与展望
- 强依赖LLM的能力上限——如果LLM推理出错，整个链路就会崩溃
- 参数准确性（Parameter Accuracy最高0.572）显示LLM对遥感领域知识仍有不足
- 工具效率>1说明模型倾向冗余调用，需要优化推理效率
- 仅评估了有限的几个LLM backbone，对开源小模型的适用性未知
- Earth-Bench规模（248题）相比NLP/CV基准仍较小
- 实时性方面未讨论——多步工具调用的延迟在实际遥感应用中可能是问题
- Training-Free Evolution的效果有待系统评估

## 相关工作与启发
- **ReAct (Yao et al., 2023)**: 思考-行动范式的奠基工作，Earth-Agent在EO领域的具体实例化
- **ToolFormer / Gorilla**: LLM工具使用的先驱工作，Earth-Agent将此扩展到104个领域专家工具
- **GeoChat / RS-ChatGPT**: 现有遥感MLLM，但仅处理RGB且不支持工具调用
- **Model Context Protocol (MCP)**: Anthropic提出的工具管理协议，Earth-Agent是MCP在科学领域的重要应用案例
- 启发：Agent + 领域工具的范式在其他科学领域（如天文、生物、材料科学）同样适用

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Measuring the Intrinsic Dimension of Earth Representations](measuring_the_intrinsic_dimension_of_earth_representations.md)
- [\[ICML 2025\] High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data](../../ICML2025/remote_sensing/high-resolution_live_fuel_moisture_content_lfmc_maps_for_wildfire_risk_from_mult.md)
- [\[CVPR 2025\] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](../../CVPR2025/remote_sensing/earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ICML 2025\] LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth](../../ICML2025/remote_sensing/lighthouse_fast_and_precise_distance_to_shoreline_calculations_from_anywhere_on_.md)

</div>

<!-- RELATED:END -->
