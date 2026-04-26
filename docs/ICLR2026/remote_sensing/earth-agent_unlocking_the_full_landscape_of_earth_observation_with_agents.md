---
title: >-
  [论文解读] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents
description: >-
  [ICLR 2026][遥感][地球观测] Earth-Agent是首个基于MCP工具生态的地球观测Agent框架，统一了RGB和光谱遥感数据，通过动态调用104个专家工具实现跨模态、多步骤、定量时空推理，配套提出的Earth-Bench基准包含248个专家任务和13,729张图像，实验证明Earth-Agent远超通用Agent和遥感MLLM。
tags:
  - ICLR 2026
  - 遥感
  - 地球观测
  - Agent框架
  - MCP工具生态
  - 多模态遥感
  - benchmark
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
Earth-Agent采用ReAct型Agent架构，核心是一个POMDP循环：
- 输入：任务目标 + 遥感图像（RGB/光谱/产品）+ 交互历史
- LLM作为策略，迭代执行：工具调用 → 内存更新 → 推理 → 动作
- 输出：定量分析结果、参数反演值、空间推理结论等

### 关键设计

1. **MCP-Based工具生态系统**:
   Earth-Agent集成了104个跨五大功能套件的专业工具：
   - **Index Kit**: 光谱指数计算（NDVI、NDWI等）
   - **Inversion Kit**: 地球物理参数反演（叶面积指数、地表温度等）
   - **Perception Kit**: RGB图像感知（目标检测、场景分类、语义分割等）
   - **Analysis Kit**: 时空分析（变化检测、趋势分析等）
   - **Statistics Kit**: 统计运算（区域统计、直方图分析等）
   
   这些工具通过Model Context Protocol (MCP)进行管理，LLM可以动态组合调用。这使得Earth-Agent能够超越预训练MLLM的能力上限——对于科学级计算任务（如从Landsat数据反演地表温度），不依赖模型的内隐知识，而是调用精确的物理模型。

2. **跨模态统一处理**:
   不同于仅处理RGB的现有EO Agent，Earth-Agent原生支持三类遥感数据：
   - **光谱数据**: 多光谱/高光谱卫星图像（如Landsat、Sentinel-2）
   - **产品数据**: 已处理的遥感产品（如MODIS地表温度产品）
   - **RGB数据**: 常规可见光遥感图像
   
   LLM根据任务需求自动判断调用光谱工具还是感知工具。

3. **ReAct-POMDP决策过程**:
   将复杂EO任务建模为部分可观察马尔可夫决策过程。LLM不是一次性给出答案，而是通过多轮"思考-行动-观察"循环逐步推理。例如，"分析某地区2020-2025年植被变化趋势"需要：提取多时相NDVI → 时序分析 → 趋势拟合 → 生成结论。

4. **Earth-Bench评估基准**:
   包含248个由领域专家人工策划的任务，涵盖13,729张图像：
   - **模态覆盖**: 光谱（100题）+ 产品（88题）+ RGB（60题）
   - **双层评估协议**:
     - 端到端评估：Accuracy（最终答案正确率）+ Efficiency（工具使用效率）
     - 轨迹评估：Tool-Any-Order（是否使用了所有必要工具）、Tool-In-Order（工具顺序是否正确）、Tool-Exact-Match（逐步完全匹配）、Parameter Accuracy（工具参数准确性）

### 损失函数 / 训练策略
Earth-Agent的核心是zero-shot推理——不需要针对EO任务的额外训练。LLM通过prompt engineering和工具描述理解任务。论文同时探索了Training-Free Evolution方法（类似training-free GRPO），尝试在不微调模型权重的情况下优化Agent的工具调用策略。

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

## 相关论文

- [\[ICLR 2026\] Measuring the Intrinsic Dimension of Earth Representations](measuring_the_intrinsic_dimension_of_earth_representations.md)
- [\[ICML 2025\] High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data](../../ICML2025/remote_sensing/high-resolution_live_fuel_moisture_content_lfmc_maps_for_wildfire_risk_from_mult.md)
- [\[CVPR 2025\] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](../../CVPR2025/remote_sensing/earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ICML 2025\] LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth](../../ICML2025/remote_sensing/lighthouse_fast_and_precise_distance_to_shoreline_calculations_from_anywhere_on_.md)

<!-- RELATED:END -->
