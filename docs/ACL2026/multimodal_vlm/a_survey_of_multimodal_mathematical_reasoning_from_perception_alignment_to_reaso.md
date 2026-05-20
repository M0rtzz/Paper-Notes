---
title: >-
  [论文解读] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning
description: >-
  [ACL 2026][多模态VLM][PAR 框架] 本综述提出 Perception–Alignment–Reasoning (PAR) 过程框架 + Answer–Process–Executable (APE) 评估框架两个互补视角，系统地组织几何/图表表格/视觉应用题三大任务族…
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "PAR 框架"
  - "APE 评估"
  - "几何推理"
  - "图表/表格推理"
  - "executable intermediate"
---

# A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning

**会议**: ACL 2026  
**arXiv**: [2603.08291](https://arxiv.org/abs/2603.08291)  
**代码**: Awesome Multimodal Mathematical Reasoning（GitHub 仓库，论文中给出链接）  
**领域**: 多模态 VLM / 多模态数学推理 / 综述  
**关键词**: PAR 框架, APE 评估, 几何推理, 图表/表格推理, executable intermediate

## 一句话总结
本综述提出 Perception–Alignment–Reasoning (PAR) 过程框架 + Answer–Process–Executable (APE) 评估框架两个互补视角，系统地组织几何/图表表格/视觉应用题三大任务族，把现有方法和 benchmark 都映射到这两个十字坐标上，是首篇 process-centric 多模态数学推理综述。

## 研究背景与动机
**领域现状**：LLM 在符号/算术推理上已经接近 SOTA，但实际数学问题往往是多模态的（图、表、几何图、坐标图、混合文档）。Multimodal Mathematical Reasoning (MMR) 已经出了大量数据集和方法，但缺一个能把「感知 / 对齐 / 推理 / 评估」串起来的统一视角。

**现有痛点**：(1) 过往 MMR 综述（如 Yan et al. 2024）多是 benchmark 编目或 MLLM 角色分类（Reasoner / Enhancer / Planner），偏横向；(2) 多数评估只看最终答案，无法区分「猜对的、靠 shortcut 的、真推对的」；(3) 不同方法用的 DSL / 对齐方式 / 推理范式各异，难以横向对比。

**核心矛盾**：MMR 与纯文本数学推理本质不同——多模态耦合让感知错误、对齐错误、推理错误层层传播，单一指标无法定位失败环节；需要 process-centric 视角才能诊断「在哪一步翻车」。

**本文目标**：围绕 4 个根本问题组织 MMR：(1) 从多模态输入**提取什么**；(2) 怎么**表示和对齐**文本/视觉信息；(3) 怎么**做推理**（CoT / program-aided / tool use）；(4) 怎么**评估**整个推理过程的正确性。

**切入角度**：把「方法」与「评估」分别用 PAR、APE 两个三阶段框架建模，让方法贡献和评测目标都能挂在这两个坐标上，便于横向对比与诊断。

**核心 idea**：「PAR + APE」双框架——PAR 描述方法把多模态输入处理成正确答案的三个阶段，APE 描述评估对这三个阶段不同层次的检验，两者交叉构成 MMR 研究的统一地图。

## 方法详解

### 综述结构与分类法

#### 一级框架：PAR（方法侧）
- **Perception**（感知）：从 $X \subseteq \{T, D, C, I\}$（文本/图/图表/图像）抽取数学事实 $\mathcal{F}$，三层：低层 primitives（点线/轴/物体）→ 结构关系（incidence / parallelism / row-column）→ 量化属性（长度/角度/数值/单位）。
- **Alignment**（对齐）：把感知出来的事实映射到符号或可执行表示（geometry DSL、constraint sets、proof sketch、chart/table operators、SQL、program-of-thought trace）。
- **Reasoning**（推理）：在对齐后的表示上做可解释/可验证的推理（CoT、tree/graph of thought、RL、tool use、process feedback）。

#### 一级框架：APE（评估侧）
- **Answer**：只检查最终答案准确率（exact match / numeric tolerance）——可扩展但混淆错误来源。
- **Process**：检查中间推理步骤的有效性与视觉 grounding 一致性（如 MM-MATH 步类型、MPBench step judge、CHAMP 概念标注、MathVerse 图扰动评分）。
- **Executable**：通过运行程序 / 验证证明 / 检查 constraint 来直接评估对齐和推理的忠实度（如 GeoQA+ 程序、FormalGeo 形式化证明、E-GPS 求解器、WikiSQL 执行）。

#### 二级分类：三大任务族（每族都被 PAR 切片）
1. **Geometry Problems**：$f: (T, D) \mapsto y$，要识别点线角、空间关系、grounding 文字到几何图；方法从符号 prover（GEOS）→ 神经 VLM → hybrid pipeline (E-GPS / Pi-GPS) → LMM (G-LLaVA / GeoGPT4V / GEOX)。Benchmark：Geometry3K、GeoQA/+、PGDP5K、PGPS9K、FormalGeo7K。
2. **Chart and Table Problems**：$f: (C, Q) \mapsto a$，要识别轴/图例/行列，做 numeric/logic 推理；方法从 symbolic parsing (DVQA、PlotQA) → 神经 VLM (Pix2Struct) → instruction-tuned LMM (ChartLlama, ChartQA-X)。Benchmark：PlotQA、ChartQA(Pro)、CharXiv、FinQA、TAT-QA、MultiHiertt、DocMath-Eval、WikiSQL。
3. **Visual Math Word Problems**：$f: (I, Q) \mapsto a$，做 object counting / 属性推理 / 跨图共指。方法从符号感知 (Patch-TRM) → 神经多模态 → LMM CoT。Benchmark：IconQA、CLEVR-Math、TABMWP、MV-MATH、MathVista、MATH-V、Math2Visual。

#### 二级分类：Alignment 四视角
1. **Executable intermediates**（Inter-GPS、E-GPS、Pi-GPS、R1-OneVision）：把视觉内容转成 DSL / 程序 / SQL，可执行验证。
2. **Symbolic-Neural Hybrids**（GeoGen、MathCoder-VL、AlphaGeometry）：神经感知 + 符号推理引擎。
3. **Cross-modal Alignment Frameworks**（BLIP-2、LLaVA、Math-PUMA、VCAR、TVC、VIC）：稳定的 vision-language coupling，含渐进式/curriculum 设计。
4. **Pre-training & Fine-tuning Enablers**（Geo170K、SynthGeo228K、Math-LLaVA、MAVIS、MultiMath-300K、MAmmoTH-VL、MathV360K）：大规模对齐先验 + 任务特定监督。

#### 二级分类：Reasoning 四范式
1. **Deliberate chains**：CoT (LLaVA-CoT)、TVC 持续视觉条件、VIC 文本先规划、AtomThink 原子分解，进阶到 ToT / GoT / AGoT、VisuoThink、VReST (MCTS + self-reward)。
2. **RL-based reasoning**：奖励机制（R1-VL step-wise reward、VisualPRM、MM-PRM + MCTS、MM-Eureka rule-based RL）+ 搜索算法（DeepSeek-R1 GRPO、Vision-R1、Mulberry MCTS、Skywork R1V2 MPO+GRPO、VL-Rethinker、FAST、Think-or-Not?、VLAA-Thinking、VLM-R3、MAYE、SoTA-with-Less、AlphaProof formal RL）。
3. **Tool-augmented**（Toolformer、ToRA、COPRA、MM-REACT、Visual Sketchpad、Pi-GPS、Chameleon、MathCoder-VL）：把符号步骤外包给求解器/代码。
4. **Process feedback & verification**（VisualPRM、MM-PRM、TVC 持续视觉、VIC late fusion）：用 PRM / verifier 给中间步骤打分。

#### 二级分类：APE 评估
- **Answer-level**：ChartQA、PlotQA、FigureQA、IconQA、CLEVR-Math、FinQA、TAT-QA。
- **Process-level**：MM-MATH、MPBench、ErrorRadar、Sherlock、We-Math、MathVerse、CHAMP、PolyMATH。
- **Executable-level**：GeoQA+、FormalGeo、Inter-GPS、E-GPS、Pi-GPS。
- **Comprehensive**：MathVista、MATH-V、OlympiadBench、MathScape、CMM-Math、Children's Olympiads、MM-PRM。

## 实验关键数据

### Benchmark 全景（节选论文 Table 1，按 APE 维度 + PAR stage 组织）

| Benchmark | 年份(场) | Eval Level | PAR Stage | 贡献要点 |
|-----------|----------|------------|-----------|----------|
| ChartQA | 2022 (ACL Findings) | Answer | P+R | 真实 chart + logic/numeric QA |
| FinQA | 2021 (EMNLP) | Answer | A+R | 表/文混合 + gold programs |
| MM-MATH | 2024 (EMNLP Findings) | Process | R | step type + error label |
| MathVerse | 2024 (ECCV) | Process | All | 图扰动 + CoT step scoring |
| GeoQA+ | 2022 (COLING) | Executable | A+R | 可执行 geometry 程序 |
| FormalGeo | 2024 (MATH-AI) | Executable | A+R | Olympiad 级形式化证明 |
| MathVista | 2024 (ICLR) | Comprehensive | All | 28 子集合并的综合套件 |
| MATH-V | 2024 (NeurIPS) | Comprehensive | All | 难度校准的视觉数学 |
| MM-PRM | 2025 (arXiv) | Comprehensive | All | 真实 K-12 多模态 QA |

### 数据集规模（节选 Table 2）

| 任务族 | 代表数据集 | 规模 | 关键特征 |
|--------|------------|------|----------|
| Geometry | Geometry3K | 3,002 题 | 密集 formal language |
| Geometry | GeoQA / GeoQA+ | 5,010+ | 可执行 program supervised |
| Geometry | Geo170K | ~170K image-caption + QA | 大规模 geometry pre-train |
| Chart/Table | ChartQA | 9.6K 人工 + 23.1K 生成 | 视觉+逻辑 QA |
| Chart/Table | FinQA | 8,281 | hybrid 表+文 numeric |
| Chart/Table | DocMath-Eval | 4,000 | 含 gold programs |
| Visual MWP | IconQA | 107,439 | 多格式 |
| Visual MWP | MV-MATH | 2,009 多图 | 跨图依赖推理 |
| Visual MWP | MathVista | 6,000+ | 28 套件合并 |

### 关键发现
- 多数 benchmark 仍停留在 Answer-level，Process-level 与 Executable-level 占比偏低——评估正在被「最终答案准确率」绑架，无法暴露中间推理错误。
- 几何任务的 executable 比例最高（formal geometry 天然支持 prove/check），chart/table 与 visual MWP 的可执行支撑相对薄弱。
- Reasoning 范式上 RL-based 增长最快（2024–2025 出了 R1-VL、VisualPRM、MM-PRM、Vision-R1、Mulberry 等十多篇），process reward model 成为新热点。
- Alignment 的统一 DSL 缺失是当前最大瓶颈：geometry 用 Inter-GPS DSL、chart 用 SQL/PoT、应用题用自然语言，没有跨任务共享的对齐底座。

## 亮点与洞察
- **PAR × APE 双框架是综述本身的 contribution**：很多综述只做分类，本文构建了一个「方法过程 × 评估层级」的二维坐标，让每篇工作都能找到自己的位置，方便后续工作做横向比较和补缺。
- **将「评估如何与方法阶段对齐」当主线**：把 Process-level evaluation 与 Reasoning 阶段、Executable evaluation 与 Alignment 阶段显式挂钩，强调「评估应该检验的是哪一阶段的能力」，这种对齐能直接驱动后续 benchmark 设计。
- **Failure cause attribution 的视角**：综述反复强调「Perception 错会传到 Alignment，Alignment 错会污染 Reasoning」，提醒读者诊断 MLLM 失败时必须按 PAR 三阶段做归因，而不是笼统说「模型不会推理」。
- **跨任务族归一化**：几何 / 图表 / 应用题三类任务在传统综述中常被分开讨论，本文用 PAR 把它们用同一套语言描述（perception extract 什么、alignment 用什么 DSL、reasoning 用 CoT 还是 tool），为跨任务统一建模铺路。
- **未来方向写得务实**：作者明确指出统一 DSL、轻量 reward model、自适应推理深度、process reward + symbolic verifier 是下一步关键技术；教育/无障碍/AR-VR 是潜在应用。

## 局限与展望
- 作为综述，PAR/APE 框架在某些边界案例上分类会模糊（例如 hybrid 方法可能同时跨多个 alignment 视角），后续需要更精细的子分类。
- benchmark 跟进截止到 2025 年 NeurIPS / arXiv，少数 2026 在投/在 review 工作未覆盖。
- 多数实验数字依赖原文引用，没有做统一 reproduce，跨论文的绝对数值可比性受限。
- 没有大篇幅讨论效率维度（推理延迟、显存）；MMR 方法的工程落地除了准确率还要看成本，留给后续 survey。
- 中文/多语言 MMR（仅 CMM-Math 一项）覆盖不足，全球化数学教育场景需要更多评估。
- 没有覆盖**多 agent**多模态数学推理（如 multi-agent geometry prover），这是 2025–2026 的新热点。

## 相关工作与启发
- **vs Yan et al. (2024) MMR survey**: 该工作侧重 MLLM 角色（Reasoner / Enhancer / Planner）与 benchmark 编目；本文则提出 PAR/APE 过程框架，做法更系统化。
- **vs Ahn et al. (2024) 纯文本数学推理 survey**: 文本数学推理综述聚焦 CoT / RL / verification；本文专注多模态特有挑战（perception、cross-modal alignment）。
- **vs Li et al. (2025) Perception-Reason-Think-Plan**: 后者把多模态推理拆成 4 个动作；本文 PAR 更紧凑（3 阶段）且与具体数学 DSL 绑定，更适合数学子领域。
- **vs Lu et al. (2023d) survey of deep learning for math reasoning**: 较早的综述，主要在 2022 年前文献；本文补完了 2023–2025 的 LMM / RL / tool-use 爆发期。
- **对 idea generation 的启发**：(1) Unified DSL across geometry/chart/word problem 是一个高价值开放问题；(2) Process Reward Model + Symbolic Verifier 的混合 reward 是下一步 RL agent 在数学上的主要发展方向；(3) APE 的 process-level benchmark 仍稀缺，可作为新数据集贡献点。

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架新（PAR + APE 双坐标），不只是文献编目，提供了过程视角的归因工具。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 30+ benchmark、100+ 方法引用，三大任务族 + 四种 alignment + 四种 reasoning 都展开了。
- 写作质量: ⭐⭐⭐⭐ 结构清晰、表格密集、PAR/APE 切分一致；某些子节稍偏 catalog，可读性还可以再精炼。
- 价值: ⭐⭐⭐⭐⭐ 对新进入 MMR 领域的研究者是必读地图；对评估设计、reward model 设计、跨任务统一建模都给出了清晰指向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[ICLR 2026\] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](../../ICLR2026/multimodal_vlm/visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)

</div>

<!-- RELATED:END -->
