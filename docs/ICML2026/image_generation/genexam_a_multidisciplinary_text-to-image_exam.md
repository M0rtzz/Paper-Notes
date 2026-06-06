---
title: >-
  [论文解读] GenExam: A Multidisciplinary Text-to-Image Exam
description: >-
  [ICML 2026][图像生成][多学科考试] GenExam 把"画图考试"作为衡量 T2I 模型推理-理解-生成综合能力的金标准，给 10 个学科、1000 道题各配上 ground-truth 图 + 细粒度评分点…
tags:
  - "ICML 2026"
  - "图像生成"
  - "多学科考试"
  - "文本到图像评测"
  - "评分点"
  - "MLLM-as-judge"
  - "GPT-Image-1.5"
---

# GenExam: A Multidisciplinary Text-to-Image Exam

**会议**: ICML 2026  
**arXiv**: [2509.14232](https://arxiv.org/abs/2509.14232)  
**代码**: https://github.com/OpenGVLab/GenExam (有)  
**领域**: 多模态 VLM / 评测基准 / 文本到图像生成  
**关键词**: 多学科考试, 文本到图像评测, 评分点, MLLM-as-judge, GPT-Image-1.5

## 一句话总结
GenExam 把"画图考试"作为衡量 T2I 模型推理-理解-生成综合能力的金标准，给 10 个学科、1000 道题各配上 ground-truth 图 + 细粒度评分点，结果连最强闭源模型 Nano Banana Pro 也只有 70.2% strict 分，多数开源 T2I/统一 MLLM 不到 3%。

## 研究背景与动机

**领域现状**：多学科推理已有 MMLU、MMMU、Humanity's Last Exam 等评测，但都是"看懂题目"的理解任务；T2I 端的多学科基准（MMMG、OneIG-Bench、SridBench）以"概念插图"为主，评测准则宽松，类似"用图像说明一个概念"而非"完成一道画图考题"。

**现有痛点**：现有 T2I 评测 (i) prompt 简短宽泛，(ii) 没有参考图也没有评分细则，(iii) 知识面浅且无层次化分类，(iv) 评测端要么靠 CLIP/VQA score（抓不到学科正确性）要么用 MLLM-as-judge 给一句话指令（漏掉大量细节）。导致"画对了几根化学键"、"圆和切线的位置关系"这类硬错误根本评不出来。

**核心矛盾**：多学科图像的关键不是真实感或美学，而是语义正确性——一个原子画错、一个箭头反向，整张图就废了；但通用图像评测指标无法捕捉这种细粒度对错。

**本文目标**：(1) 构造一个像 AP / A-level / IB 画图题那样有标准答案、评分细则、知识分类的 T2I benchmark；(2) 设计一套能可靠判定语义正确性 + 视觉合理性的自动评测协议；(3) 用它系统暴露当前 T2I / 统一 MLLM 在学科生成能力上的真实差距。

**切入角度**：把考试评分逻辑搬到 T2I 评测——每道题不仅有 prompt 和参考图，还有人工 + GPT-5 共同制定的"评分点"列表（如"分子是否恰好含 8 个 C 原子？"），用 MLLM 把每个评分点当 VQA 来答 Yes/No，最后按分数加权汇总。

**核心 idea**：像批改画图考卷一样评测 T2I 模型——每张图先按 customized scoring points 算"语义正确率"，再按 spelling / readability / logical consistency 三个 0-2 分项算"视觉合理性"，最终给出 strict 与 relaxed 双分数。

## 方法详解

### 整体框架
GenExam 包含三大组件：(1) 1000 题的题库，覆盖数学/物理/化学/生物/计算机/地理/经济/音乐/历史/工程 10 个一级学科，按 ISCED-F 织出 10/40/132/236 的四层分类；(2) 每题配 ground-truth 图 + 3-14 个评分点（平均 6.9 个，权重和为 1）+ 平均 74.8 词的 exam-style prompt；(3) 双维度评测协议——semantic correctness（0-1）+ visual plausibility（spelling/logic/readability 各 0-2），汇出 strict 与 relaxed 两个最终分。

### 关键设计

1. **Scoring Points 评分细则**:

    - 功能：把模糊的"图像是否正确"问题降维成一组确定的 VQA 判定题。
    - 核心思路：每题由 GPT-5 起草 3-14 个 yes/no 评分点（例如"分子是否含恰好 8 个碳？"），人工审核细化；评测时让 MLLM judge 看着生成图 + 参考图，逐点回答 Yes/No，semantic correctness $= \sum_i s_i \cdot \mathbb{1}[\text{answer}_i=\text{Yes}]$，所有分数总和为 1。
    - 设计动机：单条 MLLM 指令评测会漏掉细节（如化学键数量、几何位置关系、乐谱音符）；把每个关键约束显式拆出来才能稳定捕捉学科级错误。

2. **双分数评测协议（Strict + Relaxed）**:

    - 功能：用两套尺度同时刻画"完全正确率"和"接近正确程度"，避免一刀切。
    - 核心思路：strict 分 = 完全满足所有评分点 + spelling/logic/readability 均为 2 的图像比例（一张图错一点就算 0）；relaxed 分 = $0.7\cdot\text{semantic}+0.1\cdot\text{spell}+0.1\cdot\text{logic}+0.1\cdot\text{read}$（权重由人类偏好对齐确定）。strict 凸显"几乎没人能完美交卷"的难度，relaxed 区分大量低分模型之间的差异。
    - 设计动机：纯 strict 会让大部分模型并列 0% 失去信息量；纯加权平均又掩盖了"差一点就全错"的学科特性，所以两者并行汇报。

3. **数据策展流水线**:

    - 功能：保证题目难度、学科覆盖与评分点质量。
    - 核心思路：先按四层分类生成关键词→网图抓取 + 已有 MLLM 数据集筛选→GPT-5 按文本丰富度/学科密度/复杂度打分过滤→GPT-5 起草 prompt 和 scoring points→PhD 标注员人工审核与修订；最终 1000 题里 hard 占 38%、medium 38%、easy 24%，每题 prompt 长 24-173 词。
    - 设计动机：网图质量参差不齐，纯人工成本高，纯 GPT-5 又会"凑数"；GPT-5 + 人工双层审核兼顾规模和严谨。

### 损失函数 / 训练策略
本文是评测 benchmark，无训练；唯一可调的是评测端 MLLM judge（默认 GPT-5，reasoning effort 设为 low；附录验证 Gemini-3-Flash 等替代品仍与人类高度一致）。

## 实验关键数据

### 主实验
在 17 个模型上测 strict / relaxed 双分数（节选）：

| 模型 | 类型 | Strict ↑ | Relaxed ↑ |
|------|------|---------:|----------:|
| Nano Banana Pro | 闭源 | **70.2** | **93.0** |
| GPT-Image-1.5 | 闭源 | 42.5 | 81.5 |
| GPT-Image-1 | 闭源 | 13.1 | 62.2 |
| Seedream 4.5 | 闭源 | 12.3 | 59.5 |
| FLUX.2 max | 闭源 | 8.6 | 61.6 |
| FLUX.2 dev | 开源 T2I | 2.4 | 42.3 |
| Qwen-Image-2512 | 开源 T2I | 1.5 | 35.3 |
| BAGEL (thinking) | 开源统一 MLLM | 0.0 | 12.9 |
| Janus-Pro | 开源统一 MLLM | 0.0 | 9.5 |

最强闭源模型也未及格，多数开源 T2I 几乎全军覆没；开源统一 MLLM 全为 0 strict，比专门 T2I 还差。

### 消融实验

| 评测器 | 与人类 Kendall $\tau$ | Pearson $r$ |
|--------|----------------------:|------------:|
| Relaxed by GPT-5 | **0.675** | **0.844** |
| Relaxed by Gemini-3-Flash | 0.661 | 0.826 |
| 仅 Semantic Correctness | 0.633 | 0.806 |
| VQA Score | 0.145 | 0.179 |
| CLIP Score | 0.116 | 0.165 |

各维度 MAE：semantic 0.10、spelling 0.11、readability 0.20、logic 0.28，均很低，说明评测稳定。

### 关键发现
- **统一 MLLM 反而比专门 T2I 差**：BAGEL、Show-o2 等开源统一模型 strict 全 0，relaxed 也低于 FLUX.2 dev / Qwen-Image-2512，说明"用同一模型理解 + 生成"对学科图像还远未跑通。
- **bottleneck 不在知识，而在视觉执行**：FLUX.2 dev 在历史题里能正确指出埃及/伊朗/印度/中国的地理位置，却画不出对应的图形元素 —— 模型缺的是"把知识翻译成可读图像"的能力。
- **CLIP / VQA score 完全失效**：与人类的相关性接近 0.1，说明传统 T2I 评测指标根本抓不到学科正确性。
- **开源应先攻基本功**：开源模型在 spelling 和 logic consistency 上掉得最猛，提示先把文字渲染、坐标对齐这种基本功补齐，再谈推理。

## 亮点与洞察
- **把"评分细则"显式化是 LLM/T2I 评测可推广的范式**：把笼统的"对/错"拆成结构化 yes/no 列表后，MLLM judge 的 MAE 立刻可控、相关系数远超传统指标。这套思路也适用于 chart QA、code generation、数学解答评测等子任务。
- **strict + relaxed 双指标设计巧妙**：一个突出难度天花板（拉开顶尖闭源差距），一个揭示底层差异（区分多数 0 分模型），既不会被"全部满分"或"全部 0 分"压扁。
- **"考试视角"重新框定了 T2I 评测目标**：以往评 T2I 关心 fidelity / aesthetic / alignment，这里转向"正确性 + 可读性"，更贴近 AGI 路线上对"专家级智能"的检验。
- **数据策展协议可复用**：GPT-5 起草 + 人工细化的双层 pipeline 在很多需要 scoring criteria 的 benchmark 上都能照搬。

## 局限与展望
- 1000 题对覆盖 10 个学科 + 4 层分类来说仍偏少，部分子领域（如音乐）样本只够几十张，统计稳定性受限。
- 依赖 GPT-5 / Gemini-3-Flash 这类前沿闭源 MLLM 做 judge，长期可复现性和成本是隐患；附录测了开源 judge，但与人类相关系数有所下降。
- 评分点权重平均分配且总和为 1，没有体现"主结构 vs 细节"的层次重要度。
- 题目集中在"画图考试"，对动画、视频、3D 等学科可视化任务尚未覆盖。

## 相关工作与启发
- **vs MMMU / MMLU / Humanity's Last Exam**: 都是多学科考试，但都只评 understanding；GenExam 把同样严肃的考试规模带到了 generation 端。
- **vs MMMG / OneIG-Bench / SridBench**: 同为学科图像生成评测，但前者强调"概念插图"宽松；GenExam 的 prompt 更长、约束更硬、评分更细。
- **vs RISEBench / WiScore**: 借鉴了 strict 二值评分和人类对齐的加权方式，但首次把"customized scoring points"扩展到学科级评测。
- **可迁移启发**：把"VQA-style 评分点"做成模型评测的通用接口，对多模态推理、agent benchmark、代码生成评测同样适用；同时也提示统一 MLLM 研究者：当前 unified 架构在学科生成上的劣势提醒"理解 + 生成共用 backbone"的设计仍需重新打磨。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个学科级 T2I 考试 benchmark，scoring-points 协议是显著创新。
- 实验充分度: ⭐⭐⭐⭐⭐ 17 个模型 × 10 学科 × 双指标 + 5 位人类标注 250 题对齐 + 多 evaluator 鲁棒性，覆盖广。
- 写作质量: ⭐⭐⭐⭐ 图表清晰、协议讲得很透；附录细节稍多，主文需要回头查 token 不太友好。
- 价值: ⭐⭐⭐⭐⭐ 给 T2I 社区第一次给出了"考试级"评测，长期会成为统一 MLLM 学科能力的标尺。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[ICML 2026\] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)
- [\[ICML 2026\] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)

</div>

<!-- RELATED:END -->
