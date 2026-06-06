---
title: >-
  [论文解读] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models
description: >-
  [ACL 2025][社会计算][多模态安全] 提出"双模态隐式毒性"(dual-implicit toxicity)概念——仅当结合图文两个模态时才能被识别的偏见与歧视，构建了包含317K问题、12类23子类的MDIT-Bench基准，并通过长上下文越狱揭示了主流多模态大模型中大量可被激活的隐藏毒性。
tags:
  - "ACL 2025"
  - "社会计算"
  - "多模态安全"
  - "隐式毒性"
  - "双模态隐式毒性"
  - "大模型安全评估"
  - "长上下文越狱"
---

# MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models

**会议**: ACL 2025  
**arXiv**: [2505.17144](https://arxiv.org/abs/2505.17144)  
**作者**: Bohan Jin, Shuhan Qi, Kehai Chen, Xinyi Guo, Xuan Wang (HIT Shenzhen, Univ. Barcelona)  
**代码**: [nuo1nuo/MDIT-Bench](https://github.com/nuo1nuo/MDIT-Bench)  
**领域**: 社会计算  
**关键词**: 多模态安全, 隐式毒性, 双模态隐式毒性, 大模型安全评估, 长上下文越狱  

## 一句话总结

提出"双模态隐式毒性"(dual-implicit toxicity)概念——仅当结合图文两个模态时才能被识别的偏见与歧视，构建了包含317K问题、12类23子类的MDIT-Bench基准，并通过长上下文越狱揭示了主流多模态大模型中大量可被激活的隐藏毒性。

## 研究背景与动机

### 问题背景
多模态大模型(LMM)如GPT-4o、Gemini等被广泛使用，但其输出可能包含有害、歧视性内容。现有安全研究主要聚焦于**显式毒性**（直接包含侮辱性语言）和**单模态隐式毒性**（单一模态即可检测的隐含有害内容），但忽略了更隐蔽的跨模态毒性形式。

### 已有工作的不足
- 大部分毒性基准关注显式或单模态隐式毒性，缺少对需要**跨模态推理**才能检出的毒性评估
- 很多工作仅限于纯文本领域，缺少多模态安全评估
- 已有基准数据规模有限，且多使用judge模型打分——但在隐式毒性场景下最强模型本身表现都不合格，无法胜任judge角色

### 核心动机
填补更细粒度偏见与歧视在多模态场景下的安全评估空白。核心观察：当问句中的关键信息被图片替代（如"图中的人适合当领导吗？"），毒性只在文字+图片结合时才浮现——单独看文本是中性问题，单独看图片也无害。

## 方法详解

### 整体框架
构建流程分为四个阶段：**问题生成→数据清洗→模态扩展→基准构建**，核心方法称为 Multi-stage Human-in-loop In-context Generation。在各阶段引入人工干预以对齐人类价值观。

### 关键设计1：双模态隐式毒性的定义与分类

将毒性按隐蔽程度分为三级：
- **显式毒性**：含直接侮辱/歧视语言，容易检测
- **单模态隐式毒性**：不含攻击性词汇，但通过隐喻、讽刺等方式在单一模态中可检出
- **双模态隐式毒性**：文本和图像各自无害，仅在结合两个模态后才呈现偏见/歧视

数据集覆盖12类毒性（种族歧视、性别歧视、阶层歧视、恐同、民族主义、年龄歧视、残障歧视、宗教歧视、外貌歧视、亚文化歧视、神经歧视、其他）和23个子类、780个具体话题。

### 关键设计2：多阶段人机协同数据生成

1. **问题生成**：从CVALUES等来源收集种子问题，人工创建隐式毒性问题，然后构建"伪多模态"版本——把关键毒性词替换为"图中的[]"。以此为示例做ICL扩展
2. **数据清洗**：利用被替换词(Replaced Word)的分布进行过滤，保留780个高质量关键词
3. **模态扩展**：以被替换词为关键字爬取网络图像，人工过滤模糊/不相关图片，得到29,097张图像
4. **基准构建**：每题构建5个选项——Ans1(非毒性正确答案)、Ans2(毒性答案)、Ans3(中间嵌入毒性的长答案)、Ans4(图像描述干扰项)、Ans5(替换关键词的迷惑项)

### 关键设计3：难度分级与隐藏毒性度量

三个难度级别：
- **Easy**：基于MMHS150K的显式/单模态隐式毒性，91,892题
- **Medium**：MDIT-Dataset中的双模态隐式毒性，112,873题
- **Hard**：在Medium基础上添加长上下文越狱(Long-Context Jailbreaking)，在prompt前注入大量有毒示例(32/64/128-shot)

提出Hidden Toxicity (HT)指标量化模型在hard级别相对medium级别增加的毒性：

$$HT(\mathcal{G}) = \sum_{i \in N} \left(1 - \frac{Acc_{n=i}}{Acc_{n=0}}\right) \cdot \text{Norm}_N(i)$$

其中$N=\{32,64,128\}$，归一化因子遵循幂律衰减。HT越高说明模型隐藏毒性越多。

## 实验关键数据

### 实验1：Easy与Medium级别准确率

| 模型 | Acc (Medium) | Acc (Easy) |
|------|-------------|------------|
| Qwen2-VL-7B | **67.2%** | 85.9% |
| Qwen2-VL-72B-AWQ | 65.5% | 87.7% |
| LLaVA-NeXT | 42.3% | 79.7% |
| LLaVA-1.5-13B | 35.9% | 71.1% |
| LLaVA-1.5-7B | 27.2% | 67.1% |
| BLIP2 | 40.9% | 75.3% |
| CogVLM2 | 16.3% | 72.2% |
| InstructBLIP | 12.4% | 33.2% |
| 随机基线 | 20.0% | 20.0% |

绝大多数模型在Medium级别的准确率远低于Easy级别，InstructBLIP和CogVLM2甚至低于随机基线。

### 实验2：Hard级别与隐藏毒性指标

| 模型 | Acc (Med.) | Acc (32-shot) | Acc (64-shot) | Acc (128-shot) | HT |
|------|-----------|---------------|---------------|----------------|------|
| Qwen2-VL-7B | 67.2% | 47.7% | 41.8% | 33.7% | 0.476 |
| Qwen2-VL-72B-AWQ | 65.5% | 37.2% | 32.3% | 30.8% | 0.496 |
| BLIP2 | 40.9% | 22.5% | 20.2% | 19.5% | **0.530** |
| LLaVA-NeXT | 42.3% | 35.1% | 32.9% | — | 0.298 |
| LLaVA-1.5-13B | 35.9% | 28.9% | 26.8% | — | 0.279 |

BLIP2的隐藏毒性最高(0.530)；Qwen2-VL-7B虽然Medium上表现最好(67.2%)，128-shot后骤降至33.7%，HT = 0.476。

### 闭源模型结果（子集测试）

| 模型 | Acc (Med.) | HT |
|------|-----------|------|
| Gemini-1.5-Pro | 65.65% | 0.296 |
| Claude-3.5-Sonnet | 53.37% | 0.261 |
| GPT-4o | 41.50% | 0.124 |
| GPT-4o-mini | 43.63% | 0.401 |

GPT-4o隐藏毒性最低但Medium准确率也低，说明其毒性在常规条件下已"泄漏"；Gemini虽然Medium最强但HT也不低。

## 关键发现

1. **双模态隐式毒性是LMM的普遍盲区**：即使最强的Qwen2-VL-7B也仅67.2%准确率，离安全标准差距明显
2. **模型规模并非越大越好**：Qwen2-VL-7B反而略优于72B版本，大模型可能因生成更长回复而落入Ans3陷阱
3. **不同毒性类别检测难度差异大**：Sexism和Neurological Discrimination检测较好，Classism和Subcultural Discrimination检测困难，可能因训练数据中相应类别偏少
4. **隐藏毒性广泛存在且可被渐进激活**：随shot数增加，有毒选项选择比例呈近线性上升（对数尺度下），遵循幂律关系
5. **Medium和Hard表现不严格相关**：Medium表现好不等于HT低，说明模型的显性安全对齐不足以消除深层偏见

## 亮点与洞察

- **概念贡献突出**：清晰定义了显式→单模态隐式→双模态隐式的毒性三级体系，填补了跨模态安全评估中的概念空白
- **数据构建方法实用**：Multi-stage Human-in-loop ICL方法在自动化与质量间取得了好的平衡，仅需少量人工迭代即可扩展至317K规模
- **五选项设计精巧**：Ans3(中间嵌毒)测试段落级毒性检测、Ans4(图像描述)测试指令理解、Ans5(关键词替换)测试视觉信息使用，多维度评估模型能力
- **HT指标有启发性**：将"已显现毒性"与"隐藏可激活毒性"分开度量，为安全对齐提供了更精细的衡量维度
- **人工评估验证了基准有效性**：第二阶段人工评估整体准确率98%，确认毒性标注质量可靠

## 局限性

- **仅覆盖偏见与歧视类毒性**：不涉及隐私泄露、危险行为指导等其他安全维度
- **多选题格式的局限**：无法评估模型的自由生成行为，也不强制输出推理过程，限制了对模型决策机制的分析
- **数据生成依赖模型**：种子扩展通过LLM ICL完成，可能引入系统偏差
- **图像来源于网络爬取**：虽做了匿名化和筛选，但图像质量和代表性仍有局限
- **缺少防御/解毒方案**：仅做了评估，未提出针对双模态隐式毒性的缓解方法
- **闭源模型仅在子集上测试**：成本限制导致GPT-4o等的结果可能不够精确

## 相关工作与启发

- **MLLMGUARD** (Gu et al. 2024)：12类社交媒体数据+红队攻击评估集，但关注显式毒性
- **SafeBench** (Ying et al. 2024)：LLM judge标注的2,300条有害查询，规模较小
- **SALAD-Bench** (Li et al. 2024)：含攻击增强/防御增强子集，但限于纯文本
- **MM-SafetyBench** (Liu et al. 2025)：多模态安全评估四步法，但未聚焦于需要跨模态推理的隐式毒性
- **SIUO** (Wang et al. 2024b)：跨模态安全对齐挑战集，与本文互补
- **Many-shot Jailbreaking** (Anil et al. 2024, NeurIPS)：启发了本文Hard级别的长上下文越狱方法

**启发**：双模态隐式毒性的核心挑战在于模型需要**同步完成**跨模态信息整合和深层语义理解——当前模型难以协调这两个任务。这指向未来安全对齐需要在多模态融合层面而非仅在文本解码层面进行干预。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统定义并评估双模态隐式毒性，概念贡献清晰且重要
- 实验充分度: ⭐⭐⭐⭐ — 13个模型、317K数据、三级难度、人工验证，实验设计全面；闭源模型仅在子集上测试略有遗憾
- 写作质量: ⭐⭐⭐⭐ — 结构完整清晰，图表丰富，毒性分类体系表述合理
- 价值: ⭐⭐⭐⭐ — 揭示了多模态安全对齐的重要盲区，但缺少解毒方案降低了实际指导价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](../../ICML2025/social_computing/or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)

</div>

<!-- RELATED:END -->
