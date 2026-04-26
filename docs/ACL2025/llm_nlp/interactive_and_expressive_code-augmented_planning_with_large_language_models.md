---
title: >-
  [论文解读] Interactive and Expressive Code-Augmented Planning with Large Language Models
description: >-
  [ACL 2025][LLM/NLP][REPL-Plan] 本文提出REPL-Plan，通过让LLM与扩展的REPL（Read-Eval-Print Loop）交互，实现既能充分利用代码表达力又能动态纠错和处理模糊子问题的自顶向下规划方法，在ALFWorld、WebShop和真实网页导航任务中取得强劲表现。
tags:
  - ACL 2025
  - LLM/NLP
  - REPL-Plan
  - LLM-REPL
  - 代码增强规划
  - 交互式决策
  - 任务分解
---

# Interactive and Expressive Code-Augmented Planning with Large Language Models

**会议**: ACL 2025  
**arXiv**: [2411.13826](https://arxiv.org/abs/2411.13826)  
**领域**: LLM Agent / 代码增强规划  
**关键词**: REPL-Plan, LLM-REPL, 代码增强规划, 交互式决策, 任务分解  

## 一句话总结

本文提出REPL-Plan，通过让LLM与扩展的REPL（Read-Eval-Print Loop）交互，实现既能充分利用代码表达力又能动态纠错和处理模糊子问题的自顶向下规划方法，在ALFWorld、WebShop和真实网页导航任务中取得强劲表现。

## 研究背景与动机

**LLM规划的挑战**：LLM在常识推理和交互决策方面能力强大，但在复杂、长序列规划任务中常犯错——产生幻觉和不正确的短期决策。

**代码增强规划的局限性**：已有工作用代码结构化LLM输出来改善规划，包括使用变量跟踪信息和函数分解子任务，但纯代码方法存在三个固有问题：
   - **模糊子问题**：许多任务需要处理非结构化数据或主观判断（如"购买最匹配用户描述的商品"），代码难以直接解决
   - **自底向上的编码方式**：代码要求先编写子过程函数再处理主任务，需要精确的预先规划，这对一次性生成准确代码要求很高
   - **编码错误**：即使是熟练的人类程序员，一次性写出无bug代码也很困难

**人类开发者的启发**：真实开发者使用REPL（如IPython、Jupyter Notebook）进行交互式编程：逐行输入代码、查看结果、纠正错误。这种交互式方式天然支持动态纠错和探索性开发。

## 方法详解

### 整体框架

REPL-Plan的核心是**LLM-REPL**——REPL的递归扩展版本，LLM在REPL中逐行编写代码，并可通过代码与规划环境交互。

### LLM-REPL设计

LLM-REPL在标准REPL基础上增加了三个关键原语函数：

| 原语 | 功能 |
|------|------|
| `[subtask](args)` | 生成子LLM-REPL处理子任务，若已存在则复用历史执行状态 |
| `get_args()` | 获取父REPL传递的参数 |
| `answer(a)` | 将结果返回给父REPL，执行权回到父级 |

环境交互原语：

| 原语 | 功能 |
|------|------|
| `act(a)` | 在环境中执行动作 |
| `get_obs()` | 获取当前观察文本 |

### 关键设计

1. **递归子任务生成**：当代码中调用未定义函数时（触发`NameError`），自动创建新的子LLM-REPL来处理该子任务。子REPL有独立的变量空间，通过`get_args()`和`answer()`传递上下文。

2. **自顶向下的规划**：与传统自底向上的代码编写不同，LLM先从主任务出发，遇到子问题时再递归创建子REPL解决——这类似于真实开发中的"先设计接口，后实现细节"。

3. **动态纠错**：REPL的逐行执行特性使LLM能看到每步的执行结果和错误信息，可以立即编写新代码来修正，无需重写整个程序。

4. **k-shot REPL池**：维护全局REPL池，存储先前任务和演示中生成的REPL。新任务调用同名REPL时可复用代码/输出历史，实现上下文学习。

### 工作示例

以电商搜索为例，主REPL调用`filter_search(description)`遍历搜索页面，该函数又调用`filter_page(description)`筛选当前页匹配商品，再调用`parse_items()`解析页面元素和`item_matches(item, description)`判断是否匹配——整个过程自顶向下、递归生成。

## 实验关键数据

### ALFWorld（家庭模拟环境）

| 方法 | 成功率(%) | 外部记忆 |
|------|-----------|----------|
| ReAct | 53.7 | 否 |
| ADaPT | 82.1 | 否 |
| THREAD | 95.5 | 否 |
| **REPL-Plan** | **97.0** | 否 |
| Reflexion | 76.1 | 是 |
| RAP | 85.8 | 是 |

REPL-Plan在不使用外部记忆的情况下达到97.0%成功率，超过所有基线。

### WebShop（电商导航）

| 模型 | 设置 | 策略 | 方法 | 成功率(%) | 得分(%) |
|------|------|------|------|-----------|---------|
| GPT-3.5 | k=3 | Top-3 | THREAD | 49 | 76.3 |
| GPT-3.5 | k=3 | Top-3 | REPL-Plan | 47 | 74.2 |
| GPT-4o-mini | k=10 | Top-3 | THREAD | 21 | 42.1 |
| GPT-4o-mini | k=10 | Top-3 | REPL-Plan | 37 | 69.9 |
| GPT-4o-mini | k=10 | Top-20 | **REPL-Plan** | **52** | **77.1** |

在简单设置(k=3)下REPL-Plan与THREAD相当，但在更大搜索空间(k=10)和更复杂策略(Top-20)下优势明显。

### 真实网页任务

| 方法 | 简单任务(%) | 复杂任务(%) |
|------|-------------|-------------|
| ReACT | 86.7 | 17.6 |
| THREAD | 13.3 | 0.0 |
| **REPL-Plan** | **86.7** | **39.6** |

在复杂的循环式真实网页任务中，REPL-Plan显著优于基线（网页观察长4k-20k tokens）。

### 消融实验（WebShop k=3, n=25）

| 消融 | GPT-3.5 SR(%) | GPT-4o-mini SR(%) |
|------|---------------|-------------------|
| 完整模型 | 52 | 44 |
| 有bug的演示 | 52 | 40 |
| 无子任务REPL | 24 | 20 |
| Zero-shot子REPL | 28 | 16 |

- **递归子任务REPL是关键**：移除后性能降半
- **纠错能力强**：注入代码bug后性能几乎不变
- **Zero-shot能力有限**：移除一个子REPL的演示后约半数试验能正确推断

## 亮点与洞察

1. **代码表达力与动态性的统一**：REPL-Plan首次实现了既有完整代码表达力（循环、变量、函数）又能动态纠错和处理模糊问题的规划框架
2. **自顶向下的递归分解**：通过LLM-REPL的递归生成，自然实现了类似"先设计API再实现"的开发模式，避免了自底向上的困难
3. **对幻觉的鲁棒性**：代码控制流能缓解LLM幻觉的影响——即使预测了错误的元素ID，循环结构确保agent继续搜索而非永远卡住
4. **可扩展性**：在长观察（4k-20k tokens）和复杂任务中的优势尤其明显

## 局限性

1. **依赖高质量演示**：Zero-shot推断子REPL的能力有限，当没有demo时性能大幅下降
2. **API调用开销**：每一行代码都需要LLM推理，递归生成多个子REPL会显著增加API调用次数和成本
3. **环境约束**：目前仅在文本环境中测试，能否推广到多模态环境有待验证
4. **子REPL的任务描述质量**：依赖LLM生成子REPL的任务描述，若描述不准确会导致级联失败

## 相关工作

- **LLM Agent**：ReAct (Yao et al., 2022b)、Reflexion (Shinn et al., 2023) 以观察-动作序列为主；THREAD (Schroeder et al., 2024) 用文本分治策略
- **代码增强LLM**：PAL (Gao et al., 2023) 用代码辅助推理；ProgPrompt (Singh et al., 2022) 用程序结构做规划
- **递归任务分解**：ADaPT (Prasad et al., 2024) 进行自适应分解；THREAD 用树状子任务结构

## 评分

⭐⭐⭐⭐ — 设计思路精巧且直觉明确，将人类交互式编程经验（REPL）系统化地融入LLM规划，在真实网页导航任务中展现了独特优势。递归REPL的设计优雅，但API调用成本和演示依赖需关注。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] On the Limit of Language Models as Planning Formalizers](limit_llm_planning_formalizer.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)

<!-- RELATED:END -->
