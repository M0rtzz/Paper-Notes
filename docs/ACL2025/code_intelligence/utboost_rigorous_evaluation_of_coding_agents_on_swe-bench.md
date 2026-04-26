---
title: >-
  [论文解读] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench
description: >-
  [ACL 2025][SWE-Bench] 本文提出 UTBoost 框架，通过基于 LLM 的测试用例生成器 UTGenerator 和改进的解析器来增强 SWE-Bench 的测试用例覆盖率，发现 36 个测试不充分的实例和 345 个被错误标记为通过的补丁，导致 SWE-Bench Lite 排行榜 40.9% 和 SWE-Bench Verified 24.4% 的排名发生变化。
tags:
  - ACL 2025
  - SWE-Bench
  - 测试用例增强
  - 代码生成评估
  - 变形测试
  - LLM编程智能体
---

# UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench

**会议**: ACL 2025  
**arXiv**: [2506.09289](https://arxiv.org/abs/2506.09289)  
**代码**: [github.com/CUHK-Shenzhen-SE/UTBoost](https://github.com/CUHK-Shenzhen-SE/UTBoost)  
**领域**: 其他（软件工程/代码生成评估）  
**关键词**: SWE-Bench, 测试用例增强, 代码生成评估, 变形测试, LLM编程智能体

## 一句话总结

本文提出 UTBoost 框架，通过基于 LLM 的测试用例生成器 UTGenerator 和改进的解析器来增强 SWE-Bench 的测试用例覆盖率，发现 36 个测试不充分的实例和 345 个被错误标记为通过的补丁，导致 SWE-Bench Lite 排行榜 40.9% 和 SWE-Bench Verified 24.4% 的排名发生变化。

## 研究背景与动机

SWE-Bench 是评估代码生成智能体在真实世界 Python 项目上能力的标准基准。它基于 GitHub 的 issue 和对应的 pull request 构建，使用人工编写的测试用例来验证生成的补丁是否解决了问题。

然而，**人工编写的测试用例往往不充分**：生成的补丁可能通过了测试但并未真正解决问题。例如，在 `mwaskom__seaborn-3010` 实例中，issue 要求 `PolyFit` 函数处理缺失数据，但原始测试仅考虑了 x 和 y 同时缺失的情况，未覆盖仅 x 缺失的边界条件。IBM SWE-1.0 的补丁通过了原始测试，但在仅 x 缺失时会报错。

此外，SWE-Bench 的**测试日志解析器存在缺陷**：使用正则表达式提取测试结果时，无法处理跨多行的测试用例日志，导致错误的测试注释。例如在 `django__django-13710` 中，测试名 `test_immutable_content_type` 跨越两行，解析器错误地将 "Regression for #9362" 提取为测试名。

## 方法详解

### 整体框架

UTBoost 系统由三个关键组件构成：

1. **UTGenerator**：基于 LLM 的测试用例生成器
2. **变形测试（Intramorphic Testing）**：构建测试预言机（Test Oracle）
3. **改进的解析器**：修复 SWE-Bench 原始解析器的缺陷

工作流程：
- Step 1：筛选通过原始测试的生成补丁（满足 P(T_orig) = P'(T_orig)）
- Step 2：用 UTGenerator 生成增强测试用例 T_aug
- Step 3：检查增强测试上的变形关系 P(T_aug) = P'(T_aug) 是否成立
- Step 4：若不成立则标记为可疑实例

### 关键设计

**变形测试（Intramorphic Testing）**：
- 白盒自动化测试技术，通过比较原始系统和修改后系统对同一输入的输出来建立测试预言机
- Gold patch 应用到程序 P，生成补丁应用到程序 P'
- 预言关系：如果两个补丁等价地解决了问题，则 P(T) = P'(T) 应该成立
- 若增强测试打破了这一关系，说明原始测试不充分

**UTGenerator：三级定位 + 测试生成**

1. **文件级定位**：

    - 构建代码库的树形结构表示
    - LLM 接收 issue 描述、原始测试补丁和树形结构
    - 输出 Top-N 个最可能需要添加测试的文件

2. **函数/类级定位**：

    - 压缩代码文件，仅保留类和函数的头部
    - LLM 分析压缩格式，定位最可能添加测试的函数或类

3. **行级定位**：

    - 提取具体代码片段
    - LLM 确定添加增强测试用例的具体行范围

4. **测试用例生成**：

    - 使用 x 行的上下文窗口扩展定位行
    - LLM 生成增强测试用例及其依赖

**改进的解析器**：
- 使用队列跟踪相邻日志数据
- 正则表达式精确匹配测试用例名称
- 当测试跨多行时，迭代搜索直到找到正确的测试名
- 修复了原始解析器无法处理的多种边界情况

### 损失函数 / 训练策略

UTGenerator 使用 GPT-4o 作为 LLM 后端。采用多温度采样策略以增加测试用例多样性：
- 温度 0：1 个确定性补丁
- 温度 0.8：20 个补丁
- 温度 0.9：20 个补丁
- 温度 0.99：20 个补丁

文件级定位使用 Top-3，定位阶段温度 0.8，上下文窗口 10 行代码。每个 SWE-Bench 实例平均 API 使用成本 $1.6。

## 实验关键数据

### 主实验

**测试用例不充分的实例发现**：
- SWE-Bench Lite：23 个实例测试不充分（共 300 个）
- SWE-Bench Verified：26 个实例测试不充分（共 500 个）
- 总计 36 个不同实例（两个集合有重叠）

**错误补丁识别**：
- SWE-Bench Lite：599 个通过原始测试的补丁中 170 个 (28.4%) 实为错误补丁
- SWE-Bench Verified：584 个通过补丁中 92 个 (15.7%) 实为错误补丁

**解析器缺陷影响**：
- SWE-Bench Lite：54.7% (164/300) 的实例注释受影响
- SWE-Bench Verified：54.2% (271/500) 的实例注释受影响
- 修正后发现额外 64（Lite）和 79（Verified）个错误补丁

**总计识别错误补丁**：
- SWE-Bench Lite：176 个（增强测试 + 改进解析器）
- SWE-Bench Verified：169 个

### 消融实验

**项目分布分析**：
- django 和 sympy 占错误最多：SWE-Bench Lite 中 84.1% 的错误补丁，Verified 中 82.6%
- 测试不充分实例分布在 12 个项目中的 9 个

**排行榜影响**：
- SWE-Bench Lite：40.9% (18/44) 的排名发生变化
- SWE-Bench Verified：24.4% (11/45) 的排名发生变化
- 典型案例：Amazon-Q-Developer-Agent 从第 1 名（55%）降至与 devlo 并列第 1（53.6%），因其有 7 个错误补丁

### 关键发现

1. **即使 93 名专业开发者人工审查也遗漏了测试不充分问题**：UTBoost 在 SWE-Bench Verified 中发现了 26 个问题实例
2. **解析器缺陷影响面极广**：超过 54% 的实例注释存在错误
3. **测试不充分导致的虚假通过率高达 28.4%**：近三分之一的"通过"补丁实际是错误的
4. **django 和 sympy 是最薄弱的项目**：集中了绝大多数错误

## 亮点与洞察

- **首次系统性解决 SWE-Bench 的测试不充分问题**：之前的工作（Aleithan et al., Chen and Jiang）只是人工发现问题，UTBoost 提供了自动化解决方案
- **首次将变形测试应用于评估开源软件系统**：巧妙地利用 Gold Patch 和生成补丁的等价性建立测试预言机
- **三级定位策略有效处理了大规模代码库**：从文件→函数/类→行逐步缩小范围
- **改进的解析器修复了一个被长期忽视的基础设施缺陷**：影响了超过一半的实例

## 局限与展望

- **依赖至少一个智能体已解决实例**：UTBoost 需要交叉验证 Gold Patch 和生成补丁，无法处理没有智能体解决的实例（当前覆盖 Lite 的 74.6%，Verified 的 81.6%）
- **仅使用 GPT-4o**：集成其他 LLM 可能增加测试多样性
- **架构较简化**：基于 Agentless 的简化版，采用更复杂的代码智能体框架可能生成更多样的测试
- **每实例成本 $1.6**：大规模使用成本不低
- **仅覆盖 Python 项目**：尚未扩展到其他编程语言

## 相关工作与启发

- 与 EvalPlus (Liu et al., 2024) 的对比：EvalPlus 通过类型感知变异为 HumanEval/MBPP 添加测试，但无法处理 SWE-Bench 的多文件/多依赖复杂性
- 对代码生成基准设计的启示：**测试覆盖率与解析器正确性是基准可信度的基础**，不能仅依赖人工审查
- 为未来 SWE-Bench 提交提供了即插即用的增强测试用例

## 评分

- **创新性**：★★★★☆（变形测试应用新颖，但整体方法较为工程导向）
- **实验充分性**：★★★★★（全面覆盖两个基准，人工审查确认结果）
- **实用价值**：★★★★★（直接影响 SWE-Bench 排行榜，已发布代码和数据）
- **写作质量**：★★★★☆（结构清晰，案例详细，但部分内容重复）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2025\] CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)
- [\[ICML 2025\] Training Software Engineering Agents and Verifiers with SWE-Gym](../../ICML2025/code_intelligence/training_software_engineering_agents_and_verifiers_with_swe-gym.md)
- [\[ACL 2025\] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)

<!-- RELATED:END -->
