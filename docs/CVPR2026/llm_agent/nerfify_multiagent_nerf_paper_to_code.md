---
title: >-
  [论文解读] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code
description: >-
  [CVPR 2026][LLM Agent][NeRF] 提出 Nerfify，一个多智能体框架，通过上下文无关文法约束、图思维代码合成和组合式引用恢复，将 NeRF 论文自动转换为可训练的 Nerfstudio 插件代码，在 30 篇论文基准上实现 100% 可执行率，视觉质量与专家实现差距仅 ±0.5 dB PSNR。
tags:
  - CVPR 2026
  - LLM Agent
  - NeRF
  - paper-to-code
  - multi-agent
  - code synthesis
  - context-free grammar
---

# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**会议**: CVPR 2026  
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)  
**代码**: 待开源  
**领域**: LLM Agent  
**关键词**: NeRF, paper-to-code, multi-agent, code synthesis, Nerfstudio

## 一句话总结
提出 Nerfify，一个领域感知的多智能体框架，通过上下文无关文法（CFG）约束、图思维（GoT）代码合成和组合式引用依赖恢复，将 NeRF 论文自动转化为可训练的 Nerfstudio 插件，实现 100% 可执行率，视觉质量与专家实现仅差 ±0.5 dB PSNR。

## 研究背景与动机
**领域现状**：NeRF 自 2020 年发表以来已有超过 1000 篇后续工作，但大多数论文没有公开代码或标准化实现，每篇后续工作都需要数周重新实现前人方法。

**现有痛点**：通用 paper-to-code 系统（Paper2Code、AutoP2C）对 NeRF 几乎完全失败——95% 的情况下无法生成可训练代码。GPT-5 单次生成模式在复杂论文上仅达 26.6% 准确率。NeRF 实现涉及体积渲染、计算机视觉和神经优化的交叉，一个错误的激活函数或射线-球体交叉就可能导致 NaN 梯度或退化解。

**核心矛盾**：通用方法缺乏领域知识，无法处理 NeRF 的隐式依赖链（如"我们采用了 [3] 的蒸馏损失"需要遍历引用、定位公式、翻译为代码并实现 stop-gradient），也无法满足 Nerfstudio 框架的模块组合约束。

**本文目标**：自动将 NeRF 论文转化为可训练、收敛、高质量的标准化 Nerfstudio 代码，从数周人工缩短到几分钟。

**切入角度**：将 Nerfstudio 架构形式化为上下文无关文法（CFG），用领域约束指导 LLM 代码合成，多智能体协作解决依赖链和视觉反馈优化。

**核心 idea**：领域感知（CFG 约束 + 引用依赖恢复 + GoT 合成 + 视觉反馈）让 NeRF paper-to-code 从不可行变为高质量自动化。

## 方法详解

### 整体框架
Nerfify 采用四阶段 pipeline：(1) CFG 形式化与上下文学习——解析论文 PDF 为结构化 markdown 并构建领域知识库 $\mathcal{K}$；(2) 组合式依赖解析——遍历引用图递归获取被引论文中的关键组件；(3) 文法引导的仓库代码生成——GoT 多智能体按拓扑序合成多文件代码；(4) 视觉驱动反馈——训练后渲染图像，通过 PSNR 分析和 VLM 诊断迭代修复。

### 关键设计

1. **上下文无关文法（CFG）约束合成**:

    - 功能：确保生成代码满足 Nerfstudio 的架构不变量和接口契约
    - 核心思路：将 Nerfstudio 的模块组合模式形式化为 CFG，LLM 在生成代码时被文法规则硬约束，保证代码在架构层面的正确性。使用 MinerU 将论文 PDF 转为 markdown，清洗后保留方程、伪代码、架构图和引用关系，与对应 Nerfstudio 实现配对存入知识库 $\mathcal{K}$ 和上下文示例库 $\mathcal{X}$
    - 设计动机：通用代码生成把所有框架一视同仁，不理解 NeRF 的 config→datamanager→field→model→pipeline 耦合链；CFG 编码了框架先验，从源头消除架构级错误

2. **组合式引用依赖恢复**:

    - 功能：自动检索并集成论文引用链中的隐式组件（采样器、编码器、损失函数等）
    - 核心思路：构建引用依赖图 $G' = (V', E')$，通过四步迭代执行多跳检索。(a) 依赖发现：解析目标论文提取引用列表及借用组件；(b) 递归解析：$\text{Dependencies}(c_i) = \{c_i\} \cup \bigcup_{d \in \text{cited}(c_i)} \text{Dependencies}(d)$；(c) 组件提取：提取架构模块、损失函数、训练协议；(d) 终止条件：所有接口契约满足。如 K-Planes 需从 7 篇直接引用和 12 篇传递依赖中提取组件
    - 设计动机：NeRF 论文天然是组合式的，"我们采用 [3] 的 proposal network"这种描述要求系统能自动追溯多层引用并提取精确实现

3. **图思维（GoT）多智能体代码合成**:

    - 功能：按拓扑依赖序生成多文件仓库，验证类型签名、张量形状和循环依赖
    - 核心思路：主合成智能体将论文映射为 Nerfstudio 组件依赖 DAG，分四阶段执行：DAG 构建→接口冻结（按拓扑序建立最小公共 API）→实现（每个节点合成并验证代码）→集成测试（端到端 smoke test + 自动修复）。仓库定义为 $\mathcal{C} = (F, G)$，其中 $G = \text{BuildRepoDAG}(F)$ 为有向无环图，$(f_i, f_j) \in E(G)$ 意味着不存在从 $f_j$ 到 $f_i$ 的路径
    - 设计动机：单体代码生成无法处理多文件间的耦合关系，图思维比 CoT/ToT 更适合仓库级的依赖感知生成

### 视觉驱动反馈
Stage 4 对生成代码执行 3k 次迭代 smoke training，从多视角渲染图像后交给批判智能体。批判智能体含三路分支：(1) 度量分支——计算局部窗口 PSNR/SSIM 图，通过形态学操作定位最大误差区域；(2) 几何分支——跨视图伪影一致性检测，识别 floater 和鬼影；(3) 语义分支——利用 Qwen3 VLM 分析伪影三元组，输出结构化诊断和代码补丁。反馈循环直到无新反馈、达最大迭代或达到论文报告的 PSNR 目标。

## 实验关键数据

### 主实验
Nerfify-Bench 30 篇论文，Set 1（无公开代码论文，对比专家人工实现）：

| 论文 | Paper PSNR/SSIM | 专家实现 PSNR/SSIM | Nerfify PSNR/SSIM |
|------|----------------|-------------------|-------------------|
| KeyNeRF | 25.65/0.89 | 25.70/0.89 | 26.12/0.90 |
| mi-MLP NeRF | 24.70/0.89 | 22.64/0.87 | 22.85/0.87 |
| ERS | 27.85/0.94 | 26.87/0.90 | 27.02/0.90 |
| TVNeRF | 27.44/0.93 | 26.81/0.92 | 27.30/0.92 |

可执行性对比（所有基线均无法生成可训练代码）：

| 指标 | Paper2Code | AutoP2C | GPT-5 | R1 | Nerfify |
|------|-----------|---------|-------|-----|---------|
| 可编译/可训练 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 训练稳定性 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 收敛到论文结果 | ✗ | ✗ | ✗ | ✗ | ✓ |

### 消融实验（新颖性保留 Set 4，Score↑）

| 论文 | Nerfify | GPT-5 | Paper2Code | AutoP2C |
|------|---------|-------|-----------|---------|
| Mip-NeRF | 1.00 | 0.58 | 0.85 | 0.20 |
| BioNeRF | 1.00 | 0.82 | 0.35 | 0.15 |
| TensoRF | 0.98 | 0.72 | 0.12 | 0.28 |
| Tetra-NeRF | 1.00 | 0.58 | 0.22 | 0.08 |
| E-NeRF | 1.00 | 0.60 | 0.48 | 0.05 |

### 关键发现
- 通用 paper-to-code 系统在 95% 的 NeRF 论文上无法生成可训练代码，Nerfify 实现 100% 可执行率
- 视觉质量平均在专家实现 ±0.5 dB PSNR、±0.02 SSIM 范围内
- 已有 Nerfstudio 集成的论文（NeRF、Nerfacto），Nerfify 生成完全相同的代码
- 新颖性保留（正确实现论文核心创新）方面 Nerfify 远超所有基线

## 亮点与洞察
- 将框架形式化为 CFG 从根本上把"理解框架"转化为"遵循文法"，降低了 LLM 的生成难度
- 组合式引用依赖恢复解决了学术论文中关键实现细节隐藏在引用链中的长期难题
- 视觉反馈环路首次将 VLM 引入 NeRF 代码自动调试，三路分支设计兼顾像素级、几何级和语义级诊断
- 实验设计很周到，Set 1 无代码论文排除了 LLM 训练数据泄露的影响

## 局限与展望
- 仅支持 Nerfstudio 框架，3DGS、gsplat 等新范式未覆盖
- CFG 需要人工构建，扩展到新框架需额外工程投入
- 视觉反馈依赖 3k 次迭代训练，计算成本不可忽略
- 论文虽称"分钟级"，但包含 smoke training 的端到端时间可能更长

## 相关工作与启发
- Paper2Code、AutoP2C 验证了通用方法在复杂视觉系统上的天花板，凸显领域感知的必要性
- Graph-of-Thought 提供了比链式/树式推理更灵活的 DAG 结构，适合仓库级代码生成
- 该范式可迁移到其他领域的 paper-to-code（robotics、NLP、medical imaging），核心是设计相应的领域 CFG

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将 CFG 约束 + GoT 合成 + 引用依赖恢复结合的领域 paper-to-code 系统
- 实验充分度: ⭐⭐⭐⭐ 30 篇论文的 benchmark 很全面，四类论文分组设计合理
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，问题动机强，但篇幅较长
- 价值: ⭐⭐⭐⭐⭐ 对 NeRF 社区及可复现研究有重大推动意义
# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**会议**: CVPR 2026  
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)  
**代码**: 待开源  
**领域**: 3D Vision / Code Generation  
**关键词**: NeRF, paper-to-code, multi-agent, code synthesis, Nerfstudio

## 一句话总结
提出 Nerfify，一个领域感知的多智能体框架，通过上下文无关文法约束、图思维代码合成和引用依赖恢复，将 NeRF 论文自动转化为可训练的 Nerfstudio 插件代码，实现 100% 可执行率，视觉质量与专家实现仅差 ±0.5 dB PSNR。

## 研究背景与动机
**领域现状**：NeRF 自 2020 年发表以来已有超过 1000 篇后续工作，但大多数论文没有公开代码或标准化实现。每篇后续工作都需要花费数周重新实现前人方法。

**现有痛点**：通用 paper-to-code 系统（如 Paper2Code、AutoP2C）在 NeRF 代码生成上表现极差，95% 的情况下无法生成可训练代码。GPT-5 等前沿模型在复杂论文上仅达 26.6% 准确率。NeRF 实现涉及体积渲染、计算机视觉和神经优化的交叉，一个错误的激活函数或射线-球面交叉就会导致灾难性失败。

**核心矛盾**：通用代码生成方法缺乏领域知识，无法处理 NeRF 论文的隐式依赖链（如"我们采用了 [3] 的蒸馏损失"需要导航到那篇论文、定位正确公式、转译为代码），也无法满足多文件架构的接口契约和张量形状约束。

**本文目标**：如何自动将 NeRF 论文转化为可训练、高质量的标准化代码，从数周人工实现缩短到几分钟。

**切入角度**：将 Nerfstudio 框架形式化为上下文无关文法（CFG），用领域特定约束指导 LLM 代码合成，并利用多智能体协作解决依赖链解析和视觉反馈迭代。

**核心 idea**：用 CFG 约束 + 图思维合成 + 引用依赖恢复构建领域感知的 NeRF paper-to-code 多智能体系统。

## 方法详解

### 整体框架
Nerfify 采用四阶段 pipeline：(1) CFG 形式化与上下文学习——解析论文为 markdown 并构建领域知识库 $\mathcal{K}$；(2) 组合式依赖解析——遍历引用图获取隐式组件；(3) 文法引导的代码生成——通过图思维（GoT）多智能体按拓扑序合成代码仓库；(4) 视觉驱动反馈——通过 PSNR 分析和 VLM 指导修复视觉伪影。

### 关键设计

1. **上下文无关文法（CFG）约束合成**:

    - 功能：确保生成代码满足 Nerfstudio 架构不变量
    - 核心思路：将 Nerfstudio 的模块组合和接口契约形式化为 CFG，LLM 生成代码时被文法规则约束，保证架构正确性。智能体系统 $\mathcal{A}: (\mathcal{E}(\mathcal{P}); \mathcal{R}) \mapsto \mathcal{C}$，其中资源 $\mathcal{R} = (\mathcal{K}, \mathcal{W}, \mathcal{X})$ 分别为领域知识库、Web 资源和代码模板。论文提取函数 $\mathcal{E}(\mathcal{P}) = \langle T(\mathcal{P}), I(\mathcal{P}), Q(\mathcal{P}), B(\mathcal{P}) \rangle$ 涵盖文本、图像、公式和引用
    - 设计动机：通用代码生成缺乏对 NeRF 框架结构的理解；CFG 编码了领域先验知识，把"理解框架"转化为"遵循文法"

2. **组合式引用依赖恢复**:

    - 功能：自动获取论文引用但未详细说明的组件实现
    - 核心思路：构建引用依赖图 $G' = (V', E')$，通过四步递归获取：(a) 依赖发现——解析引用并识别借用组件；(b) 递归解析——$\text{Dependencies}(c_i) = \{c_i\} \cup \bigcup_{d \in \text{cited}(c_i)} \text{Dependencies}(d)$；(c) 组件提取——识别架构模块、损失函数、训练协议；(d) 终止——所有接口契约均满足。例如 K-Planes 需要 7 篇直接依赖和 12 篇传递依赖
    - 设计动机：NeRF 论文本质上是组合式的，"我们采用了 [3] 的蒸馏损失"这类描述在NeRF中极为常见，仅检索目标论文无法获取完整实现

3. **图思维（GoT）多智能体代码合成**:

    - 功能：按拓扑依赖顺序生成多文件代码仓库，确保可编译和可训练
    - 核心思路：仓库 $\mathcal{C} = (F, G)$，其中 $G = \text{BuildRepoDAG}(F)$ 为有向无环图。主合成智能体协调专门的文件智能体，分四阶段：(i) DAG 构建——映射到 Nerfstudio 组件依赖图；(ii) 接口冻结——按拓扑序建立最小 API 契约；(iii) 实现——各节点合成经类型签名和张量形状验证的代码；(iv) 集成测试——端到端 smoke test 并自动修复
    - 设计动机：单体代码生成无法处理 NeRF pipeline 中配置→数据管理器→场→模型→训练管线的紧密耦合，需要图级别的依赖管理

### 视觉驱动反馈
生成可训练代码后进行 3k 次迭代 smoke training，批判智能体通过三个分支分析渲染图像：(1) 度量分支——计算局部窗口 PSNR/SSIM 图，定位最高误差区域；(2) 几何分支——跨视图伪影一致性检测，识别 floater 和鬼影；(3) 语义分支——利用 Qwen3 VLM 分析伪影并输出结构化诊断与候选补丁。反馈循环持续到无更多反馈、达到最大迭代或达到论文 PSNR 目标。

## 实验关键数据

### 主实验
Nerfify-Bench Set 1（无公开代码论文，对比专家实现）：

| 论文 | Paper报告 PSNR/SSIM | 专家实现 PSNR/SSIM | Nerfify PSNR/SSIM |
|------|---------------------|-------------------|-------------------|
| KeyNeRF | 25.65/0.89 | 25.70/0.89 | 26.12/0.90 |
| mi-MLP NeRF | 24.70/0.89 | 22.64/0.87 | 22.85/0.87 |
| ERS | 27.85/0.94 | 26.87/0.90 | 27.02/0.90 |
| TVNeRF | 27.44/0.93 | 26.81/0.92 | 27.30/0.92 |

可执行性对比（所有基线均无法生成可训练代码）：

| 指标 | Paper2Code | AutoP2C | GPT-5 | R1 | Nerfify |
|------|-----------|---------|-------|-----|---------|
| 编译/可训练 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 训练稳定性 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 收敛到论文结果 | ✗ | ✗ | ✗ | ✗ | ✓ |

### 消融实验（新颖性保留 Set 4 代表结果）

| 论文 | Nerfify Score | GPT-5 Score | Paper2Code Score |
|------|--------------|-------------|-----------------|
| Mip-NeRF | 1.00 | 0.58 | 0.85 |
| BioNeRF | 1.00 | 0.82 | 0.35 |
| TensoRF | 0.98 | 0.72 | 0.12 |
| Tetra-NeRF | 1.00 | 0.58 | 0.22 |
| E-NeRF | 1.00 | 0.60 | 0.48 |

### 关键发现
- 通用 paper-to-code 系统在 95% 的 NeRF 论文上无法生成可训练代码
- Nerfify 在全部 30 篇论文上实现 100% 可执行率和训练收敛
- 视觉质量与专家实现在 ±0.5 dB PSNR 和 ±0.02 SSIM 范围内
- 已有 Nerfstudio 集成的论文（如 Vanilla NeRF、Nerfacto），Nerfify 生成完全相同的代码

## 亮点与洞察
- 将框架形式化为 CFG 是非常精巧的思路，把"理解框架"问题转化为"遵循文法"问题，大幅降低 LLM 生成难度
- 组合式引用依赖恢复解决了一个长期被忽视的问题：学术论文中大量关键实现细节隐藏在引用链中
- 视觉反馈环路首次将 VLM 用于 NeRF 代码调试，实现了从"能跑"到"跑得好"的质量提升

## 局限与展望
- 仅针对 Nerfstudio 框架，3DGS 等其他范式未覆盖
- CFG 需要人工构建和维护，扩展到新框架需要额外工程工作
- 视觉反馈仍需 3k 次迭代训练，计算开销较大
- Set 2/3 结果中 LLM 可能在预训练中见过已有代码库，存在数据泄露风险

## 相关工作与启发
- Paper2Code、AutoP2C 证明了通用方法的局限性，领域感知是关键
- Graph-of-Thought 扩展了 CoT/ToT 推理结构到代码生成领域
- 方法论可迁移到其他领域的 paper-to-code（robotics、NLP 等），核心在于设计领域 CFG

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将 CFG + GoT + 引用依赖恢复结合的领域 paper-to-code 系统，开创性工作
- 实验充分度: ⭐⭐⭐⭐ 30 篇论文的大规模 benchmark，与多种基线和人类专家充分对比
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，四阶段框架层次分明
- 价值: ⭐⭐⭐⭐⭐ 对 NeRF 社区和更广泛的可复现研究有重大推动意义
# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**会议**: CVPR 2026  
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)  
**代码**: 即将公开  
**领域**: 3D视觉 / 代码生成  
**关键词**: NeRF, paper-to-code, multi-agent, code synthesis, context-free grammar

## 一句话总结
提出 Nerfify，一个多智能体框架，通过上下文无关文法约束、图思维代码合成和组合式引用恢复，将 NeRF 论文自动转换为可训练的 Nerfstudio 插件代码，在 30 篇论文基准上实现 100% 可执行率，视觉质量与专家实现差距仅 ±0.5 dB PSNR。

## 研究背景与动机
**领域现状**：NeRF 领域自 2020 年以来已有超过 1000 篇后续论文，但大多数缺少公开代码或标准化实现，每项后续工作都需要大量人力重新实现已有方法。

**现有痛点**：通用的 paper-to-code 系统（如 Paper2Code、AutoP2C）在 NeRF 领域严重失败——当前最佳系统 O1 在复杂论文上仅达 26.6% 准确率，且基本无法生成可训练的 NeRF 代码。GPT-5 等前沿模型也只能生成语法正确但无法收敛的代码。

**核心矛盾**：NeRF 实现要求跨体积渲染、计算机视觉和神经优化的专业知识，一个错误的激活函数或不正确的射线-球体交叉就会导致从 NaN 梯度到退化解的各种故障。更困难的是，现代 NeRF 论文具有深层的引用依赖——例如一句 "we adopt the distortion loss from [3]" 就需要追溯多篇论文提取正确实现。

**本文目标**：如何自动将 NeRF 研究论文转化为可训练、可收敛、且匹配专家实现视觉质量的标准化代码。

**切入角度**：用领域特定的多智能体框架替代通用 paper-to-code 方法，将 Nerfstudio 架构形式化为上下文无关文法来约束代码生成。

**核心 idea**：通过将 Nerfstudio 架构编码为 CFG 约束 LLM 合成、图思维拓扑有序生成多文件仓库、组合式引用恢复自动追溯依赖论文，实现 NeRF 论文到可训练代码的可靠转换。

## 方法详解

### 整体框架
Nerfify 分四个阶段将 NeRF 论文转化为代码：(1) CFG 形式化与上下文学习——将 Nerfstudio 架构形式化为 CFG，构建知识库 $\mathcal{K}$；(2) 组合式依赖解析——遍历引用图谱恢复隐含依赖；(3) 语法引导的仓库生成——通过 GoT 方法按拓扑序生成多文件仓库；(4) 视觉驱动反馈——通过训练运行的视觉分析迭代改进实现质量。

### 关键设计

1. **上下文无关文法 (CFG) 约束合成**:

    - 功能：确保生成的代码满足 Nerfstudio 的架构不变量
    - 核心思路：将 Nerfstudio 的模块组合和接口规约形式化为 CFG，即仓库 $\mathcal{C} = (F, G)$，其中 $F = \{f_1, f_2, \ldots, f_n\}$ 为文件集，$G = \text{BuildRepoDAG}(F)$ 为有向无环依赖图。LLM 在 CFG 约束下生成代码，保证编译正确性
    - 设计动机：通用 LLM 生成的代码虽语法正确，但缺乏领域知识导致模块接线错误、数学公式实现不正确

2. **图思维 (GoT) 代码合成**:

    - 功能：协调多个专用智能体按拓扑序生成多文件仓库
    - 核心思路：分四阶段——(1) DAG 构建将论文映射到 Nerfstudio 组件依赖；(2) 接口冻结按拓扑序建立 API 规约；(3) 实现阶段每个节点生成经验证的代码（检查类型签名、tensor 形状）；(4) 集成测试运行烟雾测试并自动修复
    - 设计动机：NeRF 管线中文件间存在紧密耦合（配置→数据管理器→场→模型→训练管线），单体生成容易导致接口不一致

3. **组合式引用恢复**:

    - 功能：自动遍历引用图谱检索论文间的隐含依赖组件
    - 核心思路：构建引用依赖图 $G' = (V', E')$，对目标论文执行迭代多跳检索——依赖发现→递归解析→组件提取→终止判断。例如 K-Planes 需从 7 篇直接引用和 12 篇传递依赖中提取 proposal network、hash encoder、VM 分解等组件
    - 设计动机：NeRF 论文本质上是组合性的，一篇论文可能隐含依赖数十篇论文的具体技术组件

### 损失函数 / 训练策略
第四阶段视觉反馈包含三个分析分支：
- **指标分支**：计算局部窗口 PSNR/SSIM 图，通过形态学操作定位最高误差区域
- **几何分支**：实现跨视图伪影共识 (Cross-View Artifact Consensus)，标记视图不一致的浮体和鬼影
- **语义分支**：利用 Qwen3 VLM 分析伪影三元组，输出结构化诊断和候选补丁

迭代精修直至：(1) 无更多反馈，(2) 达到最大迭代次数，或 (3) 达到论文报告的 PSNR 目标。

## 实验关键数据

### 主实验

| 论文 | 数据集 | 专家 PSNR↑ | Nerfify PSNR↑ | 专家 SSIM↑ | Nerfify SSIM↑ |
|------|--------|-----------|--------------|-----------|-------------|
| KeyNeRF | Blender | 25.70 | 26.12 | 0.89 | 0.90 |
| mi-MLP NeRF | Blender | 22.64 | 22.85 | 0.87 | 0.87 |
| ERS | DTU | 26.87 | 27.02 | 0.90 | 0.90 |
| TVNeRF | Blender | 26.81 | 27.30 | 0.92 | 0.92 |

所有基线 (Paper2Code, AutoP2C, GPT-5, R1) 均**无法生成可训练代码**。

### 消融实验

| 配置 | 可执行率 | 关键发现 |
|------|---------|---------|
| Nerfify (完整) | 100% | 与专家实现差距 ±0.5 dB PSNR |
| Paper2Code | 5% | 可编译但不可训练 |
| AutoP2C | 0% | 导入无法解析 |
| GPT-5 | 0% | 可编译但训练不收敛 |
| 无引用恢复 | 部分 | 缺失关键依赖组件 |
| 无视觉反馈 | 100% | 性能差距更大 |

### 关键发现
- Nerfify 在从未实现过的论文上（Set 1）也能达到与专家实现相当的视觉质量
- 对已有 Nerfstudio 实现的论文，Nerfify 生成的代码与官方仓库完全一致
- K-Planes 的引用依赖图涉及 7 篇直接依赖和总共 12 篇论文的传递依赖

## 亮点与洞察
- 首次证明领域感知的多智能体框架能可靠地将复杂视觉论文转化为可训练代码
- CFG 约束是关键——将框架的架构知识编码为形式化文法，从构造上保证代码正确性
- 组合式引用恢复解决了 NeRF 论文中普遍存在的隐含依赖问题
- 实现时间从数周缩短到数分钟，民主化了 NeRF 研究的可重现性

## 局限与展望
- 目前仅针对 Nerfstudio 框架，扩展到其他框架需要重新定义 CFG
- 依赖高质量的 PDF 解析工具（MinerU），论文中的数学公式或图表解析错误可能传播
- 视觉反馈需要实际训练 3k 迭代，计算开销非零
- 对于完全原创的、不基于已有组件的 NeRF 方法，引用恢复的帮助有限

## 相关工作与启发
- **Paper2Code/AutoP2C**：通用 paper-to-code 系统，缺乏领域特定约束
- **Scene Language**：类似地使用 CFG 约束视觉程序合成
- **Graph of Thoughts**：将推理泛化为有向图，Nerfify 将其应用于代码生成
- 启发：任何具有标准化框架的领域（如 MMDetection、HuggingFace Transformers）都可以用类似方法构建 paper-to-code 系统

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次实现 NeRF 论文到可训练代码的自动转换，CFG+GoT+引用恢复的组合独创
- 实验充分度: ⭐⭐⭐⭐ 30 篇论文基准，包含从未实现的论文，对比充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，四阶段分解逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 极具实用价值，有望加速整个 NeRF 社区的可重现性研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_mllm_agent_3d_reasoning_gaussian.md)
- [\[CVPR 2026\] EpiAgent: An Agent-Centric System for Ancient Inscription Restoration](epiagent_agent_centric_system_for_ancient_inscription_restoration.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)

</div>

<!-- RELATED:END -->
