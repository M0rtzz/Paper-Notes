---
title: >-
  [论文解读] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search
description: >-
  [CVPR 2026][LLM Agent][长视频理解] HAVEN 提出音视频实体凝聚 + 层次索引 + Agent搜索的统一框架，通过说话人身份作为跨模态一致性信号，构建全局-场景-片段-实体四级层次数据库，在LVBench上达到84.1%整体准确率的SOTA。
tags:
  - CVPR 2026
  - LLM Agent
  - 长视频理解
  - 层次索引
  - 实体一致性
  - Agent搜索
  - 音视频融合
---

# HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search

**会议**: CVPR 2026  
**arXiv**: [2601.13719](https://arxiv.org/abs/2601.13719)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: 长视频理解, 层次索引, 实体一致性, Agent搜索, 音视频融合

## 一句话总结
HAVEN 提出音视频实体凝聚 + 层次索引 + Agent搜索的统一框架，通过说话人身份作为跨模态一致性信号，构建全局-场景-片段-实体四级层次数据库，在LVBench上达到84.1%整体准确率的SOTA。

## 研究背景与动机
1. **领域现状**：长视频理解是VLM面临的重大挑战，现有方案（RAG、Agent框架）在处理小时级视频时仍存在严重不足。
2. **现有痛点**：（i）基于朴素分块的RAG导致信息碎片化和全局连贯性丧失；（ii）缺乏层次化视频表示，Agent只能做低效的多轮检索来恢复跨片段连续性。
3. **核心矛盾**：长视频中的事件跨越长时间跨度和多场景演变，局部片段的描述无法捕捉全局叙事结构和远程实体关联。
4. **本文目标**：从碎片化检索转向连贯的结构化理解——通过离线构建层次化数据库+在线Agent自适应搜索。
5. **切入角度**：利用说话人身份作为跨模态的长程一致性信号（即使视觉线索不可靠时仍有效），构建稳健的实体表示。
6. **核心idea**：音视频实体凝聚（通过说话人身份整合碎片化观察）+ 四级层次数据库 + 目标驱动的多粒度Agent搜索。

## 方法详解

### 整体框架
离线：构建四级层次数据库 $\mathcal{D} = \{\tilde{\mathcal{C}}, \tilde{\mathcal{E}}, \tilde{\mathcal{S}}, \tilde{\mathcal{G}}\}$（片段→实体→场景→全局）。在线：Agent以全局摘要初始化，通过think-act-observe循环在层次数据库中自适应搜索和推理。

### 关键设计

1. **音视频实体凝聚**:
    - 功能：跨时间和模态整合碎片化的实体观察为一致的规范实体
    - 核心思路：对每个片段提取音频标注（WhisperX说话人分割+ASR）和视觉描述（VLM生成），构建片段表示 $C_i^t = [P_i'; T_i; V_i]$。实体整合分两步——（1）嵌入聚类：将实体描述编码后聚类形成候选组；（2）LLM规范化：验证每个聚类，产生规范实体或拆分。关键：当多个片段共享相同说话人标签时，优先合并对应的角色实体，即使视觉描述因遮挡/视角变化而不同。
    - 设计动机：说话人身份是比视觉外观更稳定的长程线索——遮挡、镜头切换、光照变化都不影响声音身份。这是一个被严重忽视但强大的一致性信号。

2. **四级层次数据库**:
    - 功能：多粒度组织视频内容，支持不同层次的查询
    - 核心思路：（1）片段级 $\tilde{\mathcal{C}}$：每30秒一个片段，包含文本+视觉嵌入；（2）实体级 $\tilde{\mathcal{E}}$：规范实体及其在每个关联片段中的聚焦重描述；（3）场景级 $\tilde{\mathcal{S}}$：由LLM自适应分组语义连续的片段并生成场景摘要；（4）全局级 $\tilde{\mathcal{G}}$：从场景摘要生成的总体概述。
    - 设计动机：不同类型的查询需要不同粒度的信息——"视频讲什么？"需要全局层，"12分钟发生了什么？"需要片段层，"Sarah的表情怎么变化？"需要实体层。

3. **多粒度Agent搜索**:
    - 功能：目标驱动地在层次数据库中导航和推理
    - 核心思路：配备5类工具——全局场景浏览 $T_{\text{scene}}$、片段描述搜索 $T_{\text{caption}}$、片段视觉搜索 $T_{\text{visual}}$、实体搜索 $T_{\text{entity}}$、定向检查 $T_{\text{inspect}}$（含文本和视觉两种模式）。Agent以全局摘要初始化，通过think-act-observe循环迭代：选择工具→执行查询→收集证据→推理→回答。
    - 设计动机：不同查询需要从不同层级入手。Agent可以自主决定最有效的搜索路径，如先粗后细或直接定位实体。

### 损失函数 / 训练策略
离线构建数据库无需训练。Agent搜索使用预训练的推理LLM，无额外训练。

## 实验关键数据

### 主实验

| 方法 | LVBench Overall | LVBench Reasoning | EgoSchema | 说明 |
|------|----------------|-------------------|-----------|------|
| HAVEN (2fps) | **84.1** | **80.1** | - | SOTA |
| DVD w. subtitle | 76.0 | 68.7 | - | 之前最优Agent |
| OpenAI o3 | 57.1 | 50.8 | 63.2 | 闭源模型 |
| GPT-4o | 48.9 | 50.3 | 70.4 | 闭源模型 |

### 消融实验

| 配置 | Overall | 说明 |
|------|---------|------|
| Full HAVEN | 84.1 | 完整框架 |
| w/o 说话人身份 | 下降 | 实体整合质量降低 |
| w/o 层次索引 | 显著下降 | 退化为平坦RAG |
| w/o 多粒度工具 | 下降 | 搜索效率降低 |

### 关键发现
- HAVEN在推理类别上表现尤为突出（80.1%），说明层次化结构对复杂推理特别有帮助。
- 说话人身份在长视频实体整合中是不可替代的线索。
- 与DVD相比，HAVEN在所有子类别上都有提升，且需要的搜索迭代更少。

## 亮点与洞察
- **说话人身份作为实体凝聚的"胶水"**是一个被严重忽视但非常有效的创新。
- **四级层次架构**的设计符合人类理解长视频的认知模式（先整体后细节）。
- 离线构建+在线搜索的架构使得重复查询不需要重新处理视频。

## 局限与展望
- 离线构建层次数据库本身需要一定计算成本（多次LLM调用）。
- 依赖WhisperX的说话人分割质量，对非对话类视频效果有限。
- 片段固定长度（30秒）可能不是所有视频类型的最优划分。

## 相关工作与启发
- **vs DVD**: DVD使用简单的片段描述+全局实体注册，缺少层次结构。HAVEN的四级层次提供了更高效的导航。
- **vs VideoRAG**: 基于碎片化片段检索，缺乏全局连贯性。HAVEN通过层次索引保持了叙事结构。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 音视频实体凝聚和四级层次索引都是创新设计
- 实验充分度: ⭐⭐⭐⭐ LVBench SOTA + 多基准验证
- 写作质量: ⭐⭐⭐⭐⭐ 框架清晰，方法描述系统化
- 价值: ⭐⭐⭐⭐⭐ 长视频理解领域的里程碑式工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)

</div>

<!-- RELATED:END -->
