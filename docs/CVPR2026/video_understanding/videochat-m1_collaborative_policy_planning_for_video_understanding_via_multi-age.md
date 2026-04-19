---
title: >-
  [论文解读] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning
description: >-
  [CVPR2026][视频理解][多智能体系统] 提出VideoChat-M1，用多智能体协作策略规划（CPP）+ 多智能体强化学习（MARL）替代传统固定工具调用策略，让多个策略Agent动态生成、执行和沟通工具调用计划，在8个视频理解基准上取得SOTA，LongVideoBench超Gemini 2.5 Pro 3.6%、超GPT-4o 15.6%。
tags:
  - CVPR2026
  - 视频理解
  - 多智能体系统
  - 多智能体强化学习
  - 协作策略规划
  - 视频问答
  - 工具调用
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning

**会议**: CVPR2026  
**arXiv**: [2511.19524](https://arxiv.org/abs/2511.19524)  
**代码**: 待确认  
**领域**: 视频理解  
**关键词**: 多智能体系统, 多智能体强化学习, 协作策略规划, 视频问答, 工具调用

## 一句话总结

提出VideoChat-M1，用多智能体协作策略规划（CPP）+ 多智能体强化学习（MARL）替代传统固定工具调用策略，让多个策略Agent动态生成、执行和沟通工具调用计划，在8个视频理解基准上取得SOTA，LongVideoBench超Gemini 2.5 Pro 3.6%、超GPT-4o 15.6%。

## 研究背景与动机

**视频理解是核心难题**：多模态大语言模型（MLLM）在短视频上表现优异，但面对长时序、复杂空间结构的视频仍然力不从心

**Agent框架的兴起**：基于Agent的框架通过调用各类工具提取关键视频线索，避免将海量帧直接喂入MLLM，已展现出超越端到端模型的潜力

**现有Agent策略固定且不可学习**：已有多Agent视频理解框架采用静态的、预定义的工具调用规则，无法自适应地发现多样化线索，限制了对复杂视频的感知与推理能力

**单一策略局限性**：单Agent或固定策略无法同时兼顾感知、检索和综合摘要，难以处理不同时间尺度上的丰富线索

**多Agent协作缺乏训练**：现有多Agent系统（如CAMEL、MetaGPT）依赖工程逻辑和固定角色，缺少针对视频多模态任务的联合训练机制

**RL方法局限于文本领域**：已有多Agent RL方法大多局限于单模态文本任务，忽略了视频特有的时序和感知挑战

## 方法详解

### 整体框架：协作策略规划（CPP）

VideoChat-M1包含一组策略Agent $\mathcal{G}=\{\mathcal{G}_i\}$、一组视频感知工具 $\mathcal{T}=\{\mathcal{T}_j\}$、以及共享记忆缓冲区 $\mathcal{M}=\{\mathcal{M}_i\}$。CPP范式包含三个核心阶段：

**阶段一：策略生成（Policy Generation）**
每个Agent根据用户查询 $\mathcal{Q}$ 和可用工具集 $\mathcal{T}$，独立生成各自的工具调用策略：$\mathcal{P}_i = \mathcal{G}_i(\mathcal{Q}, \mathcal{T})$，策略为一个有序的工具调用计划 $\mathcal{P}_i = \{\mathcal{P}_{i,1} \to \mathcal{P}_{i,2} \to \ldots \to \mathcal{P}_{i,N}\}$。

**阶段二：策略执行（Policy Execution）**
Agent按照策略逐步执行工具调用：$\mathcal{A}_{i,n} = \mathcal{P}_{i,n}(\mathcal{V}, \mathcal{T}, \mathcal{A}_{i,n-1})$，每一步根据前一步的中间结果选择对应工具分析视频。

**阶段三：策略沟通（Policy Communication）**
执行每一步后，所有Agent将中间结果存入共享记忆 $\mathcal{M}$。每个Agent参考自身策略和团队中间记忆，决定是否更新后续策略：$\mathcal{P}'_i = \mathcal{G}_i(\mathcal{Q}, \mathcal{T}, \mathcal{M}, \mathcal{P}_i)$。若当前策略仍然最优则保持不变，否则修订后续步骤。

沟通和执行交替进行多轮迭代，每个Agent在执行过程中持续利用团队的中间结果作为历史经验，通过多轮沟通不断修正自身策略。

最终答案的聚合策略因任务而异：多选题通过多数投票决定；开放式问题和时序定位任务由团队中表现最佳的模型（Qwen3-8B）负责汇总。

### 关键设计：多智能体强化学习（MARL）

**策略SFT阶段**：利用GPT-4o和DeepSeek-R1自动标注高质量策略计划数据集，筛选标准为：(1) 能产出正确答案；(2) 可一次执行成功无需修改。用交叉熵损失微调每个Agent，使其掌握基本策略生成能力。

**MARL阶段**：基于GRPO联合优化所有Agent，设计三种奖励信号：

### 损失函数与奖励

- **结果奖励 $\mathcal{R}_{res}$**：正确答案给正奖励，错误答案给负惩罚
- **格式奖励 $\mathcal{R}_{format}$**：输出格式正确（可解析的计划、有效工具调用）给奖励，格式错误给惩罚
- **协作奖励 $\mathcal{R}_{col}$**：用GPT-4o评估每个Agent的中间规划轨迹质量（计划可行性、工具调用恰当性、步骤管理合理性），输出二值奖励（1/0）；超过5次工具调用的轨迹施加强惩罚

总奖励 $\mathcal{R} = \mathcal{R}_{res} + \mathcal{R}_{format} + \mathcal{R}_{col}$，使用GRPO目标函数优化模型参数。训练中采用Agent Dropout，每步随机采样DAG通信拓扑，增强泛化性。

## 实验

### 主要结果

在8个基准、4类任务上取得SOTA，Agent团队总参数37B：

| 任务 | 基准 | VideoChat-M1 | 对比模型 | 提升 |
|------|------|:---:|------|:---:|
| 长视频QA | LongVideoBench | **82.3** | Gemini 2.5 Pro (78.7) | +3.6 |
| 长视频QA | LongVideoBench | **82.3** | GPT-4o (66.7) | +15.6 |
| 视频推理 | Video-Holmes | **60.5** | GPT-4o (42.0) | +18.5 |
| 视频推理 | VideoMMMU | **80.0** | Qwen3-VL-235B (74.7) | +5.3 |
| 空间智能 | VSIBench Avg | **71.9** | Gemini 1.5 Pro (45.4) | +26.5 |
| 时序定位 | Charades-STA | **67.7** | Eagle-2.5-8B (65.9) | +1.8 |

### 效率对比

VideoChat-M1平均仅使用69.9帧（其他模型的12%-18%），推理时间19.8s（其他模型的9%-22%），但性能全面领先。相比GPT-4o使用384帧/153.6s、Gemini 1.5 Pro使用568帧/227.2s，VideoChat-M1通过智能工具调用实现了极其高效的帧采样策略，用不到1/5的计算资源取得更优结果。

### 实现细节

训练使用8×A100 80G GPU，SFT学习率1e-6，MARL学习率1e-7。SFT训练1个epoch（batch size 32），MARL最佳性能在200步时达到（4 rollouts，batch size 8）。Agent团队由Qwen2.5-3B、Qwen2.5-7B、Qwen3-4B、Qwen3-8B四个异构模型组成，总参数约37B。

### 消融实验与关键发现

- **Agent数量**：1→4个Agent性能稳步提升，超过4个后趋于饱和
- **架构多样性关键**：异构Agent组（Qwen2.5-3B/7B + Qwen3-4B/8B）优于同构Agent组，结构冗余降低讨论多样性
- **MARL各组件贡献**：移除协作奖励 $\mathcal{R}_{col}$ 降1分，移除格式奖励类似，移除Agent Dropout降2分（最关键正则化手段）
- **SFT+MARL互补**：单独SFT提升6.6/+6.6，单独MARL提升5.8/+10.9，两者结合达到峰值（60.5/82.3），初始化先验+涌现协作缺一不可
- **LoRA接近全参微调**：LoRA仅更新2%参数，效果仅略低于全参微调（59.4 vs 60.5），提供轻量部署选项
- **投票机制最优**：多数投票（60.5/82.3）> Agent决策（60.2/81.6）> 最高分选择（59.9/81.2）
- **超越闭源Agent团队**：训练后的37B团队大幅超越未训练的4×GPT-4o（+7.8/+9.4）和4×DeepSeek-R1（+8.7/+10.9），证明协作微调注入了零样本推理无法发现的任务特异性协调模式
- **Full finetune vs LoRA**：LoRA仅更新约2%参数即可达到接近全参微调的效果（VideoHolmes 59.4 vs 60.5），为资源受限场景提供了实用的轻量部署方案

## 亮点

- **首个多Agent策略学习框架用于视频理解**：用可学习的协作策略规划替代固定工具调用，是该方向的范式转变
- **CPP三阶段范式设计精巧**：生成→执行→沟通的迭代循环使Agent能动态修正策略，充分利用团队中间信息，比静态分工更灵活
- **效率极高**：仅用69.9帧和19.8s推理时间，远低于GPT-4o等模型，却全面领先
- **37B参数媲美235B**：在VideoMMMU等基准上，4个小模型的协作效果可比Qwen3-VL-235B
- **消融极其充分**：涵盖Agent数量/组成/多样性/奖励组件/训练策略/决策机制等多个维度

## 局限性

- **协作奖励依赖GPT-4o评估**：中间过程的协作奖励用GPT-4o作为外部评估器，引入额外API成本和评估偏差，大规模训练时可拓展性受限
- **工具集范围未充分讨论**：论文未详细说明工具集 $\mathcal{T}$ 的具体组成和可扩展性，新工具的接入方式不明确
- **训练成本较高**：需要先用GPT-4o+DeepSeek-R1标注策略数据做SFT，再做MARL，整体流程复杂且依赖强闭源模型
- **仅在QA类任务上验证**：虽然覆盖4类任务，但均为问答或定位形式，对视频生成、编辑、摘要等任务的泛化性未知
- **Agent间通信成本**：每步执行后所有Agent共享中间结果，通信开销随Agent数和步数增长，超过4个Agent后性能饱和也可能与此相关

## 相关工作

- **单Agent视频工具调用**：VideoAgent、VideoChat-Flash、InternVideo2.5等通过单Agent+检索/搜索工具增强视频理解，但策略固定不可学习
- **多Agent无训练框架**：LVAgent、VCA等采用静态协作和固定角色分工，缺乏自适应性，性能受限于预定义规则
- **Agent+RL训练**：VideoChat-R1/R1.5用RL训练单Agent的推理能力，本文首次将RL扩展到多Agent联合训练，优化Agent间协作
- **多Agent RL（文本领域）**：CAMEL、MetaGPT等局限于纯文本领域的多Agent协作，本文将多Agent RL首次引入视觉-语言多模态任务
- **视频RAG方法**：VideoRAG、ReAgent-V等通过检索增强提升长视频理解，但缺乏可学习的协作机制

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个可训练的多Agent协作策略学习框架用于视频理解，CPP+MARL范式新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 8个基准4类任务，消融覆盖Agent组成/奖励/训练/决策等多维度
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式化描述规范，图示直观
- 价值: ⭐⭐⭐⭐⭐ — 37B多Agent协作超越GPT-4o和Gemini 2.5 Pro，展示小模型协作的巨大潜力

<!-- end of note -->

<!-- RELATED:START -->

## 相关论文

- [A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](dual-agent_reinforcement_learning_for_adaptive_and_cost-aware_visual-inertial_od.md)
- [LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)
- [Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)

<!-- RELATED:END -->
