---
title: >-
  [论文解读] DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services
description: >-
  [LLM/NLP] 提出 DiSCo，一个端-云协同的 LLM 推理调度器，通过成本感知的请求分发和 token 级迁移机制，在成本约束下优化用户的首 token 延迟 (TTFT) 和 token 间延迟 (TBT)。
tags:
  - LLM/NLP
---

# DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services

| 项目 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2502.11417](https://arxiv.org/abs/2502.11417) |
| 代码 | 未开源 |
| 领域 | LLM 推理系统 / 端云协同 |
| 关键词 | Device-Server Collaboration, QoE, TTFT, TBT, Token Migration, LLM Serving |

## 一句话总结

提出 DiSCo，一个端-云协同的 LLM 推理调度器，通过成本感知的请求分发和 token 级迁移机制，在成本约束下优化用户的首 token 延迟 (TTFT) 和 token 间延迟 (TBT)。

## 研究背景与动机

**研究问题：** LLM 文本流式服务面临严峻的服务质量 (QoE) 和成本挑战。TTFT（首 token 延迟）和 TBT（token 间延迟）是实时交互的关键指标，但现有部署方式难以同时满足。

**现有方法的不足：** (1) 云端部署成本高昂，受请求排队、批处理竞争和网络延迟影响，TTFT 抖动严重（GPT-4-mini 在高负载时从 0.3s 飙升到数秒）；(2) 端侧部署受限于设备资源，长 prompt 的 prefill 慢且能耗高（iPhone 运行 7B 模型仅能持续不到 2 小时）。

**核心动机：** 观察到云端 TTFT 不可预测但与 prompt 长度弱相关，而端侧 TTFT 可预测且与 prompt 长度线性相关；两种部署的 token 生成速度均超过人类消费速率。利用这种互补特性进行端云协同调度。

## 方法详解

### 整体框架

DiSCo 作为中间件，包含两个核心控制器：**分发控制器 (Dispatch Controller)** 决定请求初始执行端点，**迁移控制器 (Migration Controller)** 在生成过程中动态切换执行端点。两者共同在成本约束下优化 TTFT 和 TBT。

### 关键设计

1. **统一成本模型与感知分发策略：** 通过动态汇率 $\lambda$ 统一云端货币成本和端侧能耗成本。在设备受限场景下，采用等待时间策略：先尝试云端执行，等待 $w(l)$ 后再启动端侧推理，分两阶段分配预算（尾部保护 + 平均优化）。在服务器受限场景下，根据 prompt 长度阈值 $l_{th}$ 路由——短 prompt 发送到端侧节省服务器预算，长 prompt 双端并行取最快结果。

2. **Token 级迁移框架：** 利用 token 生成速度 $r_g$ 与人类消费速率 $r_c$ 之间的差值构建 token 缓冲区 $B = r_c \times t_m$。当缓冲区积累足够 token 以覆盖迁移开销时触发迁移，源端点停止生成，目标端点无缝接管。迁移仅在预期成本节省超过迁移开销时执行：$C_{migration} = \Delta c_{decode} \cdot l_{remaining} > \text{Overhead}_{migration}$。

3. **高效 Token 传输：** 当端点共享词表时传输 token ID 而非完整文本，数据量减少 35-54%；不同词表时先转文本再重新 tokenize。避免传输中间状态（如 KV cache），因端点常使用不同架构。

### 损失函数/优化目标

优化目标为在成本约束 $\mathbb{E}[I_d(l)l] \leq b \cdot \mathbb{E}[l]$ 或 $\mathbb{E}[I_s(l)l] \leq b \cdot \mathbb{E}[l]$ 下，最小化均值和尾部 TTFT，同时维持稳定的 TBT。

## 实验

### 主实验结果

在四种商业 LLM 服务（GPT-4o-mini、DeepSeek-V2.5、Cohere Command、LLaMA-3-70b）和三种端侧配置上评估：

| 平台/模型 | 约束 | 尾部 TTFT 降低 (Pixel 7 Pro B-1.1B) | 尾部 TTFT 降低 (Xiaomi 14 Q-0.5B) |
|-----------|------|--------------------------------------|-------------------------------------|
| GPT | Server | 23.85% | 44.04% |
| GPT | Device | 26.39% | 16.32% |
| LLaMA | Server | 11.08% | 26.29% |
| LLaMA | Device | 35.67% | 21.29% |
| Command | Server | 47.93% | 52.23% |
| Command | Device | 34.78% | 24.42% |

### 消融实验

| 维度 | 结论 |
|------|------|
| 迁移机制 | 设备受限场景成本降低最高 72.7%，服务器受限最高 83.6% |
| 请求到达间隔 | 在 DiffusionDB 真实工作负载模式下优势持续 |
| 迁移对生成质量 | 三个 LLM 评委 (GPT-4o, Gemini, Qwen) 评估显示质量一致保持 |
| 可扩展性 | DiSCo-S 在 100K 样本上仅需 9.08ms，DiSCo-D 需 14.86ms |

### 关键发现

- DiSCo 将均值 TTFT 降低 6-78%，尾部 TTFT 降低 11-52%；迁移机制在保持可比 QoE 的同时最多节省 84% 服务成本。
- 迁移过程中仅延迟少量 token（平均 3-17 个），相比数百上千的生成长度可忽略不计，TBT P99 不受影响。
- 端侧 TTFT 和 TBT 的稳定性显著优于云端，为协同策略提供了可靠的预测基础。

## 亮点

- 首次提出端-云协同的 LLM 推理调度范式，而非简单的路由分流。
- Token 级迁移机制利用生成-消费速度差实现无感知切换，设计巧妙。
- 基于真实商业 LLM 服务（GPT、DeepSeek 等）的大量实测数据支撑结论。

## 局限性

- 聚焦于端侧 LLM 已达到足够精度的应用场景（如聊天、翻译），不适用于复杂推理任务。
- 设备能耗使用基于 FLOPs 的线性模型，实际能耗受电池状态、温度等因素影响更复杂。
- 仅考虑单设备场景，多设备协同带来的协调开销和资源分配问题未探讨。
- 未考虑隐私保护问题，假设用户接受数据在端云之间传输。

## 相关工作

- **LLM 端云协同：** EdgeShard (Zhang et al., 2024) 和 WDMoE (Xue et al., 2024) 将模型分片跨端点部署；LLMCad (Xu et al., 2023) 用端侧模型降低服务器成本。
- **LLM 路由：** Ong et al. (2024) 和 Ding et al. (2024) 基于请求复杂度路由到不同模型，但不优化 token 交付指标。
- **LLM 推理优化：** vLLM (Kwon et al., 2023) 和 Sarathi-Serve (Agrawal et al., 2024) 优化服务端吞吐-延迟权衡。

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 实用性 | 9 |
| 实验充分度 | 8 |
| 写作质量 | 7 |
| 总体评分 | 8 |

<!-- RELATED:START -->

## 相关论文

- [EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)

<!-- RELATED:END -->
