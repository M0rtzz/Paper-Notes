---
title: >-
  [论文解读] PerfGuard: A Performance-Aware Agent for Visual Content Generation
description: >-
  [ICLR 2026][LLM Agent][LLM Agent] 提出PerfGuard——面向视觉内容生成的性能感知Agent框架：用多维评分矩阵替代文本描述建模工具性能边界(PASM)→自适应偏好更新(APU)动态校准理论排名与实际执行的偏差→能力对齐规划优化(CAPO)引导Planner生成与工具能力匹配的子任务，在图像生成和编辑任务上全面超越GenArtist/T2I-Copilot等SOTA方法。
tags:
  - ICLR 2026
  - LLM Agent
  - 工具选择
  - 性能边界建模
  - 视觉生成
  - AIGC
  - 偏好优化
---

# PerfGuard: A Performance-Aware Agent for Visual Content Generation

**会议**: ICLR 2026  
**arXiv**: [2601.22571](https://arxiv.org/abs/2601.22571)  
**代码**: [GitHub](https://github.com/FelixChan9527/PerfGuard)  
**领域**: AI Agent/视觉内容生成  
**关键词**: LLM Agent, 工具选择, 性能边界建模, 视觉生成, AIGC, 偏好优化

## 一句话总结
提出PerfGuard——面向视觉内容生成的性能感知Agent框架：用多维评分矩阵替代文本描述建模工具性能边界(PASM)→自适应偏好更新(APU)动态校准理论排名与实际执行的偏差→能力对齐规划优化(CAPO)引导Planner生成与工具能力匹配的子任务，在图像生成和编辑任务上全面超越GenArtist/T2I-Copilot等SOTA方法。

## 研究背景与动机

**领域现状**：LLM驱动的Agent已能通过推理和工具调用实现自动任务处理，在视觉内容生成(AIGC)领域涌现出CompAgent、GenArtist等多工具协调系统。

**理想化假设的问题**：现有研究普遍假设"工具调用总是成功的"，缺乏对工具实际执行成功率的系统评估→工具选择的不确定性直接影响Agent规划和决策的整体准确性。

**文本描述的局限**：当前系统依赖通用文本描述定义工具能力(如"能生成与文本语义对齐的图像")→无法区分不同模型在细粒度维度上的性能差异→无法支持精确的工具匹配。

**性能边界缺失**：以文生图为例，FLUX/SD3/DALL·E3在颜色、形状、纹理、空间关系等维度上性能差异显著→但Agent无法感知这些差异→导致规划和执行中引入不确定性。

**静态评估的不足**：即使有基准测试分数，预设的性能边界可能与实际任务执行结果存在偏差→需要根据真实使用反馈动态调整。

**规划与工具的脱节**：现有方法的任务规划过程未考虑工具的实际性能能力→Planner可能生成工具难以高质量完成的子任务→需要将性能感知融入规划过程。

## 方法详解

### 整体架构：四角色Agent系统

PerfGuard基于标准化Agent系统，包含四个核心角色：
- **Analyst**: 解析多模态输入→生成任务摘要 $\tau^*$、目标图像语义 $s^*$、评估目标 $g$
- **Planner**: 利用 $\tau^*$, $s^*$ 和工具性能画像 $\mathcal{B}$ 分解为子任务 $u_t$
- **Worker**: 从工具库选择合适工具执行子任务→生成图像输出 $o_t$
- **Self-Evaluator**: 多维度评估 $o_t$ 与目标 $g$ 的对齐度→反馈用于迭代优化

### 核心机制一：Performance-Aware Selection Modeling (PASM)

**工具性能边界定义**：构建多维评分系统→
- 图像生成工具：基于T2I-CompBench评估7个维度（颜色、形状、纹理、2D空间、3D空间、非空间语义、数量）
- 图像编辑工具：基于ImgEdit-Bench评估7个维度（添加、移除、替换、属性变更、运动变化、风格迁移、背景替换）

**性能驱动选择**：Worker根据子任务特征生成偏好权重向量，与归一化性能矩阵相乘得到工具适合度排名：

$$\mathcal{W}_{task} = \pi_{\text{Worker}}(u_t, \mathcal{B}, \mathcal{D})$$

$$S_{tools} = \mathcal{W}_{task} \cdot \text{Normalize}(M_p)^\top$$

$$\mathcal{R} = \text{argsort}(S_{tools}, \text{descending})$$

其中 $M_p \in \mathbb{R}^{d \times l}$ 是工具性能边界矩阵（$d$维度 × $l$工具），$\mathcal{W}_{task} \in \mathbb{R}^{1 \times d}$ 是任务偏好权重。

### 核心机制二：Adaptive Preference Updating (APU)

引入探索-利用策略：选取top-m高分工具 + 随机采样n个工具→执行后对比理论排名与实际排名→自适应更新性能矩阵：

$$M_p^{\text{new}} = \text{Normalize}\big(M_p + \mathcal{W}_{task} \cdot \eta \cdot \Delta\big)$$

$$\Delta = \frac{\mathcal{R}_{theory} - \mathcal{R}_{actual}}{m+n}$$

- 当工具实际表现优于理论预期→提高其性能边界分数；反之降低
- $\eta$ 是更新步长，实验表明 $\eta=0.13$ 达到最佳平衡
- 新增工具→用同类工具的平均分初始化→不会被忽略

### 核心机制三：Capability-Aligned Planning Optimization (CAPO)

扩展Step-aware Preference Optimization (SPO)到Agent规划领域：
- 每步生成 $k$ 个候选子任务 $\{u_t^1, u_t^2, \ldots, u_t^k\}$
- Self-Evaluator评估每个子任务的执行结果→选winning/losing样本
- 优化目标（DPO变体）：

$$\mathcal{L}(\theta) = -\mathbb{E}\Big[\log\sigma\Big(\alpha\big(\log\frac{p_\theta(u_t^w | \tau^*, s^*, \mathcal{B}, h_{t-1})}{p_{\text{ref}}(u_t^w | \tau^*, s^*, \mathcal{B}, h_{t-1})} - \log\frac{p_\theta(u_t^l | \tau^*, s^*, \mathcal{B}, h_{t-1})}{p_{\text{ref}}(u_t^l | \tau^*, s^*, \mathcal{B}, h_{t-1})}\big)\Big)\Big]$$

- 集成记忆检索机制：用CLIP相似度检索历史成功任务序列作为上下文指导

### 关键设计亮点
- **评估器双粒度**：全局语义 $g^{global}$ + 局部语义 $g^{local}_i$，加权综合评估
- **探索-利用策略**：APU中 $\beta k$ 个候选从历史经验检索，$(1-\beta)k$ 个随机生成→平衡利用和探索
- **性能矩阵直接复用基准分数**：从T2I-CompBench和ImgEdit-Bench直接采用→降低评估成本

## 实验关键数据

### 基础图像生成 (T2I-CompBench)

| 方法 | 类型 | Color↑ | Shape↑ | Texture↑ | Spatial↑ | Non-Spatial↑ | Complex↑ |
|------|------|--------|--------|----------|----------|-------------|----------|
| FLUX | Diffusion | 0.7407 | 0.5718 | 0.6922 | 0.2863 | 0.3127 | 0.3771 |
| SD3 | Diffusion | 0.8132 | 0.5885 | 0.7334 | 0.3200 | 0.3140 | 0.3703 |
| GoT | CoT | 0.4793 | 0.3668 | 0.4327 | 0.2238 | 0.3053 | 0.3255 |
| T2I-R1 | CoT | 0.8130 | 0.5852 | 0.7243 | 0.3378 | 0.3090 | 0.3993 |
| GenArtist | Agent | 0.8482 | 0.6948 | 0.7709 | 0.5437 | 0.3346 | 0.4499 |
| T2I-Copilot | Agent | 0.8039 | 0.6120 | 0.7604 | 0.3228 | 0.3379 | 0.3985 |
| **PerfGuard** | **Agent** | **0.8753** | **0.7366** | **0.8148** | **0.6120** | **0.3754** | **0.5007** |

### 高级图像生成 (OneIG-Bench)

| 方法 | 类型 | Alignment↑ | Text↑ | Reasoning↑ | Style↑ |
|------|------|-----------|-------|-----------|--------|
| FLUX | Diffusion | 0.786 | 0.523 | 0.253 | 0.368 |
| SD3 | Diffusion | 0.801 | 0.648 | 0.279 | 0.361 |
| T2I-R1 | CoT | 0.793 | 0.662 | 0.297 | 0.370 |
| T2I-Copilot | Agent | 0.821 | 0.679 | 0.318 | 0.386 |
| **PerfGuard** | **Agent** | **0.834** | **0.684** | **0.350** | **0.395** |

### 复杂图像编辑 (Complex-Edit Level-3)

| 方法 | IF↑ | PQ↑ | IP↑ | Overall↑ |
|------|-----|-----|-----|----------|
| AnySD | 4.13 | 7.14 | 9.08 | 6.78 |
| Step1X_Edit | 7.95 | 8.66 | 7.70 | 8.10 |
| GenArtist | 6.14 | 7.24 | 6.19 | 6.52 |
| OmniGen | 7.52 | 8.86 | 8.01 | 8.13 |
| **PerfGuard** | **8.95** | **9.02** | **8.56** | **8.84** |

## 关键发现

1. **文本描述几乎无法区分工具**：仅靠文本描述选工具→错误率高达77.8%（QWen3-14B），即使用GPT-4o也有72.2%→多维性能评分矩阵将错误率降至30.5%→再加APU降至14.2%。

2. **PASM是核心贡献**：消融实验显示引入PASM后Color维度+3.42%、Texture维度+5.7%→性能边界建模对工具选择的正确性有根本性提升。

3. **APU的自适应效果显著**：Complex指标从0.4412→0.4738→通过实际执行反馈校准理论偏差→使性能矩阵更准确反映真实任务需求。

4. **CAPO使Planner具备工具感知能力**：训练后的Planner能感知工具性能边界→理解操作顺序对结果的影响（如先编辑背景会降低后续步骤成功率）。

5. **更新步长η的选择关键**：η=0.1收敛太慢，η=0.15初期快但后期剧烈振荡→η=0.13在步骤800达到最优14.2%错误率→需平衡收敛速度与稳定性。

6. **Token效率优势**：随工具数量从10增至200，传统文本方法Token消耗灾难性增长→PerfGuard的性能驱动选择不受工具数量影响→适合未来大规模Agent工具管理。

## 亮点与洞察

- **性能边界→可量化的工具能力画像**：将工具能力从模糊的文本描述转化为精确的多维数值矩阵→使工具选择从"猜测"变为"计算"，这是Agent系统工程化的重要一步。
- **闭环自校正**：APU形成"理论预测→实际执行→偏差反馈→矩阵更新"的闭环→系统在使用过程中持续自我改进→不依赖固定基准。
- **SPO从图像生成到Agent规划的迁移**：将Step-aware Preference Optimization从扩散模型去噪过程扩展到Agent的自回归规划过程→展示了偏好优化在Agent决策中的潜力。
- **可扩展性验证**：Token消耗实验表明PerfGuard方法在大规模工具库（200+工具）下依然高效→指向未来Agent社区的工具管理方案。

## 局限性

- **性能矩阵依赖现有基准**：直接采用T2I-CompBench和ImgEdit-Bench分数→新领域或无基准的工具需要额外评估。
- **APU收敛需要足够样本**：800步才达到最优→对使用频率低的工具可能更新不充分。
- **工具集限制了上限**：在Alignment和Text指标上PerfGuard优势不大→因为工具集本身的生成能力封顶。
- **推理开销**：每步生成k个候选子任务并分别执行评估→相比单次规划增加了计算成本（尽管选择时间已降低）。
- **仅限视觉生成/编辑任务**：尚未验证在其他Agent任务（代码生成、数据分析等）中的效果。

## 相关工作对比

### vs GenArtist (NeurIPS 2024)
GenArtist同样使用多模态LLM协调生成和编辑工具，但缺乏性能感知的工具选择策略→依赖详细文本描述→工具数量增多时推理时间显著增加。PerfGuard通过性能矩阵+量化选择→在工具选择时间和准确率上均优于GenArtist（Complex: 0.4499→0.5007）。

### vs T2I-Copilot
T2I-Copilot通过多Agent协作实现语义分解→但使用固定工具集→工具多样性受限→遗漏细节（如螺旋星系、绿色眼镜）。PerfGuard的性能感知选择→能自动匹配最佳工具→Reasoning: 0.318→0.350。

### vs CLOVA (CVPR 2024)
CLOVA通过自反思+prompt调优提升工具成功率→但仍在工具级别工作，未建模跨工具的性能比较。PerfGuard从工具选择层面系统性建模→更全面。

## 评分

- 新颖性: ⭐⭐⭐⭐ 性能边界建模+自适应更新+规划优化的组合是新的，但单个组件的技术创新有限
- 实验充分度: ⭐⭐⭐⭐ 三个基准+消融+效率分析+工具错误率分析，较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数学公式规范，但部分描述冗长
- 价值: ⭐⭐⭐⭐ 为Agent系统的工具选择提供了可行的工程化方案，实用性强

<!-- RELATED:START -->

## 相关论文

- [SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)

<!-- RELATED:END -->
