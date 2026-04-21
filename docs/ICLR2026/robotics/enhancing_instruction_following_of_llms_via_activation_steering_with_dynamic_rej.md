---
title: >-
  [论文解读] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection
description: >-
  [ICLR 2026][机器人][激活引导] 提出 Directer（Dynamic Rejection Steering），通过在每个解码步动态调节 KV 缓存引导强度并引入合理性约束，显著提升 LLM 指令遵循能力，同时避免过度引导导致的文本质量下降。
tags:
  - ICLR 2026
  - 机器人
  - 激活引导
  - 指令遵循
  - KV缓存缩放
  - 动态拒绝
  - 过度引导缓解
---

# Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection

**会议**: ICLR 2026  
**arXiv**: [2603.06745](https://arxiv.org/abs/2603.06745)  
**代码**: [有](https://github.com/mjk0618/directer)  
**领域**: 机器人  
**关键词**: 激活引导, 指令遵循, KV缓存缩放, 动态拒绝, 过度引导缓解  

## 一句话总结

提出 Directer（Dynamic Rejection Steering），通过在每个解码步动态调节 KV 缓存引导强度并引入合理性约束，显著提升 LLM 指令遵循能力，同时避免过度引导导致的文本质量下降。

## 研究背景与动机

LLM 在指令微调后仍难以完美遵循复杂用户指令。激活引导（Activation Steering）技术通过修改模型内部表示来增强指令遵循，但现有方法存在**过度引导（oversteering）**风险——过度强调指令会降低任务准确性和生成质量。

现有方法的核心问题：

**静态引导强度**：PASTA 和 SpotLight 等方法依赖手动调参的固定超参数，无法适应每个解码步最优引导强度的动态变化

**高预计算成本**：PASTA 需要数百到数千个验证样本做 attention head 的网格搜索，预计算成本接近训练级别

**计算开销大**：SpotLight 在每个解码步需要额外的 softmax 操作，有效地使延迟翻倍

**质量-遵循权衡**：增强指令遵循往往以牺牲任务正确性和文本质量为代价

## 方法详解

### 整体框架

Directer 的核心思想是在解码过程中**自动调节引导强度**，通过三个关键组件协同工作：

1. **KV 缓存引导**：对指令 token 对应的 Key 向量进行缩放（缩放因子 α=100）
2. **合理性引导解码循环**：在每步比较引导后与原始输出分布，动态决定是否接受引导
3. **基于注意力敏感度的层排序**：一次性分析确定各层对引导的敏感程度，指导渐进式强度调节

### 关键设计

**KV 缓存缩放机制**：对指令 token 区间 $\mathcal{I}$ 内的 Key 向量乘以缩放因子 $\alpha$：

$$\mathbf{k'}^{(l)}_i = \begin{cases} \alpha \cdot \mathbf{k}^{(l)}_i, & \text{if } i \in \mathcal{I} \text{ and } l \in \mathcal{L} \\ \mathbf{k}^{(l)}_i, & \text{otherwise} \end{cases}$$

选择 Key 缩放而非 Value 缩放，因为 Key 缩放的效果会被后续 softmax 自然归一化，无需额外计算。

**合理性引导解码**：每个解码步的流程如下：

1. 执行标准前向传播得到原始分布 $p_t$
2. 对候选层集合 $\mathcal{L}_{\text{cand}}$ 应用 KV 缓存引导得到引导分布 $\tilde{p}_t$
3. 检查合理性条件：$p_{t,\tilde{i}^*_t} \geq \beta \cdot p_{t,i^*_t}$（阈值 $\beta=0.5$）
4. 若不满足，渐进减半候选层数（移除敏感度最低的一半层）
5. 重复直到通过或候选层为空（回退到原始分布）

**高效门控机制**：利用原始分布 top-2 token 的概率进行预判——若 $p_{t,i^{**}_t} < \beta \cdot p_{t,i^*_t}$，则可直接跳过引导尝试，大幅减少计算开销。

**注意力敏感度层排序**：通过单层引导观测扰动传播来排序层的影响力。对每一层 $\ell$ 单独施加引导，测量对所有层 $j$ 的扰动分数：

$$D_j(\ell) = \underbrace{(\text{dist}(\mathbf{H}^{(j)}_{\text{pre}}, \mathbf{H}^{(j,\ell)}_{\text{post}}) - \text{dist}(\mathbf{H}^{(j)}_{\text{pre}}, \mathbf{H}^{(j)}_{\text{post}}))}_{\text{直接效应}} + \underbrace{(\text{dist}(\mathbf{H}^{(j,\ell)}_{\text{pre}}, \mathbf{H}^{(j)}_{\text{post}}) - \text{dist}(\mathbf{H}^{(j)}_{\text{pre}}, \mathbf{H}^{(j)}_{\text{post}}))}_{\text{传播效应}}$$

最终排序 $\text{Sensitivity}(\ell) = \frac{1}{L}\sum_{j=1}^{L} D_j(\ell)$，仅在 prompt prefill 后执行一次。

### 损失函数 / 训练策略

Directer 是纯推理时方法，无需训练。核心超参数仅有两个：
- **缩放因子 α=100**：对性能影响极小，在 $10^1 \sim 10^5$ 范围内表现稳定
- **合理性阈值 β=0.5**：控制引导介入频率，在全范围内均优于无引导基线

所有任务使用统一配置，无需任务特定调参。

## 实验关键数据

### 主实验

| 方法 | IFEval P.Acc / I.Acc | LIFBench List/OD/MD | GSM8K-Format F.Acc/T.Acc | 平均 |
|------|---------------------|---------------------|--------------------------|------|
| Zero-shot | 73.5 / 81.5 | 63.4 / 68.6 / 40.9 | 79.2 / 82.7 | 70.0 |
| PASTA* | 76.5 / 83.4 | 61.8 / 66.0 / 47.8 | 98.9 / 62.7 | 71.0 |
| SpotLight* | 76.3 / 83.6 | 61.4 / 70.8 / 38.8 | 95.4 / 78.7 | 72.1 |
| **Directer** | **78.8 / 84.8** | **64.4 / 70.0 / 51.7** | **99.1 / 86.9** | **76.5** |

| 模型规模 | Zero-shot | PASTA* | SpotLight* | Directer |
|----------|-----------|--------|------------|----------|
| Llama-3.2-1B | 61.3 | 59.7 | 60.6 | **61.6** |
| Qwen-2.5-3B | 63.9 | 65.2 | 62.8 | **67.1** |
| Qwen-2.5-7B | 72.4 | 73.0 | 74.9 | **74.4** |
| Qwen-2.5-14B | 81.6 | 80.1 | 81.7 | **83.5** |

### 消融实验

| 变体 | 准确率 |
|------|--------|
| Zero-shot | 77.5 |
| **Directer (完整)** | **81.8** |
| + 排序反转 | 79.0 |
| + 随机层引导 | 80.2±0.7 |
| + 随机 token 引导 | 79.2±1.1 |

### 关键发现

1. **过度引导问题严重**：PASTA 原始设置导致 GSM8K 任务准确率从 82.7% 暴跌至 48.1%，Directer 保持 86.9%
2. **动态优于静态**：固定引导强度中，低强度（ST1/ST2）略有提升但高强度急剧下降，Directer 自适应调节全面超越
3. **合理性约束具有通用性**：作为安全门应用于 PASTA/SpotLight 也能显著改善其过度引导问题
4. **推理效率可控**：吞吐量仅比零射基线低约 16%，比 SpotLight 快 2 倍以上，内存开销可忽略
5. **生成质量与任务保真度最优**：LLM 评审的任务保真度达 ≈92%，文本质量与无干预基线持平

## 亮点与洞察

1. **问题定义精准**：将过度引导识别为激活引导方法的核心瓶颈，而非简单地追求更强的引导
2. **设计极为优雅**：合理性检查 + 渐进减半 + 敏感度排序三者环环相扣，形成自适应闭环
3. **近乎零调参**：α 在 5 个数量级范围稳定，β 全范围优于基线，真正实现即插即用
4. **KV 缓存操作兼容 FlashAttention**：这是注意力级别干预方法不具备的实际优势

## 局限与展望

1. 需要明确的指令区间标注（instruction span），自动化识别指令边界的能力有待探索
2. 层排序基于单次 prefill 分析，对于多轮对话中指令动态变化的场景可能需要更新
3. 实验主要在 Llama 和 Qwen 系列上验证，对 Mixture-of-Experts 等架构的适用性未知
4. 当前仅验证了 greedy decoding，与 sampling 策略的兼容性有待验证

## 相关工作与启发

- **与 PASTA 的关系**：PASTA 通过抑制非指令 token 的注意力分数来实现引导，但需要大量验证样本做 head profiling，且静态配置易过度引导；Directer 无需额外数据集且动态调节
- **与 SpotLight 的关系**：SpotLight 通过 post-softmax logit biasing 维持指令 token 的目标注意力比例，计算量大；Directer 通过 KV 缓存操作实现更高效的引导
- **启发**：合理性引导解码的框架可推广到其他需要平衡干预强度的场景（如安全对齐、风格控制）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 动态拒绝引导机制和注意力敏感度排序均为新颖贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 多基准、多模型、多消融、效率分析、生成质量评估面面俱到
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，公式推导严谨
- 价值: ⭐⭐⭐⭐ — 即插即用的推理时增强模块，实用性强

<!-- RELATED:START -->

## 相关论文

- [PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)
- [ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)
- [ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments](../../ECCV2024/robotics/realfred_an_embodied_instruction_following_benchmark_in_photo-realistic_environm.md)
- [Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)
- [Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](../../ICML2025/robotics/hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)

<!-- RELATED:END -->
