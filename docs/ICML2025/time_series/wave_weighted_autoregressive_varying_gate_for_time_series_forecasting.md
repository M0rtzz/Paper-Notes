---
title: >-
  [论文解读] WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting
description: >-
  [ICML 2025][时间序列][时间序列预测] 将经典统计学中的ARMA（自回归移动平均）结构引入自回归Transformer注意力机制，通过间接MA权重生成方法在不增加时间复杂度和参数量的前提下，解耦长短期时序模式，显著提升时间序列预测性能。
tags:
  - ICML 2025
  - 时间序列
  - 时间序列预测
  - ARMA
  - 自回归注意力
  - 移动平均
  - 线性注意力
---

# WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting

**会议**: ICML 2025  
**arXiv**: [2410.03159](https://arxiv.org/abs/2410.03159)  
**代码**: 有  
**领域**: 时间序列  
**关键词**: 时间序列预测, ARMA, 自回归注意力, 移动平均, 线性注意力

## 一句话总结

将经典统计学中的ARMA（自回归移动平均）结构引入自回归Transformer注意力机制，通过间接MA权重生成方法在不增加时间复杂度和参数量的前提下，解耦长短期时序模式，显著提升时间序列预测性能。

## 研究背景与动机

**领域现状**：近年来decoder-only自回归Transformer在NLP、CV、音频等领域取得了巨大成功，但在时间序列预测（TSF）领域，最优模型仍以encoder-only Transformer（如PatchTST、iTransformer）、MLP（如TiDE、CATS）甚至线性模型（DLinear、FITS）为主。少量关于自回归模型的探索主要集中在利用预训练LLM做few-shot/zero-shot预测，很少直接评估端到端训练的AR Transformer性能。

**痛点**：
1. 自回归Transformer在长期预测时面临**误差累积**问题——迭代式单步预测会导致误差逐步放大
2. 现有高效线性注意力机制（如gated linear attention）引入的**指数衰减门控**虽然能增强局部模式建模，但会削弱对长期和周期性模式的捕捉能力
3. 时间序列中**长期周期性模式和短期局部效应**的耦合是核心矛盾——单纯的EMA机制无法有效解耦二者

**核心矛盾**：gated linear attention中的衰减因子使AR权重专注于近期token，但TSF数据常包含稳定的季节性效应，这些效应不应随时间衰减。如何让模型既能建模短期波动又能捕捉长期稳定的周期模式？

**本文方案**：受经典Box-Jenkins ARMA模型启发，将完整的MA（移动平均）项引入现有AR注意力机制，构建WAVE注意力。MA项专门处理短期波动，使AR项能专注于长期和周期性模式。

**切入角度**：首先证明适当分词（tokenization）下AR Transformer可达到SOTA水平，再通过ARMA结构进一步提升。设计间接MA权重生成方法，避免显式计算$N \times N$的MA权重矩阵，保持$O(N)$时间复杂度。

## 方法详解

### 整体框架

采用GPT-2风格的decoder-only Transformer架构。输入时间序列经过通道独立处理和RevIN归一化后，使用非重叠patch进行分词（patch size = $L_P$，即预测长度），将输入分为$N = (L_I + P) / L_P$个token。每个token经线性投影到$d$维空间后加上可学习位置嵌入，输入$m$层WAVE Transformer。最后一个token的输出即为长度$L_P$的预测结果，避免了迭代式预测带来的误差累积。

### 关键设计

1. **适当分词避免误差累积**：借鉴PatchTST的patch策略，将patch size设为预测长度$L_P$，使得自回归的"下一个token预测"恰好覆盖整个预测区间。这样一步预测就完成了全部预测，无需迭代。同时采用通道独立方式（channel-independent），每个序列独立预测并施加RevIN归一化。核心公式：

   $$N = \frac{L_I + P}{L_P}$$

   其中$P$为零填充以保证整除。设计动机：消除decoder-only架构的误差累积问题，使其性能可与encoder-only模型对标。

2. **ARMA注意力结构（WAVE Attention）**：将标准注意力输出分解为AR项和MA项。AR项由原有注意力机制计算，MA项建模预测残差的短期模式：

   $$\bm{v}_{t+1} = \underbrace{\sum_{i=1}^{t} \mathbf{w}_{t,i} \odot \bm{v}_i}_{\text{AR项 } \bm{o}_t^{AR}} + \underbrace{\sum_{j=1}^{t-1} \bm{\theta}_{t-1,j} \odot \bm{\epsilon}_j}_{\text{MA项 } \bm{o}_t^{MA}} + \bm{\epsilon}_t$$

   其中$\bm{\epsilon}_t$为引入MA项后的剩余误差，$\bm{\theta}_{t-1,j}$为MA权重。这一结构来源于经典ARMA模型——AR项捕捉长期依赖和周期性模式，MA项捕捉短期波动和局部效应，实现对二者的有效解耦。

3. **间接MA权重生成方法**：直接计算MA权重需要对$N \times N$矩阵求逆（$\bm{\epsilon} = (\mathbf{I} + \mathbf{\Theta})^{-1} \mathbf{r}$），复杂度回到$O(N^2)$。本文的核心创新是用AR残差$\bm{r}_j = \bm{v}_{j+1} - \bm{o}_j^{AR}$替代$\bm{\epsilon}_j$作为MA的值输入：

   $$\bm{o}_t^{MA} = \sum_{j=1}^{t-1} \bm{\beta}_{t-1,j} \odot \bm{r}_j$$

   其中$\bm{\beta}_{t-1,j} = \phi_q^{MA}(\bm{q}_{t-1}^{MA}) \phi_k^{MA}(\bm{k}_j^{MA})^\top$通过线性注意力形式高效计算。间接生成的权重$\mathbf{B}$与隐式MA权重$\mathbf{\Theta}$的关系为：

   $$\mathbf{B} = \mathbf{\Theta} \cdot (\mathbf{I} + \mathbf{\Theta})^{-1}, \quad \mathbf{\Theta} = \mathbf{B} \cdot (\mathbf{I} - \mathbf{B})^{-1}$$

   设计动机：保持线性注意力的$O(N)$复杂度，同时产生有效的MA权重。

4. **激活函数选择确保MA权重特性**：MA项应建模短期效应，因此隐式$\mathbf{\Theta}$需呈现**近对角线元素大、远离对角线元素衰减**的模式。将$\mathbf{\Theta}$展开为$\mathbf{B} + \mathbf{B}^2 + \mathbf{B}^3 + \cdots$，若$\beta$的均值为$b$，则：

   $$\theta_{ij} = b(1+b)^{i-j-1}, \quad i > j$$

   为保证衰减，需$b \in (-1, 0)$。最终选择：
   - Key激活：$\phi_k^{MA}(\bm{k}_j^{MA}) = \sigma(\alpha \bm{k}_j^{MA} / \sqrt{d})$（sigmoid，$\alpha=0.05$）
   - Query激活：$\phi_q^{MA}(\bm{q}_t^{MA}) = -\text{LeakyReLU}(-\bm{q}_t^{MA} / \sqrt{d})$（负斜率0.02）

   设计动机：LeakyReLU提供灵活性——大部分输出为负值保证MA的负向平滑效应，少量正值增加建模灵活性。

5. **参数共享策略**：MA项引入了额外的$\mathbf{W}_q^{MA}, \mathbf{W}_k^{MA}, \mathbf{W}_v^{MA}$。为公平比较：AR和MA共享$\mathbf{W}_q$，MA的$\mathbf{W}_v$设为单位矩阵。最终可训练参数为$\mathbf{W}_q, \mathbf{W}_k^{AR}, \mathbf{W}_k^{MA}, \mathbf{W}_o$——与纯AR模型参数量相同。

### 损失函数 / 训练策略

- **损失函数**：标准MSE损失，采用next-step prediction目标。对最后一个token的损失额外乘以权重因子$N$（仅在小数据集ETTs上略有影响）
- **优化器**：AdamW（$\beta=(0.9, 0.95)$，weight decay=0.1），遵循GPT-2设置
- **学习率**：线性warm-up前5个epoch（$6\times10^{-5} \to 6\times10^{-4}$），之后逐步衰减
- **正则化**：AR项和MA项各施加0.1的dropout
- **模型维度**：$d = 16\lfloor\sqrt{C}\rfloor$（$C$为通道数），$m=3$层，8头
- **早停**：patience=12 epochs，最大100 epochs
- **归一化**：RevIN + RMSNorm

## 实验关键数据

### 主实验

在12个数据集上评估短期预测（$L_P \in \{12, 24, 48, 96\}$，$L_I=512$）：

| 数据集 | 指标(MSE) | WAVE Lin Attn | 纯AR Lin Attn | PatchTST | iTransformer | 提升(vs AR) |
|--------|-----------|---------------|---------------|----------|--------------|------------|
| Weather | Avg MSE | 0.100 | 0.104 | 0.107 | 0.117 | 3.8% |
| ETTm1 | Avg MSE | 0.222 | 0.238 | 0.244 | 0.259 | 6.7% |
| Traffic | Avg MSE | 0.330 | 0.337 | 0.358 | 0.330 | 2.1% |
| PEMS08 | Avg MSE | 0.116 | 0.119 | 0.121 | 0.117 | 2.5% |
| Solar | Avg MSE | 0.119 | 0.122 | 0.150 | 0.145 | 2.5% |

**排名统计**：WAVE Lin Attn平均排名2.333（#Top1=25/48），远超所有baseline和纯AR模型。

### 消融实验

| 配置 | Weather MSE | ETTm1 MSE | 说明 |
|------|-------------|-----------|------|
| WAVE (m=3) vs AR (m=1~8) | 0.100 vs 0.102~0.109 | 0.222 vs 0.230~0.241 | 3层WAVE优于任意层数AR |
| Lin Attn AR | 0.104 | 0.238 | 基础线性注意力 |
| Lin Attn +ARMA | 0.100 | 0.222 | 加MA项后一致提升 |
| GLin Attn AR | 0.119 | 0.407 | 门控衰减损害长期模式 |
| GLin Attn +ARMA | 0.105 | 0.260 | ARMA对门控注意力提升最大 |
| MEGA (EMA) | 0.121 | 0.412 | EMA不如ARMA解耦效果好 |

### 关键发现

1. **适当分词下AR Transformer可匹敌SOTA**：纯AR Transformer使用patch tokenization后，性能已与PatchTST、iTransformer等基线相当
2. **ARMA一致性提升**：在所有5种注意力机制（Softmax、Linear、Gated Linear、Element-wise Linear、Fixed）上，加入MA项后预测性能均显著提升
3. **线性注意力优于Softmax**：在TSF任务中，更简单的线性注意力模式和无归一化的输入shortcut有更好的泛化性
4. **门控注意力受益最大**：WAVE对gated linear attention带来最大提升，因为MA项接管了局部效应建模，释放了衰减因子在AR遗忘机制中的正常功能
5. **扩展回看长度不衰退**：$L_I$从512增至4096时，AR/WAVE模型持续改善，而基线模型普遍性能下降——体现了AR架构在长期依赖利用上的优势
6. **计算开销极低**：参数共享保证MA不增加参数量，额外FLOPs不显著（如ETTm1上Lin Attn从7.387M到7.415M FLOPs）

## 亮点与洞察

1. **经典统计学与深度学习的优雅融合**：将ARMA模型的核心思想（解耦长短期效应）成功引入Transformer注意力，理论动机清晰、实证效果显著
2. **间接MA权重生成是关键创新**：通过用$\bm{r}_j$替代$\bm{\epsilon}_j$作为值输入，巧妙避免了矩阵求逆，保持了线性复杂度——这一思路在技巧性和实用性上都很出色
3. **激活函数设计有理论支撑**：通过分析$\mathbf{\Theta} = \mathbf{B} + \mathbf{B}^2 + \cdots$的累积行为，推导出$b \in (-1, 0)$的约束，从而选择LeakyReLU+Sigmoid组合——不是随意调参而是有推导依据
4. **对EMA/门控机制的深刻分析**：指出指数衰减对TSF中稳定周期模式的破坏性，解释了为什么gated linear attention单独使用不如linear attention——这一洞察对该领域后续研究有启发
5. **参数共享保持公平性**：通过共享$\mathbf{W}_q$和设$\mathbf{W}_v^{MA}=\mathbf{I}$，消除了"性能提升来自参数增加"的质疑

## 局限与展望

1. **仅限通道独立模式**：未探索与多变量预测模型（如iTransformer的跨变量注意力）结合，可能错失变量间重要的相关性信息
2. **仅验证TSF任务**：WAVE注意力是否适用于NLP、音频等通用序列建模任务尚未验证
3. **大规模验证不足**：未在大规模数据集（如大规模NLP预训练）上测试，ARMA结构在大模型中的表现有待观察
4. **MA项的阶数固定**：当前MA项类似于完整阶的MA，未探索限制MA阶数（如ARMA(p,q)中的q）是否能进一步提升效率或性能
5. **$\alpha$参数敏感性**：key激活中的$\alpha=0.05$控制MA权重关注长/短期信息的程度，文中未充分探讨其敏感性

## 相关工作与启发

- **PatchTST** (Nie et al., 2022)：本文的patch tokenization策略直接借鉴PatchTST，证明了patch在AR架构中同样有效
- **Gated Linear Attention** (Yang et al., 2024)：本文在此基础上指出纯衰减机制在TSF中的局限性，启发了ARMA的引入
- **MEGA** (Ma et al., 2022)：使用EMA增强gated attention，但ARMA比EMA更有效地解耦长短期效应
- **ARMA Cell** (Schiele et al., 2022)：尝试在RNN中引入ARMA结构，但无法保证MA权重的短期建模特性，效果未超越现代注意力模型
- **RetNet** (Sun et al., 2023)：线性注意力与保持机制结合，与WAVE的高效线性注意力思路互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 将ARMA结构引入注意力机制的思路新颖，间接MA权重生成方法巧妙，但核心仍基于已有的线性注意力框架
- 实验充分度: ⭐⭐⭐⭐⭐ 12个数据集、5种注意力机制的全面交叉验证、层数消融、回看长度分析、计算成本对比、可视化分析——非常充分
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰、公式推导严谨、可视化分析直观，但部分符号较多需仔细阅读
- 价值: ⭐⭐⭐⭐ 为自回归Transformer在TSF中的应用提供了坚实的理论和实践基础，ARMA结构作为即插即用模块有较好的通用性

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[ICML 2025\] When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)
- [\[AAAI 2026\] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](../../AAAI2026/time_series/scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)

<!-- RELATED:END -->
