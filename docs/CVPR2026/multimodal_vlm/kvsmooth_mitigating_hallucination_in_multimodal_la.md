---
title: >-
  [论文解读] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing
description: >-
  [CVPR 2026][多模态][多模态幻觉] KVSmooth 提出免训练即插即用方法，通过对 KV-Cache 施加注意力行熵引导的自适应 EMA 平滑，将 LLaVA-1.5 的 CHAIR_S 从 41.8 降至 18.2（降 56%），同时 F1 从 77.5 提升到 79.2，精度召回同时提升。
tags:
  - CVPR 2026
  - 多模态
  - 多模态幻觉
  - KV缓存平滑
  - 注意力行熵
  - EMA
  - 免训练
---

# KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing

**会议**: CVPR 2026  
**arXiv**: [2602.04268](https://arxiv.org/abs/2602.04268)  
**代码**: 无  
**领域**: 多模态VLM / 幻觉缓解  
**关键词**: 多模态幻觉, KV缓存平滑, 注意力行熵, EMA, 免训练

## 一句话总结

KVSmooth 提出免训练即插即用方法，通过对 KV-Cache 施加注意力行熵引导的自适应 EMA 平滑，将 LLaVA-1.5 的 CHAIR_S 从 41.8 降至 18.2（降 56%），同时 F1 从 77.5 提升到 79.2，精度召回同时提升。

## 研究背景与动机

**领域现状**：MLLM（LLaVA、MiniGPT-4、InstructBLIP）在图像描述等视觉语言任务上取得显著进展，但生成内容中常出现与输入图像不一致的幻觉（hallucination）。

**现有痛点**：(1) 重训练/微调方法（如 POVID）成本高昂；(2) 对比解码方法（如 VCD）以牺牲召回率为代价降低幻觉（F1 从 77.5 降至 71.1）；(3) 注意力重分配方法（PAI、MiddleLayer）同样抑制正确对象。

**核心矛盾**：随解码序列增长，早期视觉 token 影响力在隐状态中逐渐衰减（语义漂移），模型越来越依赖语言先验。现有方法只能治标：要么牺牲召回换精确度，要么牺牲效率换质量。

**本文目标** 在不需要重训练、不牺牲召回率、几乎零额外开销的前提下，抑制解码中的语义漂移和幻觉。

**切入角度**：从隐状态动态演化视角出发，发现 attention sink 是幻觉的直接诱因，提出对 KV-Cache 做自适应平滑。

**核心 idea**：行熵量化 sink 程度 → 高 sink token 给更强 EMA 平滑 → 抑制语义漂移 → 消除幻觉。

## 方法详解

### 整体框架

KVSmooth 是推理时应用于 KV-Cache 的自适应 EMA 平滑方法。完全免训练、即插即用，仅需在推理时对每步新 token 的 Key 和 Value 做 EMA 更新，平滑系数由注意力行熵自适应决定。

### 关键设计

1. **三个关键观察（诊断因果链）**:

    - **Obs1（logit 动态分歧）**：在 200 张图像上统计发现，真实对象 logit 均值/方差随解码单调下降，幻觉对象 logit 则稳步上升——解码越长、幻觉越严重
    - **Obs2（行熵 ≈ sink 强度）**：提出注意力行熵作为 token sink 程度的实时度量。行熵高 → 注意力分布均匀 → 隐状态接近历史均值 → 角距离小 → 后续步吸引大量注意力形成 sink。与传统列和方法余弦相似度达 0.79
    - **Obs3（行熵↑ → 幻觉↑）**：幻觉对象的行熵-logit排名余弦相似度最高，真实对象则为负相关——sink token 通过全局平均系统性膨胀幻觉分数

2. **EMA Smoothing on KV-Cache**:

    - 功能：对 KV-Cache 中的 Key 和 Value 向量施加指数移动平均平滑
    - 核心思路：将隐状态演化建模为马尔可夫随机游走，贝叶斯 MAP 估计自然推导出 EMA 形式
    - 关键公式：$\hat{K_t^l} = (1-\tilde{\lambda}_t^l)K_t^l + \tilde{\lambda}_t^l K_{t-1}^l$，Value 同理
    - 设计动机：实验表明同时平滑 K+V 效果最佳（优于仅平滑 K 或直接平滑 hidden state），因为同时平滑能最大限度抑制 logit 均值和方差增长

3. **Entropy-Guided Coefficient Adaptation**:

    - 功能：根据每个 token 的 sink 程度自适应调节平滑强度
    - 核心思路：维护长度 M=15 的 FIFO 队列记录最近行熵值，当前 token 的平滑系数等于其行熵在队列中的百分位排名。行熵越高→百分位越高→平滑越强
    - 稳定化：系数裁剪到 $[\lambda_{ref}-0.2, \lambda_{ref}+0.2]$ 防止极端值
    - 设计动机：不同 token 对幻觉贡献不同，固定系数会过度抑制正常 token 的语义流，自适应机制让高 sink token 得到精准打击

### 损失函数 / 训练策略

完全免训练。超参数：平滑层范围 3-31 层（排除 0-2 层和最后层），FIFO 队列长度 15，$\lambda_{ref}$ 为 LLaVA-1.5=0.9 / MiniGPT-4=0.5 / InstructBLIP=0.7（跨 benchmark 固定）。

## 实验关键数据

### 主实验（CHAIR 幻觉评估）

| 模型 | 方法 | CHAIR_S↓ | F1↑ | 说明 |
|------|------|----------|-----|------|
| LLaVA-1.5 | Baseline | 41.8 | 77.5 | - |
| LLaVA-1.5 | VCD | 56.0 | 71.1 | 幻觉反增+召回骤降 |
| LLaVA-1.5 | OPERA | 44.2 | 78.6 | 推理慢10× |
| LLaVA-1.5 | PAI | 22.6 | 75.5 | 精度损失 |
| LLaVA-1.5 | MiddleLayer | 17.8 | 75.9 | 精度损失 |
| LLaVA-1.5 | **KVSmooth** | **18.2** | **79.2** | **精度召回同时↑** |
| MiniGPT-4 | Baseline | 31.8 | 69.9 | - |
| MiniGPT-4 | **KVSmooth** | **17.0** | **71.7** | CHAIR_S 降 47% |
| InstructBLIP | Baseline | 61.4 | 71.6 | - |
| InstructBLIP | **KVSmooth** | **42.2** | **75.1** | CHAIR_S 降 31% |

### 消融实验

| 配置 | CHAIR_S | F1 | 说明 |
|------|---------|-----|------|
| 仅平滑 K | 高于 K+V | - | 幻觉抑制弱 |
| 仅平滑 hidden state | - | 骤降 | 召回率大幅下降 |
| 同时平滑 K+V（最佳） | 18.2 | 79.2 | 最优组合 |
| 固定系数（最佳固定） | >自适应 | <自适应 | 两项指标均劣于自适应 |
| 排除 0-2 层 + 最后层 | 最优 | 最优 | 这些层不适合平滑 |

### 关键发现

- 推理速度 3.61s/caption，仅比 baseline（3.36s）慢 7%，远快于 OPERA（34.62s）
- Object HalBench 上 CHAIR_SR 下降 63.1%
- $\lambda_{ref}$ 越大平滑越强/CHAIR_S 越低，但 F1 几乎不变——方法对超参数鲁棒
- **唯一在 CHAIR_S 和 F1 上同时优于所有 baseline 的方法**

## 亮点与洞察

- 独特的因果链分析：logit 分歧→行熵=sink 度→sink 放大幻觉，三个观察构成完整诊断
- 行熵作为 sink 强度实时度量是重要贡献——比 OPERA 列和方法更高效（无需回溯）
- 理论推导优雅：从贝叶斯 MAP 估计严格推导出 EMA 平滑的最优性
- 精度和召回率同时提升（其他方法通常此消彼长），这是方法的核心竞争力
- 推理开销几乎为零（+7% 延迟，memory 不变），实用性极强

## 局限与展望

- $\lambda_{ref}$ 需要对不同模型手动调参，虽然跨 benchmark 可固定
- 仅在 7B 模型上验证，70B+ 规模模型效果待验证
- 当前仅在图像描述任务评估，VQA、对话等任务效果未知
- EMA 窗口固定为 1 步（仅用前一个 token），更长窗口可能进一步提升

## 相关工作与启发

- **vs VCD（对比解码）**：VCD 大幅降低召回率（F1 71.1），KVSmooth 精度召回同时提升（F1 79.2）
- **vs OPERA（注意力惩罚）**：OPERA 需回溯重分配注意力，推理慢 10×；KVSmooth 几乎无额外开销
- **vs PAI/MiddleLayer（注意力重分配）**：这些方法提升精度但损害召回率，KVSmooth 通过自适应机制精准识别需平滑的 token
- **vs PruneHal（KV 剪枝）**：PruneHal 删除冗余 token，KVSmooth 保留所有 token 但调节影响力
- 行熵度量可迁移到异常注意力检测场景；EMA 平滑 KV-Cache 可与量化/剪枝技术结合

## 评分

⭐⭐⭐⭐⭐ (4.5/5)

- **新颖性** ⭐⭐⭐⭐：行熵→sink→幻觉的因果分析新颖，EMA 平滑 KV-Cache 视角独特
- **实验充分度** ⭐⭐⭐⭐⭐：4 个 benchmark × 3 个模型 + PR 曲线 + 效率分析 + 丰富消融
- **写作质量** ⭐⭐⭐⭐⭐：观察→推导→方法→验证的逻辑链非常清晰
- **实用价值** ⭐⭐⭐⭐：免训练、即插即用、几乎零开销的幻觉缓解方案
