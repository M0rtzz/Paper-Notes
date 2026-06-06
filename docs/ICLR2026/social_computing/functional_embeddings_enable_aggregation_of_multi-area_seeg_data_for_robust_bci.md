---
title: >-
  [论文解读] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI
description: >-
  [ICLR 2026][社会计算][脑机接口] 提出 FunctionalMap 框架，通过对比学习从颅内局部场电位（LFP）中学习被试无关的功能嵌入作为"功能坐标系"，替代不可靠的 MNI 解剖坐标，结合 Transformer 实现跨被试、跨电极的神经数据聚合和信号重建…
tags:
  - "ICLR 2026"
  - "社会计算"
  - "脑机接口"
  - "SEEG"
  - "功能嵌入"
  - "对比学习"
  - "Transformer"
  - "跨被试建模"
  - "神经信号"
---

# Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI

**会议**: ICLR 2026  
**arXiv**: [2510.27090](https://arxiv.org/abs/2510.27090)  
**代码**: [GitHub](https://github.com/ICLR-Functional-Embedding/ICLR2026_Functional_Map)  
**领域**: 社会计算  
**关键词**: 脑机接口, SEEG, 功能嵌入, 对比学习, Transformer, 跨被试建模, 神经信号

## 一句话总结

提出 FunctionalMap 框架，通过对比学习从颅内局部场电位（LFP）中学习被试无关的功能嵌入作为"功能坐标系"，替代不可靠的 MNI 解剖坐标，结合 Transformer 实现跨被试、跨电极的神经数据聚合和信号重建，在 20 名被试的多脑区 SEEG 数据集上验证有效。

## 研究背景与动机

颅内神经记录（如 SEEG/DBS）的跨被试建模面临两大核心困难：

**解剖变异性和不一致的电极覆盖**：电极的数量、位置和覆盖区域因临床需求而异。标准的 MNI 图谱对齐假设空间对应等于功能相似，但**匹配解剖坐标处的记录常捕获不同的功能角色**，极端情况下甚至是完全不同的脑区。

**多区域记录的异质性**：现代 DBS 手术同时从多个基底节和丘脑核团采样（GPi、STN、VO、VA、VIM 等），提供了研究区间通信的独特机会，但其异质性放大了对齐问题。

现有方法的局限：
- EEG 基础模型（如 LaBraM）假设固定的高密度电极网格
- MNI 坐标系方法（Mentzelopoulos et al., 2024）依赖不可靠的解剖定位
- PopT（Chau et al., 2025）聚合冻结的单通道嵌入但使用位置编码

**核心假设**：神经信号通过其功能特征（而非解剖坐标）可以更可靠地跨被试对齐。

## 方法详解

### 整体框架

FunctionalMap 分两阶段：

1. **功能嵌入学习**：Siamese 编码器 + 对比学习，从 LFP 信号中学习 32 维被试无关的功能身份嵌入
2. **功能 Transformer**：利用功能嵌入作为 token 坐标，对可变数量通道建模区间关系，执行掩蔽区域重建

### 关键设计

**功能嵌入网络**：轻量级 CNN 编码器 $f_\theta: \mathbb{R}^T \to \mathbb{R}^d$（$d=32$），将 10 秒 LFP 片段映射到嵌入空间。两种对比学习方法：

**方法 1：配对 Siamese 对比（PSC）**：

$$\mathcal{L}_{\text{pair}} = \frac{1}{|\mathcal{B}|} \sum_{(i,j) \in \mathcal{B}} \left[(1-y_{ij}) d_{ij}^2 + y_{ij} (\max(0, m-d_{ij}))^2\right]$$

同区域（$y_{ij}=0$）的配对拉近，不同区域（$y_{ij}=1$）推开至距离 $m=0.5$ 以上。

**方法 2：改进的有监督对比（MSC）**：多正样本 InfoNCE + 类内方差惩罚：

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}}$$

其中 $\mathcal{L}_{\text{sup}}$ 使用余弦相似度（温度 $\tau=0.2$），$\mathcal{L}_{\text{var}}$ 惩罚同区域嵌入的方差（$\lambda_{\text{var}}=0.05$）。MSC 在超球面上操作，强调角度分离。

**训练采样**：输入对可来自同一 session 或跨被试/session，时间不同步。区域一致的随机采样确保编码器学习对被试和 session 变异鲁棒的区域特异性神经签名。

**功能 Transformer（掩蔽区域重建）**：
- **任务**：遮蔽某脑区的所有通道，要求模型从其他区域预测之
- **Tokenization**：1D 卷积 tokenizer 将源通道转为时间 patch 特征，与功能嵌入拼接融合；目标通道用学习到的查询基与功能嵌入融合
- **架构**：标准 pre-LN encoder-decoder Transformer，无被试 ID、无被试特异性头
- **目标函数**：MSE + 相关性项

$$\mathcal{L} = \text{MSE}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda(1 - \rho(\hat{\mathbf{Y}}, \mathbf{Y})), \quad \lambda=0.05$$

相关性项防止纯 MSE 倾向于幅度缩减/平坦预测。

### 损失函数 / 训练策略

- 功能嵌入：对比损失（PSC 或 MSC），10 秒 LFP 片段
- Transformer：MSE + Pearson 相关损失，跨 11 名被试联合训练
- 单一共享模型，无被试特异性微调

## 实验关键数据

### 主实验

**数据集**：20 名肌张力障碍患者的颅内 LFP 记录，覆盖 GPi/STN/VO/VA/VIM/PPN/SNr，共 442.86 电极小时。

**单被试功能嵌入**：

| 评估设置 | 准确率（Mean±SD） |
|---------|-----------------|
| 留出时间段（已见通道） | 75.78% ± 17.90% |
| 留出通道（>3 通道/区域） | 45.79% ± 18.44%（高于 chance） |

**多被试联合训练 vs 单被试**：

| 设置 | 留出时间段 | 留出通道 |
|------|----------|---------|
| 单被试 | 75.78% ± 17.90% | 45.79% ± 18.44% |
| 多被试联合 | **80.71% ± 11.41%** | **49.18% ± 12.11%** |

联合模型在两个指标上均提升约 5%，且无需被试特异性微调。

### 消融实验

**坐标系消融（掩蔽区域重建，预测 VO 通道）**：

| 坐标系 | Pearson 相关 r |
|--------|---------------|
| MNI 坐标 | 基线 |
| Functional-1 (PSC) | 正趋势但不显著 |
| **Functional-2 (MSC)** | **显著优于 MNI**（$p \approx 0.002$） |

**对比方法比较（PSC vs MSC）**：

| 方法 | 留出时间段准确率 | 留出通道准确率 | 特点 |
|------|---------------|-------------|------|
| PSC | 略高 | 较低 | 欧氏空间，紧凑质心聚类 |
| MSC | 略低 | **更高** | 超球面，角度分离，通道泛化更强 |

**与被试特异性基线比较**：

Transformer + Functional-2 显著优于所有被试特异性基线（线性 FIR、TCN、2 层 GRU、CopyBest），所有校正后 $p < 0.001$。

### 关键发现

1. **功能嵌入成功聚类脑区**：跨被试形成清晰的区域一致聚类
2. **零样本迁移到未见通道**：联合模型无需微调即可处理新电极
3. **功能坐标显著优于解剖坐标**：MSC 嵌入在重建任务上显著改善
4. **MNI 的失败案例**：四个 VO 电极共享几乎相同的 MNI 坐标时，MNI 模型产生相似重建；功能嵌入产生通道特异性预测
5. **模拟验证**：在已知参数的模拟数据上，嵌入正确捕获频域特征，扰动分析确认敏感于生理相关成分

## 亮点与洞察

1. **核心假设有力**："功能作为坐标系"替代解剖坐标，对不一致定位和异质电极布局提供鲁棒对齐
2. **验证层次递进**：模拟→单被试→多被试→Transformer 重建→坐标系消融，每层验证假设的不同方面
3. **对比学习几何差异有意义**：PSC 的紧凑质心 vs MSC 的角度分离导致不同泛化行为
4. **自监督预训练目标设计巧妙**：掩蔽区域重建不需要行为标签，纯粹利用区间神经回路信息
5. **临床意义**：为 DBS 等临床神经技术的跨患者数据共享提供基础

## 局限性 / 可改进方向

1. **依赖区域标签**：对比训练需要知道电极所在脑区标签，限制了完全无监督的扩展
2. **仅限基底节-丘脑回路**：未验证在皮层 ECoG 和锋电位上的泛化性
3. **Transformer 仅用 11/20 名被试**：受限于 MNI 数据可用性
4. **任务范围有限**：仅验证了信号重建，未测试行为解码等下游任务
5. 与 PopT 等种群级预训练框架的完整对比仍待进行
6. 可探索弱监督/自监督目标替代区域标签

## 相关工作与启发

- **Mentzelopoulos et al. (2024)**：MNI 坐标 + 被试特异性头，发现位置编码无显著改善
- **PopT (Chau et al., 2025)**：种群级 Transformer，聚合冻结单通道嵌入
- **NDT/STNDT**：神经种群建模的 Transformer，假设稳定通道身份
- 启发：功能坐标 + 种群级预训练的结合可能产生最佳效果；"功能作为坐标系"的理念可推广到其他传感器对齐问题

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 功能坐标系替代解剖坐标是有力的新范式
- **技术深度**: ⭐⭐⭐⭐ — 对比学习 + Transformer 的两阶段设计完整
- **实验充分性**: ⭐⭐⭐⭐ — 模拟验证 + 真实数据 + 多层消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富
- **实用价值**: ⭐⭐⭐⭐ — 对临床神经科学和 BCI 有直接价值
- **综合推荐**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels](../../ECCV2024/social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)
- [\[ICML 2026\] ObjEmbed: Towards Universal Multimodal Object Embeddings](../../ICML2026/social_computing/objembed_towards_universal_multimodal_object_embeddings.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](../../ACL2026/social_computing/the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)
- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
