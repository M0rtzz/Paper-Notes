---
title: >-
  [论文解读] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning
description: >-
  [ICLR2026][物理/科学计算][神经算子] 提出 DRIFT-Net 双分支神经算子，通过受控低频混合（谱分支）和局部细节保真（图像分支）的带宽融合（radial gating），解决窗口注意力中全局谱耦合不足导致的自回归漂移问题，在 Navier-Stokes 基准上误差降低 7%-54%。
tags:
  - "ICLR2026"
  - "物理/科学计算"
  - "神经算子"
  - "偏微分方程"
  - "spectral coupling"
  - "dual-branch"
  - "Navier-Stokes"
---

# DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning

**会议**: ICLR2026  
**arXiv**: [2509.24868](https://arxiv.org/abs/2509.24868)  
**代码**: 无  
**领域**: 科学计算  
**关键词**: neural operator, PDE, spectral coupling, dual-branch, Navier-Stokes

## 一句话总结
提出 DRIFT-Net 双分支神经算子，通过受控低频混合（谱分支）和局部细节保真（图像分支）的带宽融合（radial gating），解决窗口注意力中全局谱耦合不足导致的自回归漂移问题，在 Navier-Stokes 基准上误差降低 7%-54%。

## 研究背景与动机

### 领域现状

**领域现状**：PDE 基础模型大多采用多尺度窗口自注意力，计算高效但全局依赖只能通过深层堆叠和窗口移位逐步传播。

**现有痛点**：窗口注意力的局部性削弱全局谱耦合，导致闭环 rollout 漂移；纯谱算子（FNO）过度强调低频。

**核心矛盾**：全局耦合和局部细节保真之间的权衡。

**本文目标** 在保持高频细节的同时增强全局谱耦合。

**核心 idea**：低频可学线性变换 + radial gating 带宽融合 + 频率加权损失。

## 方法详解

### 整体框架

DRIFT-Net 是一个 U-Net 形状的多尺度编码器-解码器，每个尺度上并排放着两条互补的分支：谱分支负责在频域里补全全局耦合，图像分支负责在空间域里保住局部细节，两者逐元素相加后再送入下一尺度。谱分支内部先做 rFFT2 把特征变到频域，对低频做受控混合，再用带宽融合把处理过的低频和原始高频重新拼回去，最后 iFFT2 变回空间域；图像分支则是一个标准 ConvNeXt 块。

### 关键设计

**1. 受控低频混合：让全局耦合不靠堆深度也能传播。** 窗口注意力的全局依赖只能靠层层堆叠和窗口移位慢慢扩散，闭环 rollout 时这种"慢传播"会累积成漂移。DRIFT-Net 的做法是把特征经 rFFT2 变到频域后，只对低频系数施加一个可学习的复线性变换，高频系数原样保留。低频对应的正是 PDE 解里那些大尺度、长程关联的结构，对它们做一次全局线性混合，等价于在单层内就完成了跨整张图的信息交换；而把高频（局部纹理、锐利边界）排除在外，避免这种全局混合把细节抹平。这样既拿到了纯谱算子（如 FNO）的全局视野，又不会像 FNO 那样过度偏向低频。

**2. 带宽融合与 radial gating：用凸组合保证融合不过冲。** 低频被改写、高频保持原样之后，需要把两段频谱重新缝合。DRIFT-Net 不用非黑即白的硬掩码切割，而是按频率半径 $k$ 做软融合：$\hat{Y}(k) = \alpha(k)\hat{V}_{low}(k) + (1-\alpha(k))\hat{X}_{high}(k)$，其中 $\alpha(k)$ 是一个随频率半径变化的门控（radial gating），在低频段接近 1、高频段接近 0，中间平滑过渡。由于这是一个系数恒非负且和为 1 的凸组合，融合后任意频率的能量都被夹在两个输入之间，不会出现某个频段被放大到溢出的能量过冲，训练因此更稳定——消融里换成硬掩码就会失稳。

**3. 频率加权损失：抵消神经网络天生的谱偏差。** 神经网络在回归任务里有偏向低频的谱偏差，直接用均匀 MSE 会让高频细节欠拟合，而高频误差恰恰是长程 rollout 漂移的来源之一。DRIFT-Net 在损失里按频率半径 $r$ 给误差加权，权重 $w(r) \propto r^\alpha$ 随频率单调增大，把优化压力更多地压到高频上，强迫模型把锐利结构也拟合好。

### 损失函数 / 训练策略

训练时采用单步 teacher forcing，即每步都用真实状态作为输入预测下一帧；测试时则换成自回归闭环 rollout，把模型自己的预测反馈回输入连续推演（实验中可达 100 步）。这种训练-测试的差异正是漂移问题的考验场景，上面三个设计共同保证了闭环推演下误差不快速累积。

## 实验关键数据

### 主实验：7 个 PDE 基准

| 任务 | scOT | FNO | **DRIFT-Net** |
|------|------|-----|------|
| NS-SL | 3.96% | 3.69% | **3.40%** |
| NS-PwC | 2.35% | 4.57% | **最佳** |
| Poisson-Gauss | - | - | **最佳** |
| Allen-Cahn | - | - | **最佳** |
| Wave-Gauss | - | - | **最佳** |

### 效率对比
参数量比 scOT 少约 15%，吞吐量更高，NS 误差降低 7%-54%。

### 消融实验

| 配置 | 效果 |
|------|------|
| 无低频混合 | 误差显著增加 |
| 硬掩码替代 radial gating | 不稳定 |
| 无频率加权损失 | 高频拟合不足 |
| 完整 DRIFT-Net | 最佳 |

### 关键发现
- 受控低频混合是关键——去掉后误差显著增加
- 100 步长程 rollout 中保持低漂移
- 对椭圆、抛物、双曲 PDE 均有效

## 亮点与洞察
- 谱-空间双分支巧妙解耦全局结构和局部细节——物理直觉强
- 非扩展融合的凸组合保证确保训练稳定
- 模块化——DRIFT block 可替换现有注意力块

## 局限与展望
- 低频掩码大小需手动设定
- 主要在 2D PDE 上验证，3D 扩展待测
- 与 DPOT 等其他 PDE 基础模型对比不足

## 相关工作与启发
- **vs scOT/POSEIDON**: 用谱分支实现全局耦合，无需深层堆叠
- **vs FNO**: FNO 全频操作但缺局部能力，DRIFT-Net 双分支互补
- **vs PDE-Refiner**: Refiner 靠迭代细化，DRIFT-Net 靠架构设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 受控低频混合+带宽融合+频率损失的精巧组合
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个 PDE 基准+消融+长程 rollout
- 写作质量: ⭐⭐⭐⭐ 物理直觉解释好
- 价值: ⭐⭐⭐⭐ 为 PDE 基础模型提供更好骨干

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](../../ICML2026/physics/topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[AAAI 2026\] SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](../../AAAI2026/physics/saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](../../ICML2026/physics/eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/physics/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)

</div>

<!-- RELATED:END -->
