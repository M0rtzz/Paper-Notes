---
title: >-
  [论文解读] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction
description: >-
  [AAAI 2026][3D视觉][前馈高斯溅射] 提出 Splat-SAP，一种前馈式方法，从大稀疏度的双目相机输入中重建尺度感知的点图（Point Map），并通过高斯平面（Gaussian Plane）实现人体中心场景的高质量自由视点渲染，全程无需3D监督。
tags:
  - AAAI 2026
  - 3D视觉
  - 前馈高斯溅射
  - 人体场景
  - 尺度感知
  - 点图重建
  - 自由视点渲染
---

# Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2511.22704](https://arxiv.org/abs/2511.22704)  
**代码**: [项目页面](https://yaourtb.github.io/Splat-SAP)  
**领域**: 3D视觉  
**关键词**: 前馈高斯溅射, 人体场景, 尺度感知, 点图重建, 自由视角视频

## 一句话总结

提出Splat-SAP，一种前馈式方法，从大间隔的双目相机输入中重建尺度感知的点图并通过高斯平面渲染人体中心场景的自由视角视频，无需逐场景优化且无需3D几何监督。

## 研究背景与动机

### 问题定义

前馈式自由视角视频合成是一个关键任务，尤其在稀疏视角设置下，可服务于远程通信、体育/舞台转播等众多下游应用。目标是从少量输入图像（如双目相机对）中实时合成任意视角的视频。

### 核心动机

现有前馈GS方法的**三大瓶颈**：

**大稀疏度下几何先验失效**：基于多视角立体（MVS）或双目立体匹配的方法（如MVSplat、GPS-Gaussian）需要输入图像有**大面积重叠**才能建立可靠的几何先验。当输入相机间距较大时，这些方法无法提供合理的几何先验。

**DUSt3R类方法的尺度不一致问题**：DUSt3R及后续方法（MASt3R）提出了点图表示，可以从极稀疏视角获取合理几何。但它们将点图归一化到**无尺度的规范空间**，导致连续帧之间重建结果**剧烈抖动**（jitter），无法用于自由视角视频。

**缺乏自监督训练方案**：训练尺度感知的几何基础模型通常需要大量3D数据，但采集3D几何数据既繁琐又昂贵。

### 核心洞察

- 点图的**尺度因子**与相机内参（焦距）和双目距离相关，可以通过嵌入这些信息来学习
- 两视角点图之间的不对齐可以通过**从粗到精的配准策略**解决——先2D仿射变换，再3D立体优化
- 在目标视角上直接锚定高斯基元（Gaussian Plane），比在两个源视角上分别建立高斯更高效

## 方法详解

### 整体框架

Splat-SAP由两个阶段组成：

**Stage 1（粗分辨率，512×288）**：
- 输入双目图像 → MASt3R预测规范空间点图 → 通过仿射变换转换到真实空间
- 仿射变换 = 缩放因子 $S$（学习）+ 逐像素平移 $T$（GRU迭代学习）

**Stage 2（精分辨率，1024×576 → 渲染1280×720）**：
- 将变换后的点图投影到目标视角 → 用3D代价体进行深度精细化 → 构建高斯平面 → 渲染

### 关键设计

#### 1. 尺度感知几何重建（Scale-Aware Geometry Reconstruction）

**功能**：将MASt3R输出的无尺度规范空间点图转换为真实空间中的尺度感知表示。

**缩放因子学习**：全局尺度因子 $S \in \mathbb{R}^3$ 与相机内参相关。通过嵌入焦距 $f$ 和双目距离 $d$：

$$e = \text{PE}(f, d)$$

然后处理MASt3R的ViT特征，通过自注意力和交叉注意力提取全局信息：

$$f_s = \text{Avg}(\text{Att}_s(\mathbf{Q}_l, \mathbf{K}_l, \mathbf{V}_l))$$
$$f_c = \text{Avg}(\text{Att}_c(\mathbf{Q}_l, \mathbf{K}_r, \mathbf{V}_r))$$
$$S = \text{MLP}(f_s, f_c, e)$$

缩放因子为3维，用于处理MASt3R原始点图重建的各方向畸变。

**逐像素平移学习**：缩放后的点图之间仍存在逐像素偏移（因MASt3R缺乏立体约束）。受MVS中视角一致性检查的启发，使用GRU迭代更新：

1. 将缩放后的点图 $SX^i$ 投影到另一视角 $j$，查询对应特征
2. 通过GRU融合两视角特征和当前点位置，计算逐像素平移：$T^i = \text{GRU}(F^i, F^{j \to i}, SX^i)$
3. 最终变换：$X_t^i = SX^i + T^i$

**设计动机**：缩放+平移共同构成从规范空间到真实空间的仿射变换。这种**学习式仿射**比DUSt3R的全局归一化更精确，且可以自监督训练。

#### 2. 通过高斯平面渲染（Rendering via Gaussian Plane）

**功能**：在目标视角建立一个高斯基元平面，减少冗余并提高渲染效率。

**3D精细化**：
1. 将变换后的点集通过α-blending投影到目标视角，得到初始深度图 $\mathcal{D}^k$
2. 在初始深度附近沿像素射线采样多个深度候选 $\{d_1, d_2, ..., d_N\}$
3. 从源视角warp特征到目标视角，构建3D代价体：$\Phi^k = \mathcal{E}_{3D}(F_f^{l \to k}, F_f^{r \to k})$
4. 回归深度概率分布 $w_n$，精细化深度：$\bar{d} = \Sigma_n w_n d_n$

**高斯平面构建**：确定位置后（精细化深度），构建四属性图：

$$\mathcal{G} = \{\mathcal{P}_c, \mathcal{P}_r, \mathcal{P}_s, \mathcal{P}_o\}$$

- 颜色：通过加权融合两个源视角warp过来的颜色初始化，再加学习的残差
- 旋转/缩放/不透明度：通过卷积头从融合的2D和3D特征中预测

$$\mathcal{P}_c = \alpha C + (1 - \alpha) \Delta C$$

其中 $\alpha = 0.8$，$\Delta C$ 为学习的残差颜色。

**设计动机**：
- 在**目标视角**建立高斯平面，而非在源视角（如GPS-Gaussian），避免了重叠区域的高斯冗余
- Stage 1的尺度感知点图为初始化提供了可靠的深度，大幅简化了Stage 2的训练
- 融合2D细粒度特征和3D代价体特征，克服了大稀疏度下的遮挡问题

#### 3. 自监督训练策略

**功能**：仅使用光度损失训练，不需要3D几何监督。

**Stage 1训练**：冻结MASt3R权重，训练仿射变换模块+辅助高斯平面

$$\mathcal{L}_{stage1} = \mathcal{L}_{render} + \gamma \mathcal{L}_{CD}$$

其中 $\mathcal{L}_{CD}$ 是两个6维点集（3D位置+RGB颜色）之间的Chamfer距离正则化，鼓励两视角点图收敛到一致的几何。

**Stage 2训练**：训练3D精细化模块和高斯平面

$$\mathcal{L}_{stage2} = \lambda_1 \mathcal{L}_{render}(\hat{I}_f, I_f^{gt}) + \lambda_2 \mathcal{L}_{render}(\hat{I}_h, I_h^{gt})$$

### 损失函数 / 训练策略

- 渲染损失：$\mathcal{L}_{render} = \beta_1 \mathcal{L}_1 + \beta_2 \mathcal{L}_{ssim}$，$\beta_1 = 0.8, \beta_2 = 0.2$
- Stage 1训练：100k迭代，混合所有相机类型的数据
- Stage 2训练：每种相机类型单独训练60k迭代
- 设备：单张RTX 3090 GPU (24GB)

## 实验关键数据

### 主实验

**渲染质量对比（多种相机类型）**：

| 方法 | Camera PSNR↑ | Camera SSIM↑ | GoPro PSNR↑ | GoPro SSIM↑ | Mobile PSNR↑ |
|---|---|---|---|---|---|
| NoPoSplat | 25.04 | 0.866 | 26.13 | 0.889 | 21.59 |
| 4D-GS | 27.81 | 0.906 | 27.24 | 0.907 | 25.66 |
| MVSplat | 27.90 | 0.902 | 29.94 | 0.934 | 26.55 |
| MVSGaussian | 29.33 | 0.957 | 27.41 | 0.926 | 19.93 |
| ENeRF | 28.27 | 0.943 | 29.91 | 0.943 | 20.58 |
| **Splat-SAP** | **32.22** | **0.957** | **31.64** | **0.955** | **25.72** |

**几何重建质量对比（Chamfer Distance）**：

| 方法 | Pred→GT↓ | GT→Pred↓ |
|---|---|---|
| DUSt3R | 0.305 | 0.160 |
| VGGT | 0.288 | 0.129 |
| MASt3R | 0.212 | 0.069 |
| Prompt-DA | 0.205 | 0.063 |
| Ours w/o Translation | 0.191 | 0.046 |
| **Full Model** | **0.172** | **0.027** |

### 消融实验

| 配置 | 说明 |
|---|---|
| Stage 1 only | 粗分辨率渲染，存在抖动和孔洞 |
| Stage 2 (完整) | 精细化深度消除了抖动，渲染质量大幅提升 |
| w/o Translation | 仅缩放无平移时几何偏差增大 |
| Full Affinity | 缩放+平移完整仿射变换最优 |
| w/o $\mathcal{L}_{CD}$ | 去除Chamfer约束后两视角点图不一致 |

### 关键发现

1. **Camera和GoPro数据集上大幅领先**：相比MVSGaussian在Camera PSNR上提升约3dB
2. **无3D监督仍优于有监督方法**：几何重建质量全面超越DUSt3R、MASt3R等使用大量3D数据训练的方法
3. **时序一致性强**：比4D-GS（逐场景优化方法）在快速运动下更好保持时间一致性
4. **平移学习至关重要**：移除平移模块后GT→Pred Chamfer距离从0.027增加至0.046
5. **MVSGaussian和ENeRF在大稀疏度下失效**：无法处理细长结构，出现缺失部分

## 亮点与洞察

1. **自监督尺度学习**非常优雅——通过嵌入相机内参和双目距离来学习全局缩放，不需要任何3D GT
2. **从粗到精的两阶段配准策略**：2D仿射（规范→真实空间）+ 3D代价体（深度精细化），每个阶段都有明确的功能划分
3. **高斯平面vs高斯点云**：在目标视角直接建立2D高斯平面，避免了源视角高斯的冗余和不对齐
4. **Chamfer距离正则化**：用6维（位置+颜色）Chamfer距离连接两视角点图，使几何和外观同时收敛
5. **实用性强**：一个模型适用于多种相机类型（工业相机、GoPro、手机），且仅需前馈推理

## 局限与展望

1. 每种相机类型需要单独训练Stage 2的精细化模块，通用性受限
2. 移动手机的交替变焦模式下表现相对较弱（25.72 vs Camera的32.22）
3. 大量运动导致的遮挡问题仍存在一定挑战
4. MASt3R的冻结权重限制了几何预测的上限
5. 目前仅处理双目输入，多视角扩展有待探索

## 相关工作与启发

- **DUSt3R/MASt3R**：点图表示的开创者，本文在此基础上引入尺度感知
- **GPS-Gaussian/GPS-Gaussian+**：在源视角建立高斯的前馈方法，本文改为目标视角高斯平面
- **MVSplat**：基于MVS代价体的前馈GS，但大稀疏度下失效
- **NoPoSplat**：同样利用MASt3R但抛弃了立体约束，导致视角不一致
- **ENeRF**：基于代价体的前馈NeRF方法，本文借鉴了其深度概率回归策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 尺度感知的自监督点图学习和目标视角高斯平面设计有新意
- **实验充分度**: ⭐⭐⭐⭐ — 多种相机类型、渲染和几何双评估，但缺少定量消融数值
- **写作质量**: ⭐⭐⭐⭐ — 层次清晰，但部分符号较多需要仔细阅读
- **价值**: ⭐⭐⭐⭐⭐ — 针对实际应用（人体自由视角视频），前馈无需优化，实用性极强
# Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2511.22704](https://arxiv.org/abs/2511.22704)  
**代码**: [项目页面](https://yaourtb.github.io/Splat-SAP)  
**领域**: 3D视觉  
**关键词**: 前馈高斯溅射, 人体场景, 尺度感知, 点图重建, 自由视点渲染

## 一句话总结

提出 Splat-SAP，一种前馈式方法，从大稀疏度的双目相机输入中重建尺度感知的点图（Point Map），并通过高斯平面（Gaussian Plane）实现人体中心场景的高质量自由视点渲染，全程无需3D监督。

## 研究背景与动机

前馈式自由视点视频合成在远程通信、体育转播等应用中至关重要。现有的前馈高斯溅射方法面临以下困境：

**困境一：大稀疏度输入下的几何失效**
- MVSplat、MVSGaussian等方法基于多视角立体匹配建立几何先验
- 这些方法要求输入视角有大量重叠区域
- 当两个输入相机间隔较大（大稀疏度）时，无法提供合理的几何先验

**困境二：DUSt3R系列的尺度不变性问题**
- DUSt3R/MASt3R提出了点图表示，能在大稀疏度下预测合理几何
- 但它们将点图归一化到尺度不变的规范空间
- 在连续帧推理时，不同帧的尺度归一化导致重建结果剧烈抖动
- 人体运动造成的深度变化会在规范空间中引发大的跳变

**困境三：3D监督数据的获取困难**
- 训练尺度感知的几何基础模型通常需要大量3D数据
- 3D几何数据的采集耗时且繁琐

Splat-SAP的核心贡献是通过自监督方式学习尺度感知的点图变换，将规范空间的点图映射到真实度量空间，无需任何3D几何监督。

## 方法详解

### 整体框架

两阶段的粗到精流程：
- **Stage 1（2D粗阶段）**：从MASt3R初始化的点图出发，学习仿射变换（缩放+平移）将其从规范空间变换到真实空间
- **Stage 2（3D精阶段）**：将变换后的点图投影到目标视角，通过3D代价体进行立体细化，构建高斯平面进行高质量渲染

### 关键设计

#### 1. **尺度感知几何重建（Scale-Aware Geometry Reconstruction）**：自监督的仿射变换学习

**点图初始化**：使用 MASt3R 从粗分辨率（512×288）的双目输入预测两个视角的点图 $X^i$（规范空间）。

**缩放因子学习**：
- 将相机内参焦距 $f$ 和双目距离 $d$ 通过位置编码嵌入
- 结合 ViT 特征的自注意力和交叉注意力全局信息
- 通过 MLP 预测3维缩放因子 $S$（处理原始点图的畸变）

$$S = MLP(f_s, f_c, e), \quad e = PE(f, d)$$

**逐像素平移学习**：
- 仅缩放无法消除两个点图之间的逐像素偏移
- 受MVS视角一致性检查启发，将一个视角的特征投影到另一个视角获取对应特征
- 使用 GRU 迭代计算逐像素平移：

$$T^i = GRU(F^i, F^{j \rightarrow i}, SX^i)$$

最终真实空间中的点位置：$X_t^i = SX^i + T^i$

**设计动机**：缩放（通过内参嵌入）+ 平移（通过外参投影）恰好构成从规范空间到真实空间的仿射变换。

#### 2. **高斯平面渲染（Gaussian Plane）**：高效且完整的渲染

**3D细化**：
- 将变换后的点集通过α-blending投影到目标视角获取初始深度图 $\mathcal{D}^k$
- 在初始深度附近沿相机射线采样多个深度候选
- 将源视角特征warp到目标视角构建3D代价体
- 通过3D卷积和深度概率回归得到精化深度 $\bar{d} = \Sigma_n w_n d_n$

**高斯平面构建**：
- 将高斯基元锚定在目标视角平面上，而非使用两个源视角的点图作为高斯位置
- 这大大减少了重叠区域的高斯冗余
- 颜色初始化：通过warp从源视角获取加权颜色
$$C^k = \Sigma_i w_c^i C^{i \rightarrow k}$$
- 其余属性（旋转、缩放、不透明度）通过卷积头从聚合特征预测
- 颜色残差学习：$\mathcal{P}_c = \alpha C + (1-\alpha) \Delta C$

最终在1024×576分辨率渲染，splatting输出1280×720高分辨率图像。

#### 3. **自监督训练策略**：无需3D几何监督

**Stage 1 损失**：
$$\mathcal{L}_{stage1} = \mathcal{L}_{render} + \gamma \mathcal{L}_{CD}$$

其中 $\mathcal{L}_{CD}$ 是两个6维点集（XYZ+RGB）之间的Chamfer距离正则化，促使两个点图收敛到一致的几何。训练时冻结MASt3R权重。

**Stage 2 损失**：
$$\mathcal{L}_{stage2} = \lambda_1 \mathcal{L}_{render}(\hat{I}_f, I_f^{gt}) + \lambda_2 \mathcal{L}_{render}(\hat{I}_h, I_h^{gt})$$

两个阶段均不需要3D几何监督，全部基于渲染损失训练。

### 损失函数 / 训练策略

- 渲染损失：$\mathcal{L}_{render} = 0.8 \mathcal{L}_1 + 0.2 \mathcal{L}_{ssim}$
- Stage 1：100k迭代训练仿射学习模块（使用全部训练数据）
- Stage 2：每种相机类型60k迭代训练渲染模块
- 单张 RTX 3090 (24GB) 即可训练

## 实验关键数据

### 主实验（渲染质量）

| 方法 | Camera PSNR↑ | Camera SSIM↑ | GoPro PSNR↑ | GoPro SSIM↑ | Mobile PSNR↑ | Mobile SSIM↑ |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| NoPoSplat | 25.035 | 0.866 | 26.128 | 0.889 | 21.594 | 0.591 |
| 4D-GS | 27.814 | 0.906 | 27.244 | 0.907 | 25.655 | 0.825 |
| MVSplat | 27.899 | 0.902 | 29.942 | 0.934 | 26.545 | 0.805 |
| MVSGaussian | 29.326 | 0.957 | 27.413 | 0.926 | 19.927 | 0.683 |
| ENeRF | 28.272 | 0.943 | 29.906 | 0.943 | 20.579 | 0.640 |
| **Splat-SAP** | **32.220** | **0.957** | **31.640** | **0.955** | **25.721** | **0.827** |

Camera和GoPro数据上PSNR大幅领先(+2.9和+1.7 dB)。

### 几何重建质量

| 方法 | Pred→GT CD↓ | GT→Pred CD↓ | 说明 |
|------|------------|------------|------|
| DUSt3R | 0.305 | 0.160 | 大量前景-背景误对齐 |
| VGGT | 0.288 | 0.129 | 两视图对齐困难 |
| Pow3R | 0.281 | 0.134 | 即使用相机标定也不够 |
| MASt3R | 0.212 | 0.069 | 基线几何 |
| Prompt-DA | 0.205 | 0.063 | 增加不确定性 |
| Ours w/o Translation | 0.191 | 0.046 | 仅缩放 |
| **Ours Full** | **0.172** | **0.027** | 缩放+平移 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| Stage 1 渲染 | 24.844 | 0.794 | 0.296 | 仅粗阶段辅助层渲染 |
| Stage 2 初始颜色 | 27.308 | 0.856 | 0.169 | 几何细化后的warp颜色 |
| **Stage 2 最终溅射** | **28.703** | **0.889** | **0.169** | 完整pipeline |

### 关键发现

1. 逐像素平移学习对消除点图对齐误差至关重要（Pred→GT CD从0.191降至0.172）
2. 3D细化模块可修正 Stage 1 中的空洞和伪影
3. 颜色残差学习和溅射机制进一步提升渲染质量
4. 在Mobile数据（交替变焦场景）上，方法仍能保持竞争力
5. 全自监督训练无需3D ground truth，但仍能超越需要3D监督的DUSt3R

## 亮点与洞察

1. **自监督尺度恢复**：巧妙利用相机内参嵌入和外参投影，无需3D监督即可学习从规范空间到真实空间的仿射变换
2. **高斯平面设计**：在目标视角平面上锚定高斯，避免了双源视角点图的冗余
3. **粗到精的几何策略**：2D仿射粗对齐→3D代价体精细化，逐步提升几何精度
4. **Chamfer距离正则化**：在6维空间（位置+颜色）上计算CD，同时约束几何和外观一致性
5. **实用的多相机支持**：一个仿射模块通用，每种相机类型只需训练一个渲染模块

## 局限与展望

1. **前景-背景边界浮点**：MASt3R可能在人物边界预测出浮点，由于该区域仅被一个视角观测到，细化模块无法修正
2. 仅支持双目输入，未探索多于两个视角的情况
3. 对MASt3R预训练模型的依赖较强
4. Mobile数据上与MVSplat差距较小，说明变焦场景仍有改进空间
5. 需要相机标定信息，限制了某些无标定场景的应用

## 相关工作与启发

- **DUSt3R/MASt3R**：点图表示的开创性工作，Splat-SAP在此基础上解决尺度问题
- **GPS-Gaussian/GPS-Gaussian+**：双目高斯的前身工作，但需要密集重叠
- **NoPoSplat/Splat3R**：利用点图进行静态场景渲染，但缺乏立体约束
- **ENeRF**：代价体+NeRF的前馈方法，Splat-SAP借鉴其深度概率回归
- **启发**：点图+立体匹配的组合可能是稀疏视点人体渲染的最佳方案

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 自监督尺度恢复和高斯平面设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 多种相机类型验证，渲染和几何双重评估
- **写作质量**: ⭐⭐⭐⭐ — 两阶段结构清晰，但部分细节需看补充材料
- **实用价值**: ⭐⭐⭐⭐⭐ — 对远程通信和体育转播等实时应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](../../ICLR2026/3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](../../CVPR2026/3d_vision/vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](../../CVPR2026/3d_vision/more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)

<!-- RELATED:END -->
