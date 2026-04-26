---
title: >-
  [论文解读] Hierarchical Material Recognition from Local Appearance
description: >-
  [ICCV 2025][3D视觉][材质识别] 提出面向视觉应用的层级式材质分类学体系(taxonomy)与野外数据集 Matador（含深度图的 ~7200 张材质图像，57类），并基于图注意力网络(GAT)利用分类学的层级亲缘关系进行材质识别，在多个基准数据集上达到 SOTA，同时支持新材质的小样本学习和场景中任意点的材质探测。
tags:
  - ICCV 2025
  - 3D视觉
  - 材质识别
  - 层级分类
  - 图注意力网络
  - 纹理识别
  - 材质分类体系
  - 新视角渲染
  - 小样本学习
---

# Hierarchical Material Recognition from Local Appearance

**会议**: ICCV 2025  
**arXiv**: [2505.22911](https://arxiv.org/abs/2505.22911)  
**代码**: [Matador 项目页](https://cave.cs.columbia.edu/repository/Matador)  
**领域**: 3d_vision  
**关键词**: 材质识别, 层级分类, 图注意力网络, 纹理识别, 材质分类体系, 新视角渲染, 小样本学习

## 一句话总结

提出面向视觉应用的层级式材质分类学体系(taxonomy)与野外数据集 Matador（含深度图的 ~7200 张材质图像，57类），并基于图注意力网络(GAT)利用分类学的层级亲缘关系进行材质识别，在多个基准数据集上达到 SOTA，同时支持新材质的小样本学习和场景中任意点的材质探测。

## 研究背景与动机

材质识别是计算机视觉中的基础问题，对自主系统与环境交互至关重要：

**材质识别的实际价值**：人类通过视觉就能推断材质的物理属性（如纸杯会烫手、陶瓷杯更重但更凉），机器也需要类似能力来进行更智能的环境交互。例如，自主机器人判断洒出的是液体后才会用毛巾而非扫帚清理

**粒度依赖应用**：不同应用需要不同粒度的材质识别——有时知道"液体"就够了，有时需要区分"咖啡"和"水"。因此材质识别天然具有层级结构需求

**现有方法局限**：传统纹理识别方法（滤波器组、CNN特征聚合等）将材质类别视为扁平集合，未利用材质间的物理亲缘关系（如橡胶⊂塑料⊂高分子）

**数据集不足**：现有材质数据集要么类别少（10-40类）、要么类内多样性差、要么缺少层级标注，无法支撑层级材质识别研究

**关键洞察**：即使在最细粒度上无法准确识别材质，在更高层级（如"形态"或"组成"）上的正确识别依然有用——可以推断力学属性等信息

## 方法详解

### 1. 材质分类学体系(Taxonomy)

受生物学"生命之树"启发，构建了覆盖57种常见材质的层级分类学体系：

- **词汇表构建**：从 M2D2 语料库（41.9亿词）中搜索 WordNet 中的材质名词出现频率，聚合同义词后保留57个最常用类别
- **层级结构**：按物理属性从上到下分为 Phase（相态）→ State（状态）→ Composition（组成）→ Form（形态）→ Material（材质）五个层级
- **实例**：Matter → Solid → Abiotic → Metal → Ferrous → {Iron, Steel}；Solid → Biotic → Natural → Vegetation → {Flower, Foliage, Ivy, Shrub}
- **力学属性表**：为每种叶节点材质汇编了密度、表面粗糙度、杨氏模量、屈服强度、抗拉强度等参数范围，使识别结果可直接关联到物理交互

### 2. Matador 数据集

构建了一个新的野外(in-the-wild)材质图像数据集：

- **规模**：~7200个样本，覆盖分类学中全部57种材质，平均每类126个实例
- **采集设备**：iPhone 15 Pro Max，自研 iOS 应用
- **每个样本包含**：
    - 局部外观图像（广角相机，12MP，74° FOV，12-bit raw）
    - 深度图（LiDAR，100点/度²）
    - 周围上下文图像（超广角，12MP，104° FOV）
    - IMU 运动数据和元数据
- **Matador-C1**：合并视觉相似类别（如所有金属合为一类）、去除纹理不足类别（如玻璃、涂料）后得到37类、~6600样本的子集

### 3. 基于深度图的新视角渲染

利用每个样本的深度图生成大量新视角训练数据：

- 从深度图创建3D网格，用外观图像做纹理映射
- 对网格施加空间变换（改变放大倍率和朝向）
- 通过光线追踪生成新视角的光学图像（薄透镜模型模拟景深/散焦效果）
- 模拟像素有效面积模糊 → 按像素间距采样 → 添加光子+传感器噪声
- 通过改变这些物理参数，为每个真实样本渲染大量不同相机、距离和角度的新视角图像

### 4. 图注意力网络(GAT)分类器

将分类学体系编码为有向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点为分类学中的类别，边为父子关系，使用 GAT 进行层级材质识别：

**节点初始化**：
- 对每个节点 $v_i$，用编码器 $\phi$（ResNet50）提取属于该节点的所有图像的平均特征：$\mathbf{h}_i^0 = \frac{1}{\|\mathcal{T}_i\|} \sum_{x_j \in \mathcal{T}_i} \phi(x_j)$

**图像分类**：
- 对输入图像 $x$，提取全局节点特征 $\mathbf{h}_g = \phi(x)$，插入图中并与所有节点相连
- GAT 消息传递更新：$\mathbf{h}_i^{k+1} = \psi_a(\mathbf{h}_i^k, \bigoplus_{j \in \mathcal{N}_i} (\alpha_{ij}^k \psi_b(\mathbf{h}_i^k, \mathbf{h}_j^k)))$
- 其中 $\alpha_{ij}^k$ 是边注意力权重，控制分类学节点间视觉特征共享的强度
- 级联 $D$ 层（匹配分类学直径）+ 残差连接

**损失函数**：
- 结合 BCE（鼓励学习完整层级路径）和逐层级 CE 的 max 损失：
$$\max\left(\text{BCE}(\hat{\ell}, \ell), \frac{1}{D}\sum_{d=0}^{D-1}\text{CE}(\hat{\ell}_d, \ell_d)\right)$$

### 5. 场景中材质探测

在推理时，给定图像中一个像素点，使用 64×64 到 1024×1024 的递增窗口，通过 Monte Carlo Dropout 构建预测分布，用最佳优先搜索在分类学树上得到层级一致的分类结果。

## 实验关键数据

### 标准基准（扁平分类，ResNet50 backbone）

| 方法 | KTH-2-b | FMD | GTOS | GTOS-M |
|------|---------|-----|------|--------|
| DeepTEN | — | — | — | —† |
| MAPNet | — | — | — | — |
| CLASSNet | — | — | — | — |
| RADAM | — | — | — | — |
| FRP | — | — | —† | —† |
| **Ours** | **最优** | **最优** | **最优** | **最优** |

在全部四个标准基准上均达到 SOTA。

### Matador 数据集性能

| 方法 | 参数量 | Matador | Matador-C1 | OOD |
|------|--------|---------|------------|-----|
| CLIP (zero-shot) | 151M | 24.8 | 40.0 | 32.3 |
| GPT-4.1 (zero-shot) | 1.76T | 51.4 | 65.9 | 53.4 |
| DeepTEN (ResNet50) | 24M | 79.2 | 88.8 | 61.5 |
| DEPNet (ResNet50) | 25M | 82.7 | 87.6 | 76.1 |
| ConvNeXt-V2 | 28M | 83.1 | 89.7 | 81.9 |
| **Ours (ResNet50)** | **28M** | **最优** | **最优** | **最优** |

- 新视角渲染数据增强在 OOD 测试集上提升最高达 **4.9%**
- 层级分类在更高层级（State → Composition → Form）的准确率逐级提升

### 小样本学习

- 仅用 **16个样本** 即可对从未见过的新材质类别达到 ~90% 准确率
- 此时平均路径距离不到 2 跳——即使误分类也落在分类学中的近邻节点

### 训练细节

- Backbone: ResNet50（IG-1B + ImageNet1k 预训练）
- GAT: 2层, 隐藏维 512, 输出维 256, 1个注意力头
- 训练: batch 400, 100 epoch, lr=1e-4, 余弦退火, 权重衰减 5e-4
- 总参数量: 28.0M; 单张 A6000 Ada 上训练不到 30 分钟（不用新视角时）

## 亮点与洞察

1. **物理驱动的分类学设计**：不同于 WordNet 等语义层级，该分类学按材质物理属性组织，使层级关系与力学性质直接对应，识别结果可推断密度、粗糙度、刚度等
2. **层级容错**：即使最细粒度分类错误（如羊毛→地毯），在更高层级仍能正确识别（如"纺织品"），这对实际应用极其实用
3. **深度图驱动的数据增强**：利用 LiDAR 深度图渲染新视角是本文独特贡献，不同于一般的几何增强，它物理地模拟了不同相机、距离和噪声条件下的成像过程
4. **模型效率**：仅 28M 参数即超越 1.76T 参数的 GPT-4.1 在材质识别上的表现，说明领域设计的重要性
5. **分类学与 GNN 的自然结合**：用图结构编码分类学亲缘关系让注意力机制自动学习哪些类别共享视觉特征，比手工设计的特征共享更灵活

## 局限与展望

1. **仅关注固体**：分类学目前只覆盖固体材质，液体和气体留待未来扩展
2. **纯纹理依赖**：玻璃、涂料等缺乏纹理的材质被排除在 Matador-C1 之外，未来需结合上下文和反射特性
3. **Lambertian 假设**：新视角渲染假设材质为朗伯体，未建模镜面反射和次表面散射，对金属等材质有偏差
4. **场景级推广**：论文展示了单点探测能力，但未实现全图材质分割；利用邻域像素抑制/增强分类是自然延伸
5. **跨域泛化**：数据集用单一手机型号采集，不同传感器/光照条件下的鲁棒性仍需验证
6. **颜色与灰度**：合并金属类别的原因是它们仅在色调上有差异，说明颜色信息的利用还有提升空间

## 相关工作与启发

- **纹理识别演进**：从 Gabor 滤波器组 → BoW/Texton → CNN 学习滤波器 → 多尺度特征聚合（如 FENet、CLASSNet、RADAM）→ 本文的层级图方法
- **层级图像分类**：利用 WordNet 等语义层级的方法（如 HGCLIP）主要用于物体识别，本文将类似思路引入材质识别并基于物理属性构建层级
- **GNN 在视觉中的应用**：图神经网络已用于零样本学习（知识图谱传播）和细粒度识别（关系发现），本文的贡献是将其与材质物理分类学结合
- **材质数据集**：KTH-TIPS2-b（受控环境BTF）、FMD（网络图像）、GTOS/GTOS-Mobile（地面地形），Matador 在类别数(57)、类内多样性(126实例/类)和数据类型(外观+深度+上下文)上全面超越
- **对启发**：层级分类学 + GNN 的范式可泛化到其他具有天然层级结构的识别任务（如生物物种、岩石矿物、织物面料），深度图驱动的新视角增强思路值得在其他 3D 感知任务中借鉴

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)
- [\[CVPR 2025\] On Denoising Walking Videos for Gait Recognition](../../CVPR2025/3d_vision/on_denoising_walking_videos_for_gait_recognition.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)

<!-- RELATED:END -->
