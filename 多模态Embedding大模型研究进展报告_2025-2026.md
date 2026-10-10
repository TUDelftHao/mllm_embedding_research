# 多模态 Embedding 大模型研究进展报告（2025–2026）

> 本报告系统梳理了 2025–2026 年间多模态 Embedding 大模型领域的研究进展。整合自两份既有文档（系统综述版 + 代表作深度解析版），并持续补充至 **2026 年 10 月初**的新工作（含微信 WeMM-Embedding 开放权重登顶、DME、UEmbed、UMER 成对判别推理、CORE 组合性蒸馏、PUMA 事后稀疏化、NeoMME 单塔编码器，以及 9 月以来的 Ovis-Embedding、MoEMB、EVIE、ColNanoVDR、ReCALL、TARA、MARS 等）。第 2–4 章以技术阅读 blog 风格展开数据集、评测与训练方法；**第 5 章按任务类别（通用嵌入、图文检索、组合检索、组合性、视觉文档、视频、全模态、推理、实例与个性化、效率、垂直领域）列出代表模型与最新进展**；**第 6 章汇总开源训练框架/仓库与后训练上手路径**。文中各工作名称均直接链接至论文 / 代码 / 官方页面。

---

## 目录

1. [引言与领域概览](#1-引言与领域概览)
2. [常用数据集](#2-常用数据集)
3. [评估方法与基准](#3-评估方法与基准)
4. [训练策略](#4-训练策略)
5. [代表模型与最新进展（按任务）](#5-代表模型与最新进展按任务)
6. [开源训练框架与后训练上手](#6-开源训练框架与后训练上手)
7. [趋势与展望](#7-趋势与展望)
8. [附录：两份原稿整合说明与新增文献](#8-附录两份原稿整合说明与新增文献)

---

## 1. 引言与领域概览

多模态 Embedding 模型旨在将不同模态（文本、图像、视频、音频等）的数据映射到统一的向量空间中，使得语义相似的跨模态数据在该空间中距离相近。这项技术是跨模态检索、多模态 RAG（检索增强生成）、视觉问答等应用的核心基础设施。

2025–2026 年，该领域经历了从**双塔架构（CLIP-style）到 MLLM-based 统一嵌入模型**的范式转变，并进一步向「全模态 + 推理增强嵌入」演进。关键趋势包括：

- **MLLM 成为主流骨干**：[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)、[LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) 等被广泛用作 Embedding 模型的基座
- **从双编码器到深度融合**：[Gemini Embedding 2](https://arxiv.org/abs/2605.27295) 等模型实现了任意交错的跨模态输入；开源侧 [Omni-Embed-Nemotron](https://arxiv.org/abs/2510.03458) / [OmniRet](https://arxiv.org/abs/2603.02098) 推进文–图–音–视频统一
- **工业「效率 × 细粒度」合流**：[WeMM-Embedding](https://arxiv.org/abs/2608.24053)（微信，开源 2B/4B/9B）把 MMEB-V2 总分推到 **80.6**；[DME](https://arxiv.org/abs/2608.02148)（抖音）用训练期潜在推理达到 **78.4**，推理仍接近普通对比编码器；[UEmbed](https://arxiv.org/abs/2608.02583) 一次前向同时产出稠密与稀疏向量
- **训练 / 推理效率继续下沉**：合成数据、GradCache、[Magic-MM](https://arxiv.org/abs/2602.05275) 视觉 token 压缩、[Eddy-VL](https://arxiv.org/abs/2607.16316) 结构剪枝蒸馏、[ReLoop-UME](https://arxiv.org/abs/2607.28751) 深度循环复用、[PUMA](https://arxiv.org/abs/2608.25780) 事后稀疏化、[NeoMME](https://arxiv.org/abs/2609.01657) 从零训的小参数单塔编码器
- **任务冲突与任务缩放**：[TSEmbed](https://arxiv.org/abs/2603.04772) 用 MoE-LoRA + Expert-Aware 难负例，把「多任务互相打架」显式拆开
- **推理时扩展维度（MRL）**：Matryoshka 表示学习成为标配，支持灵活的嵌入维度
- **推理能力成为新前沿**：[MR²-Bench](https://arxivlens.com/paperview/details/mr-2-bench-going-beyond-matching-to-reasoning-in-multimodal-retrieval-4166-ffb0e88d) 揭示匹配≠推理；[UME-R1](https://arxiv.org/abs/2511.00405) / [Embed-RL](https://arxiv.org/abs/2602.13823) / [MMEmb-R1](https://arxiv.org/abs/2604.06156) / [PLUME](https://arxiv.org/abs/2604.02073) / [RIME](https://arxiv.org/abs/2604.22280) / [UniME-R1](https://arxiv.org/abs/2608.06060)（检索反馈式 RC-CoT）/ [UMER](https://arxiv.org/abs/2608.18504)（成对判别推理 + 嵌入–重排统一）等探索显式 / 潜在 / 改写 / 反馈 / 成对推理
- **组合性回潮**：[CORE](https://arxiv.org/abs/2609.04083) 用五级匹配候选 + Rank-KL，把 reranker 的属性绑定判断蒸馏进双塔嵌入
- **2026 年 9 月以来**：[Ovis-Embedding](https://arxiv.org/abs/2609.25165) 自报 VL-9B MMEB-V2 **81.13**、Omni-3B 在 MMEB-V3 六个模态组全部第一；[MoEMB](https://arxiv.org/abs/2609.08663) 用专家扩容在公开数据口径下取得最好成绩；ViDoRe V3 榜首换成腾讯 [EVIE-8B](https://huggingface.co/tencent/EVIE-8B)（多向量 + 每页压到 32/64 个向量）

---



## 2. 常用数据集

<a id="2-常用数据集"></a>

> 本章用阅读 blog 的方式讲清：多模态 Embedding **用什么数据练、用什么数据考**。建议先建立「预训练 / 微调 / 评测」三分法，再按任务类型看具体数据集；每类都配了 Demo，方便对照论文里的样本格式。

### 2.0 先分清三类数据：预训练、微调、评测

入门时最容易把「训练集」和「榜单」混在一起。可以先记这张分工表：

| 阶段 | 数据长什么样 | 要它干什么 | 典型例子 |
|------|--------------|------------|----------|
| **预训练** | 海量、偏脏、弱监督图文/多模态对 | 把空间大致摆正：图文能对上号 | LAION-5B、DataComp、DFN |
| **微调 / 指令对齐** | 中等规模、任务格式清晰、常带 instruction | 学会分类、检索、VQA、组合检索等具体玩法 | MMEB 训练集、ShareGPT4V、CIRR |
| **评测基准** | 规模可控、标签干净、协议固定 | 公平比谁强；尽量少用来「刷分训练」 | MS-COCO 测试集、MMEB OOD、MR²-Bench |

**一条经验**：预训练看「量与覆盖」，微调看「任务多样性与难负例」，评测看「是否测到你真正关心的能力」（匹配 / 组合 / 推理 / 长上下文 / 文档）。

```text
网页图文海量对 ──► 预训练（对齐）──► 多任务微调（听指令）──► 各基准评测
     ↑                                    ↑
  LAION / DataComp / 合成数据         MMEB / CIRR / ViDoRe 训练子集
```

---

### 2.1 预训练数据集：先把「模态对齐」学出来

预训练几乎都是 **（图像/视频/…, 文本）弱监督配对**：来自网络 alt-text、维基说明、过滤后的爬虫语料。质量参差，但规模足够让对比学习找到粗对齐。

| 数据集 | 规模 | 模态 | 一句话定位 |
|--------|------|------|------------|
| **[LAION-5B](https://laion.ai/blog/laion-5b/)** | 58.5 亿对 | 图文 | 开源大规模图文「燃料库」，CLIP 系常用 |
| **[DataComp](https://www.datacomp.ai/)** | 128 亿对（含多档子集） | 图文 | 强调「数据筛选算法」本身可研究 |
| **[CC3M](https://ai.google.com/research/ConceptualCaptions/) / [CC12M](https://github.com/google-research-datasets/conceptual-12m)** | 300 万 / 1200 万 | 图文 | 更干净的概念描述，常作中小规模实验 |
| **[COYO-700M](https://github.com/kakaobrain/coyo-dataset)** | 7.47 亿对 | 图文 | 过滤更严的大规模开源替代 |
| **[WIT](https://github.com/google-research-datasets/wit)** | 3760 万 | 图文（多语言） | 维基图文，适合多语言对齐 |
| **[DFN](https://arxiv.org/abs/2309.17425)** | ~50 亿对（筛选后） | 图文 | Apple 路线：用过滤网络提纯数据 |
| **[VLDA-HM](https://github.com/PKU-YuanGroup/LanguageBind)** | 1000 万 | 视频/红外/深度/音频/文本 | 语言为锚的多模态绑定语料 |
| **[PolyGen](https://arxiv.org/abs/2602.01370) 合成** | 可扩展 | 图文 | 多生成器同题多图，2025 后很火 |

#### 怎么读这些预训练集？

**1. 规模不是唯一真相**  
LAION-5B 很大，但噪声也大（错配、水印、色情/低质内容需过滤）。DataComp / DFN 的核心叙事是：**同样算力下，更好的筛选 > 盲目堆数据**。读论文时看到 “DFN-5B” 之类名字，重点看它的过滤策略，而不只是 “5B” 这个数字。

**2. 弱监督配对长什么样**  
一张图配一句（或几句）不一定完美的描述。训练目标通常是：配对当正例，同 batch 其他样本当负例。

#### Demo：预训练图文对

**LAION / CC3M 风格（弱监督）**

```text
[Image]  一只金毛犬在草地上追飞盘，阳光从左侧斜照。
[Text]   "golden retriever catching a frisbee in a sunny park"

特点: 描述大体对，但不保证完整、不保证无错配；靠规模平均掉噪声。
```

**WIT 多语言（同一图像，多语说明）**

```text
[Image]  埃菲尔铁塔夜景
[en]     "The Eiffel Tower illuminated at night in Paris"
[zh]     "巴黎埃菲尔铁塔的夜景灯光"
[fr]     "La Tour Eiffel illuminée la nuit à Paris"

用途: 学「跨语言共享视觉概念」——同一景象，不同语言描述应对齐到相近向量。
```

**PolyGen 合成（同一 caption，多生成器图像 = 多正例）**

```text
[Caption]  "a red ceramic mug on a wooden table, soft morning light"
[Img-A]    SDXL-Turbo（偏写实）
[Img-B]    SANA（偏插画）
[Img-C]    SD2（偏摄影）
→ A/B/C 都是该 caption 的正例；还可程序化生成「差一点就不对」的硬负例。
```

**初学者选型提示**

| 你的目标 | 更合理的预训练数据起点 |
|----------|------------------------|
| 复现/小规模实验 | CC3M / CC12M |
| 追开源大规模基线 | LAION 过滤子集、COYO、DataComp Medium/Large |
| 强调数据质量 | DFN 思路或自建过滤 |
| 多语言 | WIT + 多语言合成描述 |
| 视频/更多模态 | VLDA-HM 及后续绑定类语料 |
| 缺真实数据 | PolyGen / mmE5 等合成流水线 |

---

### 2.2 微调与评估数据集：按「任务形态」来记

评测和微调数据按**输入输出形态**分类，比按名字死记更有用。

#### 2.2.1 图像–文本检索（经典双塔考场）

这是最老也最常用的设定：**文找图 / 图找文**。库里有很多图像（或很多句子），看正确答案能否排进 Top-K。

| 数据集 | 规模 | 特点 |
|--------|------|------|
| **[MS-COCO](https://cocodataset.org/)** | ~12.3 万图，每图 5 句 | 通用场景「标准尺子」，论文几乎必报 |
| **[Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)** | ~3.1 万图，每图 5 句 | 更偏日常相册风格，常与 COCO 并列 |
| **[ShareGPT4V](https://sharegpt4v.github.io/)** | ~10 万长描述 | GPT-4V 写的细粒度长 caption，利于长文本对齐 |
| **[Urban1K](https://huggingface.co/datasets/BeichenZhang/Urban1k)** | 1000 张 | 城市场景长描述检索（Long-CLIP 发布） |
| **[SugarCrepe](https://github.com/RAIVNLab/sugar-crepe)** | 自动构造硬负例 | 专门打「组合性」：属性绑错、关系颠倒 |

**任务在算法里怎么跑？**

```text
文→图:  Encode(text) 与  Encode(所有图) 算相似度，看真图是否进 Top-K
图→文:  Encode(image) 与  Encode(所有句) 算相似度，看真句是否进 Top-K
```

##### Demo：MS-COCO

```text
Image ID: 397133
[Image]  一名女子站在网球场边，手持球拍准备发球。

Captions（5 条都算正例）:
  1. "A woman with a tennis racket about to serve."
  2. "A female tennis player prepares to hit the ball."
  3. "Woman standing on a court holding a racket."
  4. "A tennis player in white preparing for a serve."
  5. "Athlete with racket on outdoor tennis court."

评测:
  Text→Image: 用 Caption-1 去图库里找 Image 397133
  Image→Text: 用图像去 caption 池里找上述正确描述之一
主指标: Recall@1 / @5 / @10
```

##### Demo：ShareGPT4V（长描述）

```text
[Image]  咖啡馆室内：木质吧台、吊灯、窗边两人交谈

[Long Caption]
"The photo shows a cozy café interior with warm wooden furniture.
A barista stands behind the counter preparing drinks, while two
customers sit by a large window with laptops. Soft pendant lights
hang from the ceiling, creating a relaxed afternoon atmosphere."

和 COCO 短句的差别: 更长、更细、更像「段落」；能暴露模型对长文本的退化。
```

##### Demo：SugarCrepe（组合性硬负例）

```text
[Image]  一只猫坐在蓝色椅子上

Positive:  "a cat sitting on a blue chair"
Hard Neg:  "a blue cat sitting on a chair"   ← 颜色绑错对象
Hard Neg:  "a chair sitting on a blue cat"   ← 主客体颠倒

意义: 袋袋词（bag-of-words）模型容易被骗；真正理解组合关系才过关。
```

---

#### 2.2.2 组合图像检索（Composed Image Retrieval）

用户说的不是纯文本，而是：**参考图 + 修改说明 → 目标图**。更接近电商「找同款但换个颜色/款式」。

| 数据集 | 规模 | 领域 |
|--------|------|------|
| **[CIRR](https://github.com/Cuberick-Orion/CIRR)** | ~2.1 万查询 | 真实场景开放域 |
| **[FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)** | ~7.7 万查询 | 服装属性修改 |
| **[NIGHTS](https://github.com/princetonvisualai/nights)** | ~2 万图像对 | 人类感知相似度比较 |

查询通常要**联合编码**参考图与修改文本（或先分别编码再融合）：

```text
q = Encode(Reference Image, Modification Text)
在候选图库中检索 Target Image
```

##### Demo：CIRR

```text
[Reference]     一只棕色泰迪犬站在草地上
[Modification]  "same dog, but sitting and wearing a red scarf"
[Target]        同一只狗坐下并围红围巾

难点: 要保留「是同一只狗」的身份，又要执行「坐下 + 红围巾」的编辑。
```

##### Demo：FashionIQ

```text
[Reference]  黑色圆领短袖 T 恤
[Caption]    "is floral and has short sleeves"
[Target]     花卉印花短袖上衣

难点: 服装属性组合（领型、袖长、图案、材质）细且易混淆。
```

##### Demo：NIGHTS（哪张更像？）

```text
[Anchor]   日落海边剪影
[Image A]  同一场景、曝光略不同     ← 人类判为更相似
[Image B]  完全不同的雪山
标签: A 比 B 更接近 Anchor

用途: 学「人眼感知相似度」，而不只是像素级或标签级匹配。
```

---

#### 2.2.3 知识密集 VQA 与视觉文档

这类数据测的是：**能不能对着图/页面找到信息**，而不只是「图文像不像」。在 Embedding 设定里，常把「问题」编成 query，把「含答案的段落/页面/图像」编成候选文档。

| 数据集 | 测什么 |
|--------|--------|
| **[WebQA](https://webqna.github.io/)** | 多模态网页证据问答 |
| **[DocVQA](https://www.docvqa.org/)** | 文档页上的问答（表格、发票、表单） |
| **[InfoVQA](https://www.docvqa.org/datasets/infographicvqa)** | 信息图（饼图、流程、海报）问答 |
| **[VisualNews](https://github.com/FuxiaoLiu/VisualNews-Repository)** | 新闻图文，偏检索/事实相关 |
| **[ViDoRe](https://huggingface.co/vidore)** | 视觉文档检索专项（V1/V2/V3） |

##### Demo：DocVQA

```text
[Page Image]  扫描发票 PDF 页

Q: "What is the total amount due?"
A: "$1,245.00"

Q: "Who is the vendor?"
A: "Acme Supplies Ltd."

Embedding 视角: query=问题（可含页图），正例=含答案的那一页或那段区域表示。
```

##### Demo：InfoVQA

```text
[Infographic]  "2024 Global EV Market Share" 饼图+柱状图
Q: "Which country had the highest EV market share in 2024?"
A: "China"

难点: 要读懂版式与图例，不是只做自然照片描述。
```

##### Demo：ViDoRe（页级检索）

```text
[Query]   "What is the attention mechanism used in Transformer?"
[Corpus]  大量论文/手册的【页面截图】（常不先 OCR）
[Relevant] Attention Is All You Need 里含 Multi-Head Attention 图的那一页

评分: ColPali/ColQwen 等多向量 MaxSim；主指标常见 NDCG@10
```

##### Demo：WebQA

```text
[Question]  "What color is the dome of the Florence Cathedral?"
[Evidence]  维基页面截图 / 相关图片
[Answer]    "The dome is covered in red terracotta tiles."

特点: 答案依赖外部多模态证据，接近「检索增强」场景。
```

---

#### 2.2.4 MMEB：把异构任务「翻译」成同一种嵌入作业

[MMEB](https://www.emergentmind.com/topics/massive-multimodal-embedding-benchmark-mmeb)（随 [VLM2Vec](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html) 提出）的关键思想是：

> 分类、VQA、检索、Grounding……表面不同，但都可以写成  
> **`Instruction + Query → Embedding`，去靠近一个 Target Embedding**。

| 元任务 | 大约多少套数据 | 代表 |
|--------|----------------|------|
| 分类 Classification | ~10 | ImageNet-1K, SUN397, Food101, CIFAR-100 |
| 视觉问答 VQA | ~6 | OK-VQA, A-OKVQA, ScienceQA, TextVQA |
| 跨模态检索 Retrieval | ~8 | MS-COCO, NIGHTS, VisualNews-t2i |
| 视觉定位 Grounding | ~6 | RefCOCO, RefCOCO+, MSCOCO-Grounding |

##### Demo：统一成「指令条件化」格式

```text
# 分类 —— Target 是类别名文本
Instruction: "Represent the given image for classification"
Query:       <寿司拼盘图像>
Target:      "sushi"

# VQA —— Target 是能回答问题的文本
Instruction: "Find a passage that answers the question"
Query:       <图像> + "How many people are in the photo?"
Target:      "There are three people standing near the fountain."

# 检索 —— Target 是匹配图像（或其表示）
Instruction: "Retrieve an image that matches the given caption"
Query:       "a dog running on the beach at sunset"
Target:      <匹配图像>

# Grounding —— Target 是区域/区域描述
Instruction: "Represent the image region described by the caption"
Query:       <图像> + "the red car on the left"
Target:      <对应区域或描述>
```

**为什么这对初学者重要？**  
一旦接受这种格式，你读 VLM2Vec / UniME / Qwen3-VL-Embedding 时，就不会被「他们怎么把分类也做成对比学习」卡住——**类别名、答案句、目标图，都只是另一种「正例文档」**。

---

### 2.3 章节小结：数据怎么选

1. **先定能力再选数据**：只要图文召回 → COCO/Flickr；要改图检索 → CIRR/FashionIQ；要 PDF → ViDoRe/DocVQA；要通用多任务 → MMEB。  
2. **训练别只喂评测集**：会过拟合榜单；预训练用大而滤噪的语料，微调用任务混合。  
3. **Demo 里的「Hard Neg」值得收藏**：组合性、假负例、文档版式，是 2025–2026 拉开差距的地方。

---

## 3. 评估方法与基准

<a id="3-评估方法与基准"></a>

> 有了数据还不够——**怎么打分、考什么能力**，决定你会被榜单带着走，还是真正理解模型强弱。本章先讲指标直觉，再带读各大基准，最后总结 2025–2026 的评测共识。

### 3.0 评测在回答哪三个问题？

1. **找得到吗？**（召回）——正确答案有没有进 Top-K  
2. **排得准吗？**（排序）——越相关是否越靠前  
3. **泛化到哪？**（分布外 / 推理 / 长上下文）——换任务、换难度是否崩盘

Embedding 评测几乎都围着这三问转；不同基准只是把考卷出得更偏某一问。

---

### 3.1 核心评估指标：用「搜狗」例子建立直觉

假设库里有 1000 张图，其中只有 1 张是正确答案。模型给出一个排序列表。

| 指标 | 含义（白话） | 常见设置 | 什么时候看它 |
|------|--------------|----------|--------------|
| **Recall@K** | Top-K 里有没有答案（有则 1，无则 0；多正例则算比例） | K=1,5,10,50 | 检索主指标；COCO 常报 R@1/5/10 |
| **Precision@K** | Top-K 里有多少是对的 | MMEB 常用 **P@1** | 每个 query 通常一个硬正例时，P@1≈R@1 |
| **nDCG@K** | 排序质量：相关的越靠前越好，且有位置折损 | K=10 很常见 | 多档相关性、文档检索（ViDoRe） |
| **MRR** | 第一个正确答案排名的倒数的平均 | — | 很在意「第一条就对」 |
| **mAP** | 多正例时的精度–召回综合 | 多标签 | 一图多标注等场景 |

#### 小计算器：Recall@K 与 MRR

```text
正确文档排在第 3 名:
  Recall@1 = 0
  Recall@5 = 1
  MRR      = 1/3 ≈ 0.33

正确文档排在第 1 名:
  Recall@1 = 1
  MRR      = 1.0
```

#### nDCG 为什么「更严」？

它不只问「进没进 Top-K」，还问「是不是排得靠前、相关性等级对不对」。两个模型都能 R@10=1，但一个把正解排第 1、一个排第 10，nDCG 会拉开差距——所以 **ViDoRe / 文档检索更爱 nDCG@10**。

#### 读榜单时的坑

- **K 不同不能横比**：R@1 和 R@50 差一个数量级很正常。  
- **候选库规模不同不能横比**：库从 5K 扩到 5M，同一模型 R@1 会掉。  
- **In-distribution vs OOD**：在训练见过的数据集上很高、换一套就掉，说明泛化虚高。

---

### 3.2 主要基准平台：一张「考纲」地图

```text
                    ┌─ 匹配能力广不广 ──► MMEB / M-BEIR / MTEB v2
评测光谱 ──────────┼─ 文档页找不找得到 ──► ViDoRe / CMDR-Bench（跨页上下文）
                    ├─ 组合绑得对不对 ──► Winoground / SugarCrepe
                    ├─ 实例身份认不认 ──► MVEB（视觉身份）
                    ├─ 会不会推理 ──► MR²-Bench
                    └─ 长上下文崩不崩 ──► MMLongEmbed
```

---

#### [MMEB](https://www.emergentmind.com/topics/massive-multimodal-embedding-benchmark-mmeb)：多模态嵌入的「高考总分」

| 版本 | 来源 | 规模印象 | 新增什么 |
|------|------|----------|----------|
| **V1** | [VLM2Vec](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html), ICLR 2025 | 36 数据集，4 元任务 | 分类 / VQA / 检索 / Grounding；20 ID + 16 OOD |
| **V2** | [VLM2Vec-V2](https://huggingface.co/papers/2507.04590), 2025.07 | ~78 任务 | 视频检索/定位/分类/问答、视觉文档 |
| **V3** | [MMEB-V3](https://arxiv.org/abs/2604.23321) (COLM 2026) | ~190 任务 | 音频、Agent（工具/GUI/记忆）；配套 **OmniSET** 诊断集 |

**主指标**：Precision@1 / Recall@1（多数子任务是「一查询一正例」）；VisDoc 常用 nDCG@5。

**实时榜**：[TIGER-Lab MMEB Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)（提交持续刷新）。

**MMEB-V3 / OmniSET 要点（2026.04）**  
MMEB-V3 把评测推进到「全模态 + Agent」。配套的 [OmniSET](https://arxiv.org/abs/2604.23321)（Omni-modality Semantic Equivalence Tuples）用跨模态语义等价元组，把「语义相似」与「模态效应」拆开诊断。论文三点结论值得记住：

1. 模型常检索不到指令指定的目标模态；  
2. 跨模态检索高度不对称，受 query 模态偏置主导；  
3. 指令诱导的表示偏移常常不够或方向不对，**不能可靠地做模态约束检索**。

这说明：仅扩大模态覆盖 ≠ 学会「听指令选模态」。

**MMEB-V2 Top（Overall / All，公开对照快照）**

> 来源综合：[Ovis-Embedding](https://arxiv.org/abs/2609.25165)（2026-09 论文）、[WeMM-Embedding 技术报告](https://arxiv.org/abs/2608.24053)（2026-08-24）、[DME](https://arxiv.org/abs/2608.02148)、[Qwen3-VL-Embedding-8B 模型卡](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)、[UEmbed](https://arxiv.org/abs/2608.02583) 与 [MMEB Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)（**2026-10** 抽查）。分数为 **All（78 任务平均）**；论文自报与活榜分开标注，活榜会继续漂移。工业私有数据模型与纯公开数据模型请分开读。

| 排名 | 模型 | Size | All | 备注 |
|------|------|------|-----|------|
| 1 | **[Ovis-Embedding-VL-9B](https://arxiv.org/abs/2609.25165)** | 9B | **81.13** | 论文自报，称将开源；Image 83.96 / Video 72.90 / VisDoc 83.06；对照表未含 WeMM-9B |
| 2 | **[WeMM-Embedding-9B](https://arxiv.org/abs/2608.24053)** | 9B | **80.6** | 微信开源系列；Image 81.9 / Video 74.3 / VisDoc 83.3 |
| 3 | DME-Large | — | **80.24** | 活榜提交；Image 81.11 / Video 74.36 / VisDoc 83.36 |
| 4 | octen-vl-embedding-large | — | **80.09** | 活榜提交，无技术报告；Video **75.95** 为榜上视频最高 |
| 5 | **[WeMM-Embedding-4B](https://huggingface.co/tencent/WeMM-Embedding-4B)** | 4B | **79.2** | 已超过多数 8–9B 对照 |
| 6 | **[DME-9B](https://arxiv.org/abs/2608.02148)**（活榜 DME-Medium） | 9B | **78.4** | 抖音工业模型；训练期潜在推理，线上仍近对比编码器 |
| 7 | QQMM-embed-v4 | — | 78.31 | 活榜提交；Image 82.02 |
| 8 | **[WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)** | 2B | **77.9** | 2B 即超过此前 8B 开放权重标杆 |
| 9 | [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | **77.8–77.9** | 仍是常用对照；Image≈80 / Video≈67 / VisDoc≈82 |
| 10 | [Ovis-Embedding-VL-2B](https://arxiv.org/abs/2609.25165) | 2B | 77.46 | 论文自报 |
| 11 | [Seed-1.6-embedding-1215](https://seed.bytedance.com/en/blog/built-on-seed1-6-flash-seed-1-6-embedding-launched) | — | 76.97 | 闭源 / 半开源 |
| 12 | **[DME-2B](https://arxiv.org/abs/2608.02148)** | 2B | **74.8** | 同档 2B 很强；视频 / VisDoc 尤其突出 |
| 13 | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 2B | 73.3–73.4 | 小模型高性价比开放权重（后训练常用基座） |
| 14 | [UEmbed-9B](https://arxiv.org/abs/2608.02583)（dense） | 9B | **71.8** | 公开数据训练；同模型 sparse **71.0** |
| 15 | [RzenEmbed](https://arxiv.org/abs/2510.27350) / TTE / 旧版 WeMM-8B 等 | 4–9B | ~71–74 | 方法基线或旧快照；勿与新 WeMM 系列混读 |

> 公开数据口径另看 [MoEMB](https://arxiv.org/abs/2609.08663)（2026-09）：3B 激活参数，在只用公开 MMEB 系数据训练的模型里 MMEB-V2 与 MRMR 均为最好。

**怎么读这个榜？**  
总分高说明「通用匹配」强，但不自动等于：视频第一、文档第一、推理第一。看子任务表往往比看总分更有决策价值（例如视频看 [WAVE-7B](https://huggingface.co/tsinghua-ee/WAVE-7B) / WeMM / DME，文档看 ColQwen / [Nemotron ColEmbed V2](https://arxiv.org/abs/2602.03992) / [NeoMME](https://arxiv.org/abs/2609.01657) / Argus）。**DME 用私有工业数据**，[WeMM](https://arxiv.org/abs/2608.24053) 亦含站内数据与合成，与 [UEmbed](https://arxiv.org/abs/2608.02583) 等「只训公开数据」路线不可直接横比「谁更强算法」。活榜上 **octen-vl-embedding-large**、DME-Large 等提交在 80 附近，Ovis 的 81.13 目前是论文自报，开源后应以复测为准。按任务看的代表模型与最新进展见 [第 5 章](#5-代表模型与最新进展按任务)。

---

#### [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)：偏「检索系统」的综合考场

- **规模**：10 个数据集、16 个检索任务、约 **560 万** 候选项  
- **覆盖**：图文检索、组合检索、知识密集检索等  
- **指标**：全局池上平均 **R@5**（Fashion200K / FashionIQ 用 R@10）  
- **基准主页**：[UniIR / M-BEIR](https://tiger-ai-lab.github.io/UniIR/) · **代表工作**：[MM-Embed](https://openreview.net/forum?id=vG8Mmp4OCP)

**M-BEIR Top 10（Global pool Avg R@5，论文报告）**

> 来源：[MM-Embed](https://arxiv.org/html/2411.02571) Table 1 与 [UniIR](https://tiger-ai-lab.github.io/UniIR/)；设置为从 **5.6M 异构候选池**检索。无持续公开排行榜时，以论文横向对比为准。

| 排名 | 模型 | 类型 | Avg R@5 |
|------|------|------|---------|
| 1 | [MM-Embed](https://openreview.net/forum?id=vG8Mmp4OCP) | MLLM bi-encoder + 持续文本微调 | **52.7** |
| 2 | Mhard(NVEmb) | NV-Embed 系 + 模态感知硬负例 | 52.3 |
| 3 | Mhard(LLaVa-P) | LLaVA-Next + 硬负例 | 51.9 |
| 4 | [UniIR](https://tiger-ai-lab.github.io/UniIR/) (CLIPSF) | CLIP 特征融合 + 指令多任务 | 50.6 |
| 5 | Mhard(CLIPSF) | CLIPSF + 硬负例 | 48.3 |
| 5 | Mrand(LLaVa-P) | LLaVA-Next + 随机负例 | 48.3 |
| 7 | Mrand(LLaVa-E) | LLaVA-Next（E 池化） | 47.9 |
| 8 | Mrand(CLIPSF) | CLIPSF + 随机负例 | 47.4 |
| 9 | Mrand(NVEmb) | NV-Embed 微调（随机负例） | 47.2 |
| 10 | [UniIR](https://tiger-ai-lab.github.io/UniIR/) (BLIPFF) / BLIPFF | BLIP 系多任务 | ~45.5–46.8 |

**和 MMEB 的差别（直觉）**：MMEB 更像「嵌入表示全能测试」；M-BEIR 更像「检索器在大型候选库上的实战」。

---

#### [UMRB](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Bridging_Modalities_Improving_Universal_Multimodal_Retrieval_by_Multimodal_Large_Language_CVPR_2025_paper.html)：通用多模态检索（含融合模态）

- **来源**：[GME](https://arxiv.org/abs/2412.16855)（CVPR 2025）汇集约 **47** 个检索数据集  
- **覆盖**：text↔text、image↔image、跨模态，以及 **fused-modal**（图文复合 query/candidate）  
- **指标**：子任务用 nDCG@10 / nDCG@5 / R@5/R@10，Overall 为异构汇总  
- **读点**：公开数据几乎只有 text↔image，融合模态稀缺——这正是 GME 合成 110 万 fused-modal 数据的动机

---

#### [MR²-Bench](https://arxivlens.com/paperview/details/mr-2-bench-going-beyond-matching-to-reasoning-in-multimodal-retrieval-4166-ffb0e88d)：从「长得像」到「想得通」

- **定位**：首个强调**推理密集**的多模态检索基准（约 1,309 条精标查询）  
- **考什么**：逻辑、空间、因果等——仅靠浅层外观相似度不够  
- **关键数字**（务必记住数量级）：  
  - 强模型在 MMEB 上可到 **~77.8 R@1**  
  - 同一时代模型在 MR² 上可掉到 **~9.9 R@1**（约 **68 个点**的鸿沟）  
  - 最佳约 **30.68 nDCG@10**；加 GPT-5 重排后约 **45.90**  
- **官方 Mini-Leaderboard**：[VectorSpaceLab/MR2-Bench](https://github.com/VectorSpaceLab/MR2-Bench)

**MR²-Bench Top 10（Avg nDCG@10，官方 Mini-Leaderboard）**

> 指标为 12 个子任务宏平均 nDCG@10。表中同时收录「原生多模态嵌入」与「文本嵌入 + 图像 caption」两条路线，便于对照。

| 排名 | 模型 | 类型 | Avg nDCG@10 |
|------|------|------|-------------|
| 1 | [Seed-1.6-Embedding](https://seed.bytedance.com/en/blog/built-on-seed1-6-flash-seed-1-6-embedding-launched) | 多模态嵌入 | **30.68** |
| 2 | [MM-Embed](https://openreview.net/forum?id=vG8Mmp4OCP) | 多模态嵌入 | 30.23 |
| 3 | ReasonIR + Captions | 文本推理检索器 + 图注 | 25.72 |
| 4 | BGE-Reasoner + Captions | 文本推理检索器 + 图注 | 25.35 |
| 5 | [VLM2Vec-v2](https://huggingface.co/papers/2507.04590) | 多模态嵌入 | 23.72 |
| 6 | Diver-Embed + Captions | 文本推理检索器 + 图注 | 23.59 |
| 7 | [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) | 多模态嵌入 | 21.59 |
| 8 | Qwen3 + Captions | 文本嵌入 + 图注 | 20.17 |
| 9 | BGE-VL | 多模态嵌入 | 19.53 |
| 10 | CLIP / BGE-M3 + Captions | 基线 | ~18.6–18.7 |

##### Demo：需要因果推理的检索

```text
[Query Image]  水杯倾倒，水流向桌边
[Query Text]   "Which image shows what happens next due to gravity?"
[Candidates]
  A. 水落在地板上形成水洼     ← 正解（物理因果）
  B. 水杯被扶正且无水渍       ← 外观相关但违因果
  C. 干净空桌面               ← 更像「结果被擦掉」

浅层匹配模型易选 B/C；MR² 专门惩罚这种「像但不对」。
```

**读完带走**：只报 MMEB 的论文，可能高估了推理检索能力；MR² 是 2025–2026 最刺痛人的镜子。

---

#### [MMLongEmbed](https://arxiv.org/abs/2606.14747)：上下文一长就退化吗？

- **定位**：长上下文多模态 Embedding（可到 **32K** token 量级）  
- **四大任务**：Needle-in-a-Haystack（视觉定位）、多源推理、逻辑摘要、时序摘要  
- **实验规模**：约 11 个模型（开源为主 + 闭源对照）  

**MMLongEmbed Top（Avg P@1，论文 Table 3）**

> 来源：[MMLongEmbed](https://arxiv.org/html/2606.14747) Table 3 全任务平均 Precision@1。官方评测集规模有限，下表按论文报告分数排序（不足 10 个时如实列出）。

| 排名 | 模型 | Size | Avg P@1 |
|------|------|------|---------|
| 1 | [Ops-MM-embedding-v1-7B](https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-7B) | ~8B | **53.90** |
| 2 | [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | 53.41 |
| 3 | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 2B | 52.31 |
| 4 | Doubao-Seed-1.6（闭源） | — | 52.11 |
| 5 | [Ops-MM-embedding-v1-2B](https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B) | 2B | 48.18 |
| 6 | [Embed-RL-4B](https://huggingface.co/ZoengHouNaam/Embed-RL-4B) | 4B | 40.84 |
| 7 | [Embed-RL-2B](https://huggingface.co/ZoengHouNaam/Embed-RL-2B) | 2B | 39.28 |
| 8 | [GME-Qwen2-VL-7B](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) | 7B | 24.63 |
| 9 | [GME-Qwen2-VL-2B](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) | 2B | 23.10 |

##### Demo：多模态「大海捞针」

```text
[Context]  拼接约 50 页幻灯片截图（16K–32K tokens）
[Needle]   藏在第 37 页: "Project budget is $2.4M"
[Query]    "What is the project budget mentioned in the slides?"

考察: 超长交错图文里，关键信息是否仍能被嵌入/检索定位。
```

**读完带走**：长上下文不是「上下文窗口写得大就行」；嵌入质量常随长度**可预测地下滑**。同系列里 8B 相对 2B 提升往往很小（甚至在 Hard 子集更差），说明**训练策略与上下文建模**比单纯堆参数更关键。

---

#### [ViDoRe](https://huggingface.co/vidore)：视觉文档检索专项

- 为 [ColPali](https://huggingface.co/vidore/colpali) / [ColQwen](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) 等「页当图像、多向量打分」路径服务  
- **V1/V2**：ArxivQA、DocVQA、InfoVQA 等多文档类型  
- **V3（2026）**：多语言；主指标 **NDCG@10**  
- **实时榜**：[ViDoRe V3 · MTEB](https://mteb-leaderboard.hf.space/benchmark/ViDoRe(v3)) / [vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)

**ViDoRe V3 前列（Mean Task，2026-10-08 快照）**

> 来源：第三方聚合页 [LLMBoard · ViDoRe v3 Overall](https://www.llmboard.ai/benchmarks/vidore-v3-overall-710)（汇总 MTEB 榜结果）；以 [ViDoRe V3 · MTEB](https://leaderboard.mteb.org/benchmark/ViDoRe(v3)) 官方页为准。

| 排名 | 模型 | Mean (Task) |
|------|------|-------------|
| 1 | [EVIE-8B](https://huggingface.co/tencent/EVIE-8B)（腾讯） | **66.24** |
| 2 | [EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B)（腾讯） | 65.70 |
| 3 | webAI-ColVec1.1-8b | 64.95 |
| 4 | VultronRetrieverPrime-Qwen3.5-8B | 64.26 |
| 5 | webAI-ColVec1.1-4b | 63.90 |
| 6 | VultronRetrieverCore-Qwen3.5-4.5B | 63.57 |

半年内榜首从 63.42 升到 66.24。EVIE 目前只有模型卡和开源代码，正式论文未发布；[ColQwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) 在同一快照里约 44.65，代差明显。

**ViDoRe V3 Top 10（Mean Task / nDCG@10，2026-04-20 快照，保留作对照）**

> 来源：[colqwen3.5-4.5B-v3 模型卡](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) 捕获的公开榜；活榜会随新提交变化。

| 排名 | 模型 | Params | Mean (Task) |
|------|------|--------|-------------|
| 1 | [nemotron-colembed-vl-8b-v2](https://arxiv.org/html/2602.03992v2) | 8.7B | **63.42** |
| 2 | webAI-ColVec1-9b | 9.4B | 63.00 |
| 3 | webAI-ColVec1-4b | 4.5B | 62.22 |
| 4 | tomoro-colqwen3-embed-8b | 8.0B | 61.59 |
| 5 | nemotron-colembed-vl-4b-v2 | 4.8B | 61.54 |
| 6 | [colqwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) | 4.6B | 61.46 |
| 7 | [Ops-Colqwen3-4B](https://huggingface.co/OpenSearch-AI) | 4.8B | 61.17 |
| 8 | tomoro-colqwen3-embed-4b | 4.0B | 60.20 |
| 9 | llama-nemotron-colembed-vl-3b-v2 | 4.4B | 59.79 |
| 10 | [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 3.9B | 57.52 |

**ViDoRe V1+V2 Top 10（Avg nDCG@5，同快照）**

| 排名 | 模型 | Avg |
|------|------|-----|
| 1 | Ops-Colqwen3-4B | **84.9** |
| 2 | nemotron-colembed-vl-8b-v2 | 84.8 |
| 3 | nemotron-colembed-vl-4b-v2 | 83.9 |
| 4 | colqwen3.5-4.5B-v3 | 83.7 |
| 5 | llama-nemotron-colembed-vl-3b-v2 | 83.6 |
| 6 | tomoro-colqwen3-embed-8b | 83.5 |
| 7 | EvoQwen2.5-VL-Retriever-7B-v1 | 83.4 |
| 8 | tomoro-colqwen3-embed-4b | 83.2 |
| 9 | llama-nemoretriever-colembed-3b-v1 | 83.1 |
| 10 | SauerkrautLM-ColQwen3-8b-v0.1 | 82.9 |

> 另：[Argus-9B](https://arxiv.org/abs/2606.04300) 在 V1+V2 上报告 **86.0 NDCG@5**（OOD V2 提升明显），是否已完整入库活榜需以 [MTEB ViDoRe](https://mteb-leaderboard.hf.space/benchmark/ViDoRe(v3)) 为准。

**和 DocVQA 的关系**：DocVQA 偏「给定页问答」；ViDoRe 偏「先在很多页里找到那一页」。做 PDF RAG，ViDoRe 往往比 COCO 更相关。

**延伸：跨页上下文文档检索（CMDR，2026.07）**  
[CMDR / CMDR-Bench](https://arxiv.org/abs/2607.05927) 指出：多数页级编码把每一页独立嵌，**答不了「需要跨多页聚合信息」的查询**。配套 [CMDR-Embed](https://arxiv.org/abs/2607.05927) 联合编码多页、从共享上下文表示里再抽出页级向量，并用 CMCL（Contextual Multimodal Contrastive Learning）在「上下文建模」与「页级可分」之间折中。读 ViDoRe 时记住：它主要考「找对页」；CMDR 考的是「页间关系」。

---

#### [MVEB](https://arxiv.org/abs/2608.01794)：视觉身份（Visual Identity）专项

- **来源**：[Illuminating Visual Identity in Universal Multimodal Embeddings](https://arxiv.org/abs/2608.01794)（2026.08）
- **动机**：通用 UME 很强于语义匹配，但对 **实例级身份**（同人 / 同物 / Re-ID / AIGC 身份保持）仍弱；许多业务其实同时需要两者
- **做法**：统一 VisID 任务表述 + 大规模 **MVEB** 基准（真实 + 合成）；用 identity-aware sampling 联合优化「通用多模态」与「身份判别」
- **读点**：刷高 MMEB ≠ 会认「是不是同一只包 / 同一个人」；选型时若涉及商品同款、人像检索、版权溯源，应单开 VisID/MVEB 回归

---

#### [MTEB v2](https://huggingface.co/blog/isaacchung/mteb-v2)：从纯文本榜到多模态接口

- 经典 MTEB 是文本嵌入总榜；**v2** 把图文等模态纳入同一评价基础设施  
- 统一检索/重排任务接口，便于横向对比与统计分析  
- 适合问：「我的多模态模型，纯文本能力掉了多少？（对齐税）」  
- **实时榜**：[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

**Multilingual / MMTEB Mean (Task) Top 10（公开对照快照）**

> 来源：[Qwen3-VL-Embedding-8B 模型卡](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) MMTEB 表 + 社区汇总；活榜（尤其 English / Multilingual 分区）分数会继续刷新。此处用 **Mean (Task)**，方便对照「多模态模型的文本对齐税」。

| 排名 | 模型 | Size | Mean (Task) |
|------|------|------|-------------|
| 1 | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 8B | **70.58** |
| 2 | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 4B | 69.45 |
| 3 | [Gemini Embedding](https://ai.google.dev/gemini-api/docs/embeddings) | API | 68.37 |
| 4 | [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | 67.88 |
| 5 | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | 64.33 |
| 6 | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 2B | 63.87 |
| 7 | [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 0.6B | 63.22 |
| 8 | [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) | 7B | 62.51 |
| 9 | Cohere-embed-multilingual-v3.0 | API | 61.12 |
| 10 | [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) | 7B | 60.92 |

> 社区后续还有 [KaLM-Embedding-Gemma3-12B](https://www.codesota.com/benchmarks/mteb) 等更高分提交（约 72+），读活榜时请以当前分区与任务集合为准。多模态旗舰 Qwen3-VL-Embedding-8B（67.88）相对专用文本版 Qwen3-Embedding-8B（70.58）低约 **2.7** 点，就是典型的「对齐税」。

---

#### [Winoground](https://huggingface.co/datasets/facebook/winoground)：组合绑定的「显微镜」

- 典型题：`"phone on a map"` vs `"map on a phone"`  
- **随机基线**：Text/Image **25%**，Group **16.67%**；人类 Group 约 **85.5%**  
- 很强的双塔嵌入也常在 Text **30–40%**、Group 远低于人类；公开报道里 [Apple DFN5B-H](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378) 约 **35.2%** 仍远未「解决」  
- **注意**：生成式 VLM + 特殊推理协议（如 Componential Analysis / SimpleMatch）可以把分数抬得很高，但不等于标准嵌入相似度评测

**Winoground 经典对照（标准配对打分）**

> 以原论文 [Thrush et al., 2022](https://arxiv.org/abs/2204.03162) 为主，按 **Group Score** 排序；并补充后续常被引用的现代双塔结果。这不是持续更新活榜，但足以说明：组合绑定仍远未解决。参考线：Text/Image 随机 **25%**，Group 随机 **16.67%**，人类 Group **85.5%**。

| 排名 | 模型 | Text | Image | Group | 备注 |
|------|------|------|-------|-------|------|
| — | MTurk Human | 89.5 | 88.5 | **85.5** | 人类上界 |
| 1 | VinVL | 37.75 | 17.75 | **14.50** | 原论文较强 ITM 系 |
| 2 | FLAVA ITM | 32.25 | 20.50 | 14.25 | 原论文 |
| 3 | ViLLA large | 37.00 | 13.25 | 11.00 | 原论文 |
| 4 | UNITER large | 38.00 | 14.00 | 10.50 | 原论文 |
| 5 | UNITER base | 32.25 | 13.25 | 10.00 | 原论文 |
| 6 | ViLT | 34.75 | 14.00 | 9.25 | 原论文 |
| 7 | FLAVA Contrastive | 25.25 | 13.50 | 9.00 | 原论文 |
| 8 | CLIP ViT-B/32 | 30.75 | 10.50 | 8.00 | 原论文双塔基线 |
| 9 | ViLLA base | 30.00 | 12.00 | 8.00 | 原论文 |
| 10 | LXMERT / 多数早期双塔 | ~19–24 | ~7 | ~4–5 | Group 常低于随机 |

> 后续数据过滤更好的双塔（如 [Apple DFN5B-H](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378)）在公开综述中常报约 **35.2%** 的组合性相关分数，仍远低于人类；生成式 VLM + 特殊推理协议可显著抬分，但与「纯嵌入相似度」评测不可直接横向对比。

##### Demo

```text
Caption A: "a phone on a map"
Caption B: "a map on a phone"
Image A:   手机放在纸质地图上
Image B:   手机屏幕显示地图 App

正确: (A,A) + (B,B)
打乱: 交叉配对应被判错

测的是: 关系/介词绑定，而不是「图里有没有 phone 和 map 这两个词」。
```

---

### 3.3 评测发现总结（2025–2026 共识）

1. **没有一家包打所有子任务**：[Gemini Embedding 2](https://arxiv.org/abs/2605.27295) 闭源综合强；开源总分常见 [Qwen3-VL-Embedding-8B](https://arxiv.org/abs/2601.04720) 领先，但视频、文档可能另有状元（如 [WAVE-7B](https://arxiv.org/abs/2509.21990)）。  
2. **开源在匹配类任务上已极具竞争力**：小到 2B 级也可能在部分跨模态检索上打过闭源 API。  
3. **组合推理仍难**：Winoground / SugarCrepe 证明「词都对、关系错」依然普遍。  
4. **长上下文退化是系统性的**：MMLongEmbed 画出了随长度下滑的曲线。  
5. **推理是最大短板**：MR² 相对 MMEB 的约 68pp 差距，说明当前嵌入仍偏浅层匹配。

### 3.4 给初学者的「读榜清单」

拿到一篇新论文的表格时，按顺序问：

1. 主榜是 **MMEB 总分** 还是子任务？有没有 OOD？  
2. 有没有报 **ViDoRe / 文档**？（做 RAG 必看）  
3. 有没有碰 **MR² / Winoground**？（宣称「理解强」时必看）  
4. 指标是 R@K 还是 nDCG？**K 和候选库多大**？  
5. 多模态变强后，**纯文本 MTEB** 掉了多少？

---

## 4. 训练策略

> 本章用「技术阅读 blog」的方式，把多模态 Embedding 训练里最常出现的概念拆开讲清楚。如果你刚入门，建议按顺序读：先理解「对比学习在干什么」，再看「四种训练范式差在哪」，最后才是合成数据、硬负样本、MRL、蒸馏这些工程技巧。

> **代码与图片说明**：代码是为解释核心机制而写的 PyTorch 教学实现，省略了生产环境中的分布式通信、混合精度、异常处理和模型专属 processor 适配；论文示意图均保存在仓库 `assets/papers/`，图注链接到原论文或官方仓库。



### 4.0 先建立直觉：Embedding 训练到底在优化什么？

把一张图、一段文字、一页 PDF、一段视频，分别压成一个（或一组）固定长度的向量，就叫 Embedding。训练的目标很朴素：

> **语义相近的东西，向量要靠近；语义无关的东西，向量要远离。**

检索时，只要算查询向量与库中向量的相似度（常用余弦相似度），就能把最相关的结果排到前面。所以训练不是教模型「生成答案」，而是教它**组织空间**：让「对的邻居」聚在一起。

工业界把这件事做成对比学习：每个样本有一个正例（该匹配的），一堆负例（不该匹配的）。损失函数负责惩罚「正例不够近」和「负例不够远」。后面所有花活——换损失、换架构、造合成数据、挖硬负例——都是在更好地完成这件事。

---



### 4.1 核心损失函数：模型怎么学会「靠近 / 远离」



#### InfoNCE（CLIP 族的标准答案）

[CLIP](https://openai.com/research/clip) 把 InfoNCE 带进了图文对齐。用一个 batch 里的例子理解最直观：

假设一个 batch 有 3 对图文：`(图1, 文1)`、`(图2, 文2)`、`(图3, 文3)`。对「图1」来说：

- 正例只有 `文1`
- 负例是同 batch 里的 `文2`、`文3`（以及分布式训练时其他 GPU 上的文本）

模型先算图1与所有文本的相似度，再做 Softmax，目标是：**正例的概率尽量高**。公式写作：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(z_i^I, z_i^T) / \tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z_i^I, z_j^T) / \tau)}$$

其中 $\tau$（温度）控制 Softmax 的「尖锐程度」：$\tau$ 越小，模型越被迫把正负例分得更开。CLIP 还做了对称版本：图找文 + 文找图，两边损失取平均。

##### 代码示例：对称 InfoNCE

下面是最小但完整的 PyTorch 写法。约定 `image_emb` 和 `text_emb` 的形状均为 `[batch_size, hidden_dim]`，且第 `i` 张图与第 `i` 段文本互为正例。

```python
import torch
import torch.nn.functional as F


def symmetric_info_nce(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    # 余弦相似度要求先做 L2 归一化。
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    # [B, D] @ [D, B] -> [B, B]
    # 对角线 logits[i, i] 是正例，其余位置是 in-batch negatives。
    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.T, labels)
    return (image_to_text + text_to_image) / 2


# 模拟 3 对图文。真实训练中这两个张量来自图像塔和文本塔。
image_emb = torch.randn(3, 768, requires_grad=True)
text_emb = torch.randn(3, 768, requires_grad=True)

loss = symmetric_info_nce(image_emb, text_emb)
loss.backward()
```

读代码时抓住三点：

1. **正例由矩阵对角线定义**，所以 DataLoader 必须保证图文顺序严格对齐；
2. **归一化后点积就是余弦相似度**，否则向量模长可能成为模型走捷径的方式；
3. 分布式训练通常会先 `all_gather` 其他 GPU 的向量，再计算更大的相似度矩阵，以获得更多负例。

**为什么大家总强调 Batch Size？**  
InfoNCE 的负例主要来自同 batch。batch 越大，负例越多、越多样，学到的判别边界越稳。CLIP 用到 32,768，就是在用「海量 in-batch negatives」换对齐质量。

> **读论文时常看到的细节**：训练收敛后 $\tau$ 往往掉到 ~0.01，余弦相似度大多挤在 0 附近，即便强匹配也很少超过 0.35–0.40。这不代表模型差，而是对比学习几何里的常态——真正重要的是**相对排序**，不是绝对分数。



#### [SigLIP](https://www.emergentmind.com/topics/sigmoid-loss-for-language-image-pre-training-siglip)：把「多分类」改成「一对一判断」

InfoNCE 有个隐式假设：同 batch 里除了正例以外都是负例。但现实中，batch 里可能碰巧出现另一张也很匹配的图——这叫**假负例（false negative）**。Softmax 会强迫它们互相竞争，训练信号被污染。

SigLIP 换了思路：不再做「在 N 个候选里选 1 个」的 Softmax，而是对每一个（图, 文）配对做独立的 Sigmoid 二分类——「这对是不是匹配？」

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i,j}\log\sigma(z_{ij} \cdot (t \cdot x_i^\top y_j + b))$$

##### 代码示例：SigLIP 的逐对二分类

SigLIP 不再把一整行看成一个多分类问题，而是把相似度矩阵中的每个格子看成独立二分类。对角线标签为 `+1`，非对角线为 `-1`。

```python
import torch
import torch.nn.functional as F


def siglip_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = image_emb @ text_emb.T
    logits = logits * logit_scale.exp() + logit_bias

    batch_size = logits.size(0)
    labels = -torch.ones_like(logits)
    labels.fill_diagonal_(1.0)

    # -log(sigmoid(label * logit))
    pairwise_loss = -F.logsigmoid(labels * logits)
    return pairwise_loss.mean()


B, D = 4, 768
image_emb = torch.randn(B, D, requires_grad=True)
text_emb = torch.randn(B, D, requires_grad=True)
logit_scale = torch.tensor(0.0, requires_grad=True)
logit_bias = torch.tensor(-10.0, requires_grad=True)

loss = siglip_loss(image_emb, text_emb, logit_scale, logit_bias)
loss.backward()
```

工程实现还会做 **loss 分块**：不用一次物化完整的 `[B, B]` 矩阵，而是按文本块或图像块计算，从而降低峰值显存。

直觉对比：


|           | InfoNCE                            | SigLIP                   |
| --------- | ---------------------------------- | ------------------------ |
| 问题形式      | 在 batch 内做多分类                      | 对每个配对做是/否判断              |
| 假负例       | 会互相抢概率                             | 不直接竞争，更宽容                |
| 通信/内存     | 常需 all-gather，偏 $\mathcal{O}(N^2)$ | 更易扩展，接近 $\mathcal{O}(N)$ |
| Batch 敏感度 | 很吃大 batch                          | 从小 batch 到 ~32K 都更稳      |


**2025 年理论补全**（NeurIPS 2025）：[Bangachev 等人](https://arxivlens.com/paperview/details/global-minimizers-of-sigmoid-contrastive-loss-2478-72532172) 用 **(m, b_rel)-Constellations** 刻画 Sigmoid 对比损失的全局最小解，发现嵌入几何会出现相变（ETF 相 / 反足相 / 混合相），并提出相对偏置参数化（LRB-Sig）加速收敛。对初学者，记住一句即可：**SigLIP 不只是工程上更好训，其最优解的几何结构也正在被理论看清。**

**[SigLIP 2](https://arxiv.org/abs/2502.14786)（Google DeepMind, 2025）**：把 sigmoid 损失与 captioning 预训练（LocCa）、自蒸馏、掩码预测、在线数据筛选统一成一条配方；语料基于 [WebLI](https://arxiv.org/abs/2209.06794)（约 100 亿图像 / 109 语）。发布 ViT-B/L/So400m/g 多档，并提供多分辨率与保持原生长宽比的 NaFlex 变体。工程选型上，它仍是**最常用的开源视觉–语言双塔骨干之一**（如 PaliGemma 2）；代码与权重见 [big_vision SigLIP 2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)。

#### 其他常见损失：各管一类任务

真实系统很少只用一种损失。按任务选型更常见：


| 损失函数                                                                   | 它在优化什么                              | 什么时候用        | 代表工作                                                    |
| ---------------------------------------------------------------------- | ----------------------------------- | ------------ | ------------------------------------------------------- |
| **CoSENT**                                                             | 相似度分数的相对排序是否合理                      | STS、分级相关、reranker 监督 | [Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) / [WeMM](https://arxiv.org/abs/2608.24053) |
| **Rank-KL**                                                            | 让嵌入模型复现 reranker 的整表排序分布           | 组合性 / 细粒度检索蒸馏 | [CORE](https://arxiv.org/abs/2609.04083) |
| **KL 散度**                                                              | 让学生分布贴近教师分布                         | 知识蒸馏、软标签对齐   | [UniME](https://dl.acm.org/doi/10.1145/3746027.3754845) / WeMM 跨尺度蒸馏 |
| **Jensen-Shannon**                                                     | 更对称的分布对齐                            | 软匹配分数蒸馏      | [UniME-V2](https://arxiv.org/abs/2510.13515)            |
| **Triplet**                                                            | anchor-positive 近、anchor-negative 远 | 早期排序模型       | 经典度量学习                                                  |
| **Listwise**                                                           | 一整页候选列表的排序质量                        | 训练重排序器       | Reranker 阶段                                             |
| **[GCL](https://neurips.cc/virtual/2025/loc/san-diego/poster/116946)** | 任意模态在 mini-batch 内互相对比              | 不想构造严格三元组时   | NeurIPS 2025                                            |


**一句话记忆**：检索主损失多用 InfoNCE/SigLIP；要「像老师一样打分」用 KL/JS；要「把列表排好」用 listwise。

---



### 4.2 四种训练范式：同一目标，四种「怎么把模态接到一起」

可以把范式理解成四种「工厂流水线」。产品都是 Embedding，但车间布局不同。

#### 范式一：双塔对比预训练（CLIP-style）

```text
  图像 ──► [图像编码器] ──► 向量_I ─┐
                                    ├─► 对比损失（InfoNCE / SigLIP）
  文本 ──► [文本编码器] ──► 向量_T ─┘
```

- **代表**：[CLIP](https://arxiv.org/abs/2103.00020)、[SigLIP](https://arxiv.org/abs/2303.15343) / [SigLIP 2](https://arxiv.org/abs/2502.14786)、[Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2)、[nomic-embed-vision](https://huggingface.co/nomic-ai/nomic-embed-vision-v1)
- **怎么训**：两个编码器互不看对方内部状态，只在最后用相似度对齐。
- **为什么好用**：线上可以**离线算好**所有图像向量入库；查询来了只编码文本，再做向量检索。延迟低、吞吐高。
- **硬伤**：模态之间没有深度交互。遇到「红围巾的狗 vs 坐着的狗」这类组合修改，或「phone on a map」这类语序敏感问题，双塔往往不够细。

> 适合：大规模召回、延迟敏感、图文检索为主的业务。



#### 范式二：MLLM 适配（把会聊天的多模态大模型，改造成会检索的编码器）

2024–2026 年的主旋律。思路不是从零训 Embedding，而是：

> 拿一个已经会看图、会理解指令的 MLLM（如 Qwen-VL、LLaVA），用对比学习把它「拧」成编码器。

![VLM2Vec 使用 VLM 分别编码 Query 与 Target，并以最后一个 token 的隐状态计算对比损失。来源：VLM2Vec 论文 Figure 3。](assets/papers/vlm2vec-framework.png)

*图 4-1：[VLM2Vec](https://arxiv.org/abs/2410.05160) 的训练框架。左、右两侧可以共享同一个 VLM；输入可分别是文本、图像或图文组合，最终都压缩为单向量。*

- **代表**：[VLM2Vec](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html)、[UniME](https://dl.acm.org/doi/10.1145/3746027.3754845)、[MM-Embed](https://openreview.net/forum?id=vG8Mmp4OCP)、[Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720)
- **关键技巧（读论文高频词）**：
  - **指令条件化**：同样一张图，加不同 instruction 会得到不同向量——「用于分类」和「用于检索」不再共用一个死板表示。
  - **LoRA**：只训练低秩适配器（rank 常取 8–16），基座知识尽量保留，省显存也少灾难性遗忘。
  - **GradCache**：对比学习想要大 batch，但 MLLM 显存吃紧。GradCache 把「前向算表征」和「反传算梯度」拆开缓存，用时间换有效 batch size。
  - **多阶段训练**：先大规模对齐 → 再多任务精调 → 再蒸馏/合并，避免一步到位不稳定。
  - **算力数量级**：开源 MLLM→Embedding 微调常见 **8×80GB GPU**；工业预训练/合成与 WAVE/UME-R1 等可达 **几十到上百卡**。详见 [§5.13](#513-训练--推理算力速查公开披露汇总)。

> 适合：要统一分类 / VQA / 检索 / grounding、需要理解复杂指令的通用 Embedding。

##### 代码示例：从 MLLM 的最后一个有效 token 提取向量

以下代码省略了具体模型的图像预处理，只展示论文中最关键的 pooling 逻辑。不能直接使用 `hidden[:, -1]`，因为右侧 padding 会让最后位置落在 PAD token 上。

```python
import torch
import torch.nn.functional as F


def last_token_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # attention_mask: [B, L]，有效 token 为 1。
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_state.size(0),
        device=last_hidden_state.device,
    )
    pooled = last_hidden_state[batch_indices, last_indices]
    return F.normalize(pooled, dim=-1)


def encode_multimodal(model, processor, samples, instruction: str):
    # samples 可以包含文本、图像或二者组合；具体字段随模型而变化。
    prompts = [
        {
            "instruction": instruction,
            "text": sample.get("text"),
            "image": sample.get("image"),
        }
        for sample in samples
    ]
    batch = processor(prompts, padding=True, return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}

    outputs = model(**batch, output_hidden_states=True)
    return last_token_pool(
        outputs.hidden_states[-1],
        batch["attention_mask"],
    )
```

同一输入可以因为 instruction 不同得到不同向量。例如：

```python
retrieval_emb = encode_multimodal(
    model, processor, samples,
    instruction="Represent the input for image-text retrieval.",
)
classification_emb = encode_multimodal(
    model, processor, samples,
    instruction="Represent the image for food classification.",
)
```

##### 代码示例：LoRA 微调与一个对比训练 step

MLLM 通常不用全参数训练，而是在注意力层插入 LoRA。下面把「挂 LoRA → 编码 query/target → 算损失 → 更新」串起来：

```python
from peft import LoraConfig, get_peft_model
import torch


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=2e-5,
)


def train_step(batch):
    model.train()
    optimizer.zero_grad()

    query_emb = encode_multimodal(
        model,
        processor,
        batch["queries"],
        instruction=batch["query_instruction"],
    )
    target_emb = encode_multimodal(
        model,
        processor,
        batch["targets"],
        instruction="Represent the candidate for retrieval.",
    )

    loss = symmetric_info_nce(
        query_emb,
        target_emb,
        temperature=0.02,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.detach()
```

真实多任务训练还会按任务类型切换 instruction，并使用采样器控制每个 batch 的数据来源，避免一个 batch 中混入定义不兼容的正负例。

##### 代码示例：GradCache 的两阶段思路

GradCache 的目标是让对比损失看到大 batch，同时不保留整个大 batch 的计算图。它不是简单的梯度累积：普通梯度累积的每个 micro-batch 仍只看自己的负例，GradCache 则先拿到**全局表示**，再求全局对比损失。

```python
# 教学伪代码：实际项目建议直接使用 GradCache 官方实现。
chunks = split_large_batch(batch, chunk_size=4)

# 阶段 A：无梯度编码所有小块，拼出全局表示。
with torch.no_grad():
    query_repr = torch.cat([encoder(c.query) for c in chunks])
    target_repr = torch.cat([encoder(c.target) for c in chunks])

# 只对表示求全局对比损失，并缓存 dL/dEmbedding。
query_repr.requires_grad_(True)
target_repr.requires_grad_(True)
loss = symmetric_info_nce(query_repr, target_repr)
query_grad, target_grad = torch.autograd.grad(
    loss, [query_repr, target_repr]
)

# 阶段 B：逐块重新前向；用缓存的表示梯度反传到模型参数。
optimizer.zero_grad()
offset = 0
for chunk in chunks:
    q = encoder(chunk.query)
    t = encoder(chunk.target)
    size = q.size(0)

    torch.autograd.backward(
        tensors=[q, t],
        grad_tensors=[
            query_grad[offset:offset + size],
            target_grad[offset:offset + size],
        ],
    )
    offset += size

optimizer.step()
```

代价是每个样本要编码两遍；收益是有效 batch 可以从几十扩大到数百乃至 1024。



#### 范式三：原生多模态大模型（Native Multimodal）

双塔是「先各自编码，再对齐」；MLLM-as-Embedder 是「借聊天模型当编码器」。原生路线更激进：

> **从一开始就设计成深度融合**：文本、图像、视频、音频可以交错输入，在 Transformer 内部早期交互，最后输出统一向量。

- **代表**：[Gemini Embedding 2](https://arxiv.org/abs/2605.27295)
- **直观差异**：你可以丢进去「一段说明文字 + 两张图 + 一段音频」，模型在同一注意力栈里理解整段交错上下文，而不是先分别压成四个向量再拼接。
- **代价**：训练数据、算力、工程复杂度都更高；开源可复现性通常弱于双塔 / MLLM 微调路线。



#### 范式四：多向量检索（Late Interaction / ColBERT 风格）

前面三种大多输出**一个向量**代表整段输入。文档检索有个痛点：一页 PDF 里表格、公式、小字标题挤在一起，压成单向量会丢细节。

多向量方法换思路：

![ColPali 将文档页编码成 patch 向量，将查询编码成 token 向量，再通过 MaxSim 做 late interaction。来源：ColPali 官方仓库与论文 Figure 1。](assets/papers/colpali-architecture.webp)

*图 4-2：[ColPali](https://arxiv.org/abs/2407.01449) 架构。文档向量可离线计算；在线阶段只编码查询，并让每个查询 token 寻找最相关的页面 patch。*

- **代表**：[ColPali](https://huggingface.co/vidore/colpali)、[ColQwen](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3)、[Argus-Retriever](https://arxiv.org/abs/2606.04300)
- **为什么强**：表格单元格、图注、栏位布局都能被某个 patch「盯住」。
- **代价**：存储从「每页 1 个向量」变成「每页几十到几百个向量」。于是才有 Light-ColPali 的 token 合并、Argus 的低维 MoE 等压缩工作。

> 适合：PDF / 扫描件 / 幻灯片 / 信息图等视觉文档检索（VDR）。

##### 代码示例：MaxSim late interaction

```python
import torch
import torch.nn.functional as F


def maxsim_score(
    query_tokens: torch.Tensor,   # [Bq, Lq, D]
    document_patches: torch.Tensor,  # [Bd, Ld, D]
) -> torch.Tensor:
    query_tokens = F.normalize(query_tokens, dim=-1)
    document_patches = F.normalize(document_patches, dim=-1)

    # [Bq, Bd, Lq, Ld]：每个查询 token 对每个文档 patch 的相似度。
    similarity = torch.einsum(
        "qld,pkd->qplk",
        query_tokens,
        document_patches,
    )

    # 每个查询 token 只保留最匹配的文档 patch，再对查询 token 求和。
    return similarity.max(dim=-1).values.sum(dim=-1)  # [Bq, Bd]


scores = maxsim_score(query_embeddings, page_embeddings)
top_pages = scores.topk(k=10, dim=-1).indices
```

直接物化四维张量在大 batch / 长页面下非常耗显存。生产实现通常用分块或 fused Triton kernel，在不创建完整 `[Bq, Bd, Lq, Ld]` 张量的情况下完成 `max + sum`。

**四种范式怎么选（给初学者的速查）**


| 你的场景              | 更优先考虑                |
| ----------------- | -------------------- |
| 海量图文召回、要极致速度      | 双塔                   |
| 统一多任务、指令复杂        | MLLM-as-Embedder     |
| 任意交错模态、闭源 API 可接受 | 原生多模态                |
| PDF/文档页级精细检索      | 多向量 Late Interaction |


---



### 4.3 数据策略：没有好数据，损失设计再漂亮也白搭



#### 合成数据：用生成模型「造对齐语料」

真实图文对贵、脏、且长尾不足。2025 年后，合成数据从「凑数」变成「可打榜」：


| 方法                                                   | 它怎么造数据                              | 你该记住的一点                                       |
| ---------------------------------------------------- | ----------------------------------- | --------------------------------------------- |
| **[mmE5](https://arxiv.org/abs/2502.08468)** ([代码](https://github.com/haon-chen/mmE5)) | MLLM 单轮合成 → 自评估 → 精炼 | 强调广度、跨模态对齐、高保真三条标准 |
| **[PolyGen](https://arxiv.org/abs/2602.01370)**      | 多生成器（SD1.5/SD2/SDXL-Turbo/SANA）同题多图 | 多正例 + 程序化硬负例 + 课程调度；相对单源 SynthCLIP **+19.0%** |
| **[Qwen3-VL](https://arxiv.org/abs/2601.04720) 流水线** | 用 Qwen3-VL-32B 自动标/写描述              | 规模到 **3 亿+**，支撑工业级预训练                         |


**初学者易错点**：合成数据不是越多越好。关键是「多样性 + 难例质量」。同一 caption 配多个风格图像（PolyGen）往往比单生成器刷量更有效。

##### 代码示例：合成数据的「生成—自评估—精炼」管道

下面的示例展示 [mmE5](https://arxiv.org/abs/2502.08468) 风格的控制流程。`mllm.generate` / `mllm.score` 是接口占位符，实际可接本地 VLM 或 API。

```python
def build_synthetic_pair(image, mllm, min_score=0.8):
    draft = mllm.generate(
        image=image,
        prompt=(
            "为该图像生成一个检索查询和一个目标描述；"
            "两者语义一致，但措辞不要重复，并覆盖细粒度属性。"
        ),
    )

    scores = mllm.score(
        image=image,
        query=draft["query"],
        target=draft["target"],
        criteria=["cross_modal_alignment", "fidelity", "difficulty"],
    )

    if min(scores.values()) < min_score:
        draft = mllm.generate(
            image=image,
            prompt=(
                "根据以下评分修正样本，删除幻觉，补足视觉事实，"
                f"并提高检索难度：{scores}\n原样本：{draft}"
            ),
        )

    return {
        "instruction": "Retrieve the target that matches the query.",
        "query": {"image": image, "text": draft["query"]},
        "positive": {"text": draft["target"]},
        "quality_scores": scores,
    }
```

数据落盘时建议保留生成模型版本、prompt、质量分和来源图像 ID，后续才能追踪坏样本来自哪一步。

#### 硬负样本：训练中最「值钱」的反例

对比学习里，太简单的负例（蓝天 vs 狗）很快就分对了，梯度几乎为 0。真正拉高上限的是**看起来很像、其实不对**的样本。

```text
Query: "a brown dog running on grass"
Positive: 棕色犬在草坪奔跑

Easy Neg:  蓝天白云           ← 太简单，学不到东西
Hard Neg:  黑色犬在草坪奔跑   ← 属性相近，值得学
Hard Neg:  棕色犬坐在草地上   ← 动作不同，值得学
False Neg: 同一只狗另一角度   ← 其实也该算正例，必须过滤！
```

2025–2026 的三类进阶做法：

1. **[EGA](https://arxiv.org/abs/2506.02020)（腾讯 QQMM）**
  不只看「负例有多像 query」，而是看**相对难度**：负例相对正例有多「抢戏」。越难分的，梯度放大越多。
2. **[UniME-V2](https://arxiv.org/abs/2510.13515) 的 MLLM-as-a-Judge**
  让 MLLM 用 Yes/No 概率当「软相关分」：既能踢掉假负例，又能当蒸馏软标签。比死阈值更懂语义。
3. **多轮检索管道**
  先粗召回一大池 → 再精打分过滤 → 再挑最难的进 batch。工业界很常见。

![UniME 第二阶段先过滤假负例，再从剩余候选中选择硬负例。来源：UniME 论文 Figure 3。](assets/papers/unime-stage2-hard-negatives.png)

*图 4-3：[UniME](https://arxiv.org/abs/2504.17432) 的硬负样本增强训练。斜线区域是假负例：它们虽然不是数据集标注的正例，却在语义上也正确，不能强行推远。*

##### 代码示例：挖硬负例并过滤假负例

```python
import torch
import torch.nn.functional as F


@torch.no_grad()
def mine_hard_negatives(
    query_emb: torch.Tensor,       # [B, D]
    candidate_emb: torch.Tensor,   # [N, D]
    positive_ids: torch.Tensor,    # [B]
    judge_scores: torch.Tensor,    # [B, N]，MLLM Judge 的相关性分数
    top_k: int = 8,
    false_negative_threshold: float = 0.8,
) -> torch.Tensor:
    query_emb = F.normalize(query_emb, dim=-1)
    candidate_emb = F.normalize(candidate_emb, dim=-1)
    similarity = query_emb @ candidate_emb.T

    # 已知正例不能被选为负例。
    row = torch.arange(query_emb.size(0), device=query_emb.device)
    similarity[row, positive_ids] = -torch.inf

    # Judge 认为高度相关的候选可能是假负例，也排除。
    similarity = similarity.masked_fill(
        judge_scores >= false_negative_threshold,
        -torch.inf,
    )

    return similarity.topk(k=top_k, dim=-1).indices
```

没有 MLLM Judge 时，可以先用「正例相似度差值 + 人工规则」过滤，但要定期抽样检查：**最相似的负例通常也是最容易标错的负例**。



#### 课程学习：别一上来就喂最难题

- **渐进注入难负例**：早期先学粗对齐，再逐步加难，避免训练早期崩掉。
- **多正例目标**：同一语义多个正样本一起拉近（合成多图、多标注）。
- **两阶段过滤**：余弦召回 → 相关性过滤（只留「和正例差不多像、但又不是正例」的难负例）。

---



### 4.4 Matryoshka 表示学习（MRL）：一个模型，多种向量长度

线上常有矛盾：维度越高通常越准，但存储/带宽越贵。MRL 的想法像套娃：

> 训练时让向量的**前缀子向量**也各自有用。推理时若只要 512 维，直接截断前 512 维，不必重新训一个小模型。

例如同时支持 128 / 256 / 512 / 1024 / 2048。实践里，[Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) 从 2048 降到 1024，MRR 往往只掉 1–2%，存储和计算却近乎减半。


| 方法                                                         | 会议             | 它在解决什么                       |
| ---------------------------------------------------------- | -------------- | ---------------------------- |
| **[SMEC](https://aclanthology.org/2025.emnlp-main.1332/)** | EMNLP 2025     | 序列化 MRL + 自适应选维 + 跨 batch 记忆 |
| **[CSR](https://icml.cc/virtual/2025/oral/47166)**         | ICML 2025 Oral | 用稀疏编码替代传统 MRL，少重训            |
| **[MIPIC](https://github.com/tmp0810/MIPIC)**              | ACL 2026       | 子矩阵对齐 + 管道 InfoNCE，强化各前缀质量   |
| **Qwen3-VL MRL+QAT**                                       | 2026           | 截断维度的同时做量化感知训练               |
| **MURE（RMRL）** | 2026 | 分辨率级 MRL，服务视觉文档（公开细节较少） |


**你可以这样向同事解释 MRL**：不是训练 5 个模型，而是训练 1 个「从头到尾信息密度递减有序」的向量。

##### 代码示例：同时监督多个前缀维度

```python
import torch
import torch.nn.functional as F


def matryoshka_info_nce(
    query_emb: torch.Tensor,
    target_emb: torch.Tensor,
    dimensions=(128, 256, 512, 1024),
    temperature=0.02,
) -> torch.Tensor:
    losses = []

    for dim in dimensions:
        # 截取前缀后必须重新归一化。
        q = F.normalize(query_emb[:, :dim], dim=-1)
        t = F.normalize(target_emb[:, :dim], dim=-1)
        logits = q @ t.T / temperature
        labels = torch.arange(q.size(0), device=q.device)
        losses.append(
            0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )
        )

    return torch.stack(losses).mean()
```

如果所有维度等权，小维度有时会拖累全维性能。实践中常给大维度更高权重，或按训练阶段逐步加入更短前缀。

---



### 4.5 知识蒸馏：让小/快的模型，学会大/慢模型的「品味」

蒸馏的核心不是抄参数，而是**抄行为**：

![UniME 第一阶段用冻结的教师嵌入模型生成离线表示，通过 batch 内相似度分布把判别能力蒸馏给学生 LLM。来源：UniME 论文 Figure 2。](assets/papers/unime-stage1-distillation.png)

*图 4-4：[UniME](https://arxiv.org/abs/2504.17432) 的文本判别知识蒸馏。教师向量可提前离线计算；学生只更新 LoRA，视觉编码器在这一阶段保持冻结。*

- **[UniME](https://dl.acm.org/doi/10.1145/3746027.3754845) 阶段一**：教师是强文本嵌入模型 NV-Embed-v2。学生不必逐维模仿教师向量，而是让「batch 内相似度分布」用 KL 对齐——学会「谁和谁该像」。
- **[Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) 阶段三**：先把 Reranker（会细看 query-doc）的判断力，蒸馏回双塔/单向量 Embedding，让召回模型也变「眼尖」。
- **[WeMM-Embedding](https://arxiv.org/abs/2608.24053) Stage-2**：9B 教师把 batch 内双向相似度分布蒸馏给 2B/4B；9B 自身则靠多配方变体 **model merging**。
- **[CORE](https://arxiv.org/abs/2609.04083)**：同一骨干既当 reranker 又当 embedder，用 **Rank-KL** 把「属性绑对了没有」蒸馏进单向量空间。
- **[xVLM2Vec](https://arxiv.org/abs/2503.09313)**：自蒸馏把单语能力迁到多语言，少依赖昂贵多语标注。

**给初学者的比喻**：教师像资深面试官，不要求学员背面试官的笔记原文，而要求学员对同一批候选人的「相对排序」与面试官一致。

##### 代码示例：蒸馏 batch 内相似度分布

逐维 MSE 要求教师和学生维度一致，也过度约束坐标系。下面改为对齐 `[B, B]` 的关系矩阵：只要求两者对「哪些样本更相似」看法一致。

```python
import torch
import torch.nn.functional as F


def relational_distillation_loss(
    student_query: torch.Tensor,
    student_target: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_target: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    student_query = F.normalize(student_query, dim=-1)
    student_target = F.normalize(student_target, dim=-1)
    teacher_query = F.normalize(teacher_query.detach(), dim=-1)
    teacher_target = F.normalize(teacher_target.detach(), dim=-1)

    # [B, B]：第 i 个 query 对 batch 内所有 target 的相对偏好。
    student_logits = student_query @ student_target.T / temperature
    teacher_logits = teacher_query @ teacher_target.T / temperature

    teacher_prob = F.softmax(teacher_logits, dim=-1)
    student_log_prob = F.log_softmax(student_logits, dim=-1)

    # batchmean 使损失量级不随 batch size 线性增长。
    return F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    ) * (temperature ** 2)


with torch.no_grad():
    teacher_query = teacher_model(query_text)
    teacher_target = teacher_model(target_text)

student_query = student_model(query_text)
student_target = student_model(target_text)
loss = relational_distillation_loss(
    student_query,
    student_target,
    teacher_query,
    teacher_target,
)
loss.backward()
```

教师表示适合离线缓存；如果教师也在训练循环中实时前向，显存和吞吐成本会明显增加。

---



## 5. 代表模型与最新进展（按任务）

> 本章按**任务**组织，不再逐个精读模型。每类任务给出主要评测、代表模型（一行一个：做法要点与结果）和截至 **2026-10 初** 的最新进展。同一个模型可能在多类任务里都强（例如 WeMM 既在通用总榜第一梯队，也是视频与文档的强对照），这里放在它最有代表性的任务下，其他地方只交叉引用。数字以论文自报或公开榜快照为准，活榜会继续刷新。

### 5.0 任务地图

| 任务 | 主要评测 | 当前领先（截至 2026-10 初） | 主流做法 | 小节 |
|---|---|---|---|---|
| 通用多模态嵌入 | MMEB-V2 / V3、M-BEIR、UMRB | MMEB-V2：Ovis-Embedding-VL-9B 自报 81.13（尚未开源）；已开源最高 WeMM-9B 80.6；活榜 DME-Large 80.24 | MLLM 末 token + 多阶段对比 + reranker 蒸馏 | 5.1 |
| 图文检索与大规模索引 | COCO、Flickr30K、MTEB 图文 | 双塔 SigLIP 2、Jina CLIP v2；闭源上限 Gemini Embedding 2 | 双塔 + MRL，或 MLLM 嵌入精召回 | 5.2 |
| 组合检索与交错查询 | CIRR、FashionIQ、CIRCO、MERIT、PinPoint | ReCALL（CIRR R@1 55.52）；免训练 G-MIXER（CIRCO mAP@50 32.39） | MLLM 单向量 + 纠错数据；免训练改写 + MLLM 重排 | 5.3 |
| 组合性与细粒度绑定 | Winoground、SugarCrepe++、COLA、NegBench | CORE-EMBED-8B / CORE-RERANKER-8B | reranker → 嵌入的 listwise 蒸馏 | 5.4 |
| 视觉文档检索 | ViDoRe V1–V3、JinaVDR、CMDR-Bench | ViDoRe V3：EVIE-8B 66.24（第三方榜快照） | 页当图像 + 多向量 MaxSim + 索引压缩 | 5.5 |
| 视频检索 | MMEB-V2 视频、MSR-VTT、CiA、RTime、长视频基准 | MMEB-V2 视频子项：octen-vl-embedding-large 75.95；细粒度时序与否定：TARA | MLLM 视频嵌入、文本难负例、多层多槽 | 5.6 |
| 音频与全模态 | MMEB-V3、MAEB、Clotho、SoundDescs | Ovis-Embedding-Omni-3B（MMEB-V3 58.46） | 直接以 Omni 骨干做对比 | 5.7 |
| 推理密集检索 | MR²-Bench、MRMR | 方法分化：自适应 CoT、潜在推理、成对判别、专家扩容 | 先想再嵌；训练期想、推理期不说 | 5.8 |
| 实例级与个性化检索 | ILIAS、MVEB（视觉身份）、PerVL-DeepFashion2、ConCon-Chi | 只认实例：FoundYou；带场景的个性化：POLAR | 身份感知采样；逐概念 LoRA；冻结 SAM 2 + 检索头 | 5.9 |
| 效率与部署 | 存储、延迟、索引兼容 | UEmbed、PUMA、NeoMME、Eddy-VL、ColNanoVDR | 稀疏化、token 压缩、蒸馏、维度弹性 | 5.10 |
| 垂直领域 | 站内集与领域基准 | 推荐 CAMIE、驾驶 TraVEL、电商 HMGCLIP、地理 Geo-Embed | 通用嵌入 + 领域监督 | 5.11 |

---

### 5.1 通用多模态嵌入（MMEB 总分）

**任务**：一个模型、一套指令，覆盖分类、VQA、检索、定位；图像、视频、视觉文档进同一向量空间。MMEB-V2 是主榜（78 个数据集），MMEB-V3 再加入音频与 Agent（见 §3.2）。

**代表模型**

| 模型 | 时间 / 出处 | 做法要点 | 结果（MMEB-V2 Overall，除注明外） |
|---|---|---|---|
| [VLM2Vec](https://proceedings.iclr.cc/paper_files/paper/2025/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html) / [VLM2Vec-V2](https://huggingface.co/papers/2507.04590) | ICLR 2025 / 2025.07 | 所有任务写成「指令 + 输入 → 向量」；InfoNCE + GradCache + LoRA；同时给出 MMEB | 立下任务格式；去掉指令性能约掉 30% |
| [E5-V](https://arxiv.org/abs/2407.12580) | 2024.07 | 只用约 27 万对 NLI 文本做对比微调；提示词「用一个词概括」让图和文按语义表示 | 训练成本约降 95%；开了「文本判别力迁移」一路 |
| [MM-Embed](https://openreview.net/forum?id=vG8Mmp4OCP) | ICLR 2025 | 模态感知硬负例，纠正模态偏置；召回后用 MLLM 提示式重排 | M-BEIR R@5 52.7；纯文本 MTEB 60.3，反超文本前身 |
| [GME](https://arxiv.org/abs/2412.16855) | CVPR 2025 | Qwen2-VL bi-encoder；约 800 万样本，其中融合模态约 200 万（110 万合成） | UMRB 当时 SOTA；MMEB-V2 7B 约 57 |
| [UniME](https://dl.acm.org/doi/10.1145/3746027.3754845) / [UniME-V2](https://arxiv.org/abs/2510.13515) | ACM MM 2025 / 2025.10 | 先蒸馏文本教师（NV-Embed-v2）的相似度分布，再挖难负例；V2 让 MLLM 用 Yes/No 概率当软标签裁判 | 336² 分辨率下组合检索仍强 |
| [MuCo](https://arxiv.org/abs/2602.06393) | 2026 | 把对比学习组织成多轮对话：有效 batch 约 ×7，额外算力 ≤3% | MuCo-7B 零样本 61.6（mmE5-11B 为 58.6） |
| [Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) | 2026.01，阿里 | 三阶段：大规模对比预训练 → 多任务精调（InfoNCE / CoSENT / 二分类）→ reranker 蒸馏 + 权重合并；MRL + QAT；3 亿+ 合成数据；训练语料未公开 | 8B 77.8，2B 73.3；仍是后训练默认基座 |
| [Magic-MM-Embedding](https://arxiv.org/abs/2602.05275) | 2026.02 | 视觉 token 双线性下采样到约 1/4，再用三阶段粗到细训练补回 | 压缩后分数不降反升 |
| [TSEmbed](https://arxiv.org/abs/2603.04772) | 2026.03 | MoE-LoRA 让冲突任务走不同专家；以路由分布相似度挖难负（EANS） | 处理多任务梯度冲突 |
| [DME](https://arxiv.org/abs/2608.02148) | 2026.08，抖音 | 对比预训练后，只在训练期加入证据型潜在推理与交叉条件重建 | 2B 74.8 / 9B 78.4；活榜 DME-Large 80.24；线上近普通对比编码器 |
| [WeMM-Embedding](https://arxiv.org/abs/2608.24053) | 2026.08，微信 | 数亿对齐 + 约 1/10 规模精炼集；score-gap 加权 CoSENT、Semantic-ID 重采样、9B→2B/4B 蒸馏、9B 多配方 merging | 2B 77.9 / 4B 79.2 / 9B 80.6；[Apache 2.0 开源](https://github.com/Tencent/WeMM-Embedding)；不支持音频 |
| [MoEMB](https://arxiv.org/abs/2609.08663) | 2026.09 | 沿专家维度扩容，保持单向量、非自回归编码；系统研究 MoE 嵌入的自适应计算 | 3B 激活参数；在只用公开 MMEB 系数据的模型里，MMEB-V2 与 MRMR 均为最好，超过激活参数大 4 倍以上的 TTE 类方法 |
| [Ovis-Embedding-VL](https://arxiv.org/abs/2609.25165) | 2026.09 | 以 Qwen3.5 2B / 9B 原生多模态骨干初始化，末 token、无投影头；同源采样组 batch、focal loss、相似度蒸馏；推理时低秩分解压维 | 9B 81.13（图像 83.96 / 视频 72.90 / 文档 83.06），2B 77.46；论文称将开源；对照表未含 WeMM-9B |
| octen-vl-embedding-large、QQMM-embed-v4 | [活榜](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)提交 | 无技术报告 | 80.09（视频 75.95）/ 78.31 |

其他常被引用的方法基线：[mmE5](https://arxiv.org/abs/2502.08468)（合成数据质量）、[QQMM-embed-v2](https://arxiv.org/abs/2506.02020)（按相对难度放大难负梯度）、[GCL](https://neurips.cc/virtual/2025/loc/san-diego/poster/116946)（不强制三元组的通用对比）、[U-MARVEL](https://arxiv.org/abs/2507.14902)（拆解 MLLM 检索吃哪些因素）、[RzenEmbed-v2](https://arxiv.org/abs/2510.27350)（文档子项很深）、[seed-1.6-embedding](https://seed.bytedance.com/en/blog/built-on-seed1-6-flash-seed-1-6-embedding-launched)（闭源，活榜 76.97）。

**最新进展（2026.07–10）**

1. **80 分线已过，拉开差距的是配方，不是骨干**。WeMM、DME、Ovis 都建在 Qwen3.5 或同档骨干上，差别在于两阶段数据（大规模对齐 + 精炼集）、分级相关监督、reranker 与跨尺度蒸馏，以及 model merging。
2. **2B 追上 8B**。WeMM-2B 77.9、Ovis-VL-2B 77.46，都已超过年初的 8B 标杆 Qwen3-VL-Embedding-8B（77.8）。
3. **扩容换了一个轴**。MoEMB 只加专家、不加稠密参数，也不加推理 token，就在公开数据上拿到最好成绩；TSEmbed 用 MoE 处理任务冲突。
4. **训练期推理、推理期不说**。DME 只在训练中加潜在推理，线上仍是普通对比编码器（推理增强一路见 §5.8）。
5. **读榜要分数据口径**。DME、WeMM 用了站内私有数据；UEmbed、MoEMB 只用公开数据；Ovis 尚未开源，octen 没有技术报告。纯文本能力仍有对齐税（见 §3.2 的 MTEB 部分）。

**读完带走**：入门先复现 VLM2Vec 的数据格式与 GradCache；做通用嵌入抄 Qwen3-VL-Embedding / WeMM 的「多阶段 + 指令 + MRL + 蒸馏」配方。开放权重默认选 WeMM-9B 或 2B；继续后训练时，Qwen3-VL-Embedding 的 ms-swift 生态更熟。

---

### 5.2 图文检索与大规模索引

**任务**：文找图、图找文，库在百万到十亿级，要求编码快、向量小、能接近似最近邻索引。

| 模型 | 做法要点 | 结果 / 定位 |
|---|---|---|
| [SigLIP 2](https://arxiv.org/abs/2502.14786) | 双塔，sigmoid 成对损失，多语言、多分辨率 | 现代双塔默认骨干；十亿级离线索引首选 |
| [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2) | 约 0.9B 双塔，89 种语言，文本最长 8192 token；三阶段分辨率课程；MRL 64–1024 维 | 截到 256 维损失常 <1%；MMEB-V2 约 55 分档，比顶级 MLLM 嵌入低约 20 分 |
| [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) | 与 nomic 文本模型共享空间 | 轻量部署 |
| [Gemini Embedding 2](https://arxiv.org/abs/2605.27295)（闭源） | 单一原生多模态骨干 + 双向注意力；3072 维，MRL 可截到 768 / 1536；图、文、视频、音频可交错输入 | COCO T→I R@1 62.9，Flickr30K 89.1；音频嵌入比「先 ASR 再嵌」约高 3.6 MRR@10 |
| MLLM 嵌入（WeMM、Qwen3-VL-Embedding 等） | 见 §5.1 | 精度上限高，编码成本明显高于双塔 |

**最新进展**：分工已经稳定。双塔负责十亿级召回，MLLM 嵌入负责中小库或第二段精召回；MRL 与量化是两路共同的标配。Gemini Embedding 2 的早期融合仍是闭源上限，开源侧还没有同等通用的原生双向交错骨干（NeoMME 只覆盖文档，见 §5.5）。

**读完带走**：不是所有业务都需要 7B 嵌入；多语言双塔 + MRL 仍是性价比默认项。

---

### 5.3 组合检索与交错多条件查询

**任务**：查询是「参考图（或视频）+ 修改文本」，或多张图与文本交错，目标是同时满足这些条件的图或视频。

| 模型 / 基准 | 出处 | 做法要点 | 结果 |
|---|---|---|---|
| [GME](https://arxiv.org/abs/2412.16855) | CVPR 2025 | 大规模合成融合模态训练对 | 融合模态检索的通用基线 |
| [InstructCIR](https://arxiv.org/abs/2412.05756) | arXiv 2024.12 | 指令微调的 MLLM 直接产出组合表示；两阶段（对比 + 三元组） | CIRR R@1 35.08，CIRCO mAP@5 22.32（零样本） |
| [ReCALL](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_ReCALL_Recalibrating_Capability_Degradation_for_MLLM-based_Composed_Image_Retrieval_CVPR_2026_paper.html) | CVPR 2026 | 指出把生成式 MLLM 压成单向量检索器会丢掉细粒度推理；先找检索器的盲点，再让基座 MLLM 生成纠错三元组并用 VQA 一致性过滤，最后分组对比续训 | CIRR R@1 55.52；FashionIQ SOTA |
| [G-MIXER](https://openaccess.thecvf.com/content/CVPR2026/papers/Lim_G-MIXER_Geodesic_Mixup-based_Implicit_Semantic_Expansion_and_Explicit_Semantic_Re-ranking_CVPR_2026_paper.pdf) | CVPR 2026 | 免训练：在参考图与文本之间做测地线 mixup 扩候选，再用 MLLM 抽出的显式语义重排 | CIRCO mAP@50 32.39（ViT-L/14） |
| [MERIT / CORAL](https://proceedings.neurips.cc/paper_files/paper/2025/file/6c7154e394e24c69409256ccf8bf0804-Paper-Conference.pdf) | NeurIPS 2025 | 多语言交错多条件商品检索基准；对比 + 嵌入重建微调 | 现有 MLLM 检索器把多图拼成一张时 R@1 约为交错输入的 16 倍；训练后交错输入提升 14.3% |
| [wikiHow-TIIR / MME](https://aclanthology.org/2025.acl-long.214.pdf) | ACL 2025 | 文图交错的查询与文档；按粒度压缩视觉 token | 保留交错结构是关键 |
| [PinPoint](https://openaccess.thecvf.com/content/CVPR2026/papers/Mahadev_PinPoint_Evaluation_of_Composed_Image_Retrieval_with_Explicit_Negatives_Multi-Image_CVPR_2026_paper.pdf) | CVPR 2026 | 7,635 条查询，显式难负例、6 种改写、13.4% 为多图查询 | 多图查询比单图低 40%–70%；免训练 MLLM 重排对各类方法都有效 |

组合视频检索（参考视频 + 修改文本）见 §5.6。

**最新进展**

1. **嵌入化造成的能力退化被点名**。ReCALL 的对策是让基座 MLLM 自己出纠错数据，而不是继续堆通用三元组。
2. **免训练路线仍有竞争力**。先改写或扩候选、再让 MLLM 重排（G-MIXER、PinPoint 的重排基线）。
3. **多图查询是共同短板**。MERIT 和 PinPoint 都显示交错多图明显退化，原因是现有训练集几乎都是单图。

**读完带走**：组合检索里，「单向量召回 + MLLM 重排」比纯单向量稳；自己训练时优先补多图和纠错类数据。

---

### 5.4 组合性与细粒度绑定

**任务**：分清「红杯子在蓝盘子上」和「蓝杯子在红盘子上」，以及否定。[Winoground](https://huggingface.co/datasets/facebook/winoground) 上仍只有 30%–40%。

| 工作 | 做法 | 结果 |
|---|---|---|
| [UniME-V2](https://arxiv.org/abs/2510.13515) | MLLM 软标签过滤假负例、做分布对齐 | SugarCrepe 上分辨率不高也很强 |
| [CORE](https://arxiv.org/abs/2609.04083) | 合成五级组合匹配候选（全对 → 部分在场 → 属性错 / 物体错 → 完全不匹配）；比较对比学习、成对 CoSENT 与 listwise Rank-KL；把 reranker 的排序蒸进嵌入 | CORE-RERANKER-8B 在 COLA / SugarCrepe++ / NegBench 总均 82.7%（比 Jina-Reranker 高 10.7）；CORE-EMBED-8B 组合性在所评嵌入中最好，迁到 COCO / Flickr30K 不掉点 |
| [TARA](https://arxiv.org/abs/2512.13511) | 文本难负例覆盖否定与时序相反的动作 | NegBench 等 SOTA，见 §5.6 |

**最新进展**：结论已经趋同。同一个 MLLM 当 reranker 能分清属性绑定，变成嵌入就糊；把 reranker 的排序分布蒸进嵌入，比单纯多加负例更对症。用图像编辑造视觉难负例这一路（CORE、[RePair](https://arxiv.org/abs/2608.29604)、COMO、TripletCLIP）已经很密集。

---

### 5.5 视觉文档检索

**任务**：在大量 PDF 或扫描页里找到能回答问题的那一页。OCR 后再做文本嵌入会丢版式和表格结构，主流改为把页当图像。

| 模型 | 出处 | 做法要点 | 结果 |
|---|---|---|---|
| [ColPali](https://arxiv.org/abs/2407.01449) / [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0) | 2024 / 2025 | 页当图像送进 VLM，保留 patch 级向量；查询每个 token 找最匹配的 patch（MaxSim）再聚合 | ViDoRe V1：84.8 / 89.3 |
| [Light-ColPali](https://aclanthology.org/2025.findings-acl.1003/) | ACL Findings 2025 | token 合并 | 约 11.8% 内存保留 98.2% 性能 |
| [Nemotron ColEmbed V2](https://arxiv.org/abs/2602.03992) | 2026.02，NVIDIA | 双向注意力、正例感知难负、聚类采样、模型汤；3B / 4B / 8B 开放权重 | ViDoRe V3 63.42（2026-04 榜首） |
| [Argus-9B](https://arxiv.org/abs/2606.04300) | 2026.06 | 查询条件化、区域感知的 MoE 晚交互 | V1+V2 86.0 nDCG@5 |
| [CMDR-Embed](https://arxiv.org/abs/2607.05927) | 2026.07 | 多页联合编码，再从共享上下文中抽出页级向量 | 处理需要跨页聚合的查询 |
| [NeoMME](https://arxiv.org/abs/2609.01657) | 2026.09 | 从零训练 260M / 800M 双向单塔，掩码离散扩散预训练；稠密头 + 多向量头 | ViDoRe V3：260M 0.523 / 800M 0.556；多向量索引可压约 255× |
| [EVIE-8B](https://huggingface.co/tencent/EVIE-8B) / [EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B) | 2026.09，腾讯 | 晚交互多向量；8B 作教师，4.5B 用关系蒸馏学单投影 Prefix-MRL（64–2048 维）；免训练层次聚类把每页约 750 个向量压到 32 或 64 个 | ViDoRe V3 第三方榜 66.24 / 65.70（2026-10-08 快照）；百万页索引约 3.81 GiB；权重与训练管线已开源，正式论文未发布 |
| [ColNanoVDR](https://arxiv.org/abs/2609.34899) | 2026.09 | 只用教师的查询向量做蒸馏、不编码任何文档页；熵正则最优传输对齐学生与教师的查询 token | 149M 纯文本学生保留教师约 95% nDCG@5，查询编码最快约 26× |

**最新进展**

1. **榜首仍是多向量晚交互，竞争点从打分移到存储**。EVIE 的层次聚类、NeoMME 的 255× 压缩、Light-ColPali 的 token 合并都在压索引。ViDoRe V3 前列还有 webAI-ColVec1.1-8b（64.95）和 VultronRetrieverPrime-Qwen3.5-8B（64.26）等无论文提交。
2. **查询侧也在瘦身**。ColNanoVDR 把数十亿参数的查询编码器蒸成 1.5 亿参数的纯文本模型，直接查教师原有索引。
3. **从单页走向多页和管线**。CMDR 处理跨页聚合；ViDoRe V3 新开了[检索管线榜](https://huggingface.co/blog/antoineedy/vidore-v3-pipeline-framework-and-leaderboard)，可以比较含重排或多步 Agent 的完整管线。
4. **小单塔可行**。NeoMME 表明文档检索不必用 2B+ 的生成式骨干。

**读完带走**：做文档 RAG，先评估页级多向量，并从第一天规划索引存储；查询延迟敏感时看查询侧蒸馏。

---

### 5.6 视频检索

**任务**：文找视频、视频找文本、片段定位，以及「参考视频 + 修改文本」的组合视频检索。

| 模型 / 基准 | 出处 | 做法要点 | 结果 |
|---|---|---|---|
| 通用嵌入的视频子项 | 见 §5.1 | — | MMEB-V2 视频：octen-large 75.95、DME-Large 74.36、WeMM-9B 74.3、Ovis-VL-9B 72.90、Qwen3-VL-Embedding-8B 67.15 |
| [WAVE-7B](https://arxiv.org/abs/2509.21990) | 2025.09 | 文、音频、视频统一的 7B Omni 骨干 | 视频榜长期参考 |
| [TARA](https://arxiv.org/abs/2512.13511) | ECCV 2026 | Tarsier2-7B 改成嵌入模型；只用精心采样难负例的文本三元组做对比微调，覆盖时序相反的动作、否定和组合视频检索 | CiA、RTime、NegBench、CoVR 均为 SOTA；8 张 A6000 约 1 小时；文本训练缩小了视频–文本模态差 |
| [MARS](https://arxiv.org/abs/2609.02565) | EMNLP 2026 | 不再只用末层单 token：组合多层隐状态成多个表示槽，加难负感知的槽特化 | 四个文本–视频检索基准 SOTA（直接检索与重排两种设置） |
| [SCALAR](https://arxiv.org/abs/2609.15335) | 2026.09 | 按查询给视频、音频、字幕、文本各模态打权重，再求球面质心；rank-8 LoRA，可训参数 4.8M | 五个基准中四个出现正的融合增益（最高 +4.0 R@1），此前对称聚合器至多一个 |
| [MoRe](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Compositional_Transformation_Reasoning_for_Composed_Video_Retrieval_CVPR_2026_paper.pdf) | CVPR 2026 | 免训练组合视频检索：多目标候选选择 + 实体、动作、场景三维成对推理 | EgoCVR / WebVid-CoVR R@1 提升 5.8 / 10.8 |
| [CoVR-R](https://arxiv.org/abs/2603.20190) | 2026.03 | 推理修改文本隐含的「后果」（状态变化、镜头、节奏）再检索 | 配套 2,800 条带推理标注的基准 |
| 长视频基准：[FLARE](https://arxiv.org/abs/2605.10228)、[ExtremeWhenBench](https://arxiv.org/abs/2606.12300)、[CoMET-Bench](https://arxiv.org/abs/2606.15320) | 2026.05–06 | 视听联合检索（10–60 分钟视频）；小时级时间定位；条件多事件定位 | 小时级定位 85% 的失败是找错时间窗；CoMET 上最好的方法 F1@0.5 只有约 16%–19% |

**最新进展**

1. **视频仍是总榜最弱的一项**。开放与工业模型的视频子项在 67–76 之间，明显低于图像和文档。
2. **文本侧训练就能大幅改善细粒度视频检索**（TARA），延续了 E5-V 的思路。
3. **一个末 token 装不下视频**。MARS 改为多层多槽；SCALAR 让查询决定各模态的权重。
4. **长视频与多事件检索刚有基准，检索器本身几乎没人做**。FLARE、ExtremeWhenBench、CoMET 的共同结论是瓶颈在「找对片段」。

---

### 5.7 音频与全模态

**任务**：文本、图像、视频、音频任意组合互相检索。WeMM、Ovis-VL 等主流图文模型不含音频。

| 模型 | 出处 | 做法要点 | 结果 |
|---|---|---|---|
| [ImageBind](https://github.com/facebookresearch/ImageBind) / [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) | CVPR 2023 / ICLR 2024 | 选一个锚点模态（图像或语言），其余模态各自与之配对训练，对齐可以传递 | 绑定范式的起点 |
| [Omni-Embed-Nemotron](https://arxiv.org/abs/2510.03458) | 2025，NVIDIA | Qwen2.5-Omni-3B 的 Thinker（去掉 Talker）；冻结音视频编码器，LLM 上 LoRA；主训几乎没用音视频样本 | 开放权重；MMEB-V3 43.60（Ovis 论文复测） |
| [OmniRet](https://arxiv.org/abs/2603.02098) | CVPR 2026 | 共享媒体重采样器把长模态 token 压成定长；注意力切片 Wasserstein 池化减轻单向量损失 | 系统处理文本 + 视觉 + 音频复合查询；另提 ACM 基准 |
| [OmniRetriever-7B](https://arxiv.org/abs/2605.26641) | 2026.05 | 用三模态融合嵌入（stop-gradient）当单模态嵌入的教师，加 Tuple-InfoNCE | Clotho / SoundDescs 上比 Gemini Embedding 2 高 13.3–18.0 R@1；发布 12 方向视听文检索基准 |
| [Ovis-Embedding-Omni-3B](https://arxiv.org/abs/2609.25165) | 2026.09 | 直接以 Qwen2.5-Omni-3B 为骨干，音频从一开始就在共享空间里，而不是给图文嵌入事后外接音频通路 | MMEB-V3 58.46（第二名 Tianmu-Emb-Uni 53.27，e5-omni-7B 47.14），六个模态组全部第一，音频组 50.08 |

**最新进展**

1. **全模态从「外挂音频塔」转向「直接用 Omni 骨干」**。Ovis-Omni-3B 以 3B 参数在 MMEB-V3 上领先 5 分以上。
2. **音频–语言对齐仍是共同瓶颈**。FLARE 的视听检索结论与 MMEB-V3 音频组的普遍低分一致。
3. **融合嵌入当老师**。OmniRetriever 把三模态联合嵌入的信号回灌给单模态。
4. Ovis 论文同时报了 MAEB（音频）和 MVEB。这里的 MVEB 是 MTEB 体系的 Massive Video Embedding Benchmark，和 §5.9 的视觉身份基准 [MVEB](https://arxiv.org/abs/2608.01794) 同名不同物。

---

### 5.8 推理密集检索

**任务**：查询与目标之间要靠推理才能对上（例如「能解释这张图里现象的那段文字」）。[MR²-Bench](https://arxivlens.com/paperview/details/mr-2-bench-going-beyond-matching-to-reasoning-in-multimodal-retrieval-4166-ffb0e88d) 上，匹配类任务约 77–78，推理类约 9.9。

| 工作 | 链接 | 核心想法 | 记住的一点 |
|---|---|---|---|
| **[UME-R1](https://arxiv.org/abs/2511.00405)** | [代码](https://github.com/XMUDeepLIT/UME-R1) | 生成式嵌入 + GRPO，统一判别与生成 | 开山，但双目标梯度易冲突、CoT 冗长 |
| **[TTE](https://arxiv.org/abs/2510.05014)**（Think-Then-Embed） | — | Reasoner 与 Embedder 解耦 | 大 Reasoner 可抬分；参数冗余 |
| **[Embed-RL](https://arxiv.org/abs/2602.13823)** | [代码](https://github.com/ZoengHN/Embed-RL) | Embedder 引导的 RL，生成可追溯证据 CoT | 让推理服务检索（ECCV 2026） |
| **[MMEmb-R1](https://arxiv.org/abs/2604.06156)** | — | 成对感知选推理路径 + 自适应决定是否推理 | 4B 达 MMEB-V2 71.2；延迟比盲目 CoT 低约 2.5× |
| **[PLUME](https://arxiv.org/abs/2604.02073)** | — | 连续隐状态的短 rollout 替代文本 CoT | 不到 10 步 latent，吞吐比显式 CoT 高 30× 以上 |
| **[RIME](https://arxiv.org/abs/2604.22280)** | — | 用检索友好的改写替代长 CoT | 微信视觉；对齐生成与判别空间 |
| **[LaME](https://arxiv.org/abs/2606.13061)** | [代码](https://github.com/PeppaWu/LaME) | 潜在空间思考 / 可学习推理 token | 与 PLUME 同属潜在推理 |
| **[DME](https://arxiv.org/abs/2608.02148)** 第二阶段 | — | 训练期证据型潜在推理 + 交叉条件重建 | 线上不加 CoT，却把细粒度判别灌进向量 |
| **[UniME-R1](https://arxiv.org/abs/2608.06060)** | — | Embedder–Adviser：针对初检候选生成检索中心的 CoT | 推理回答「检索器搞混了什么」 |
| **[UMER](https://arxiv.org/abs/2608.18504)** | — | 成对判别推理 + 同一 MLLM 同时学嵌入与重排 | 比较「这一对为什么匹配、差在哪」；推理预算可调 |
| **[ReLoop-UME](https://arxiv.org/abs/2607.28751)** | — | 固定 token 工作区，沿深度循环复用检索形成层 | 用深度计算换中间文本 |
| **[MoEMB](https://arxiv.org/abs/2609.08663)** | — | 不加推理 token，改加专家 | 3B 激活参数超过参数多 4 倍以上的 TTE 类方法 |

LamRA、CAFe 常作为指令感知与架构变体的对照基线。

**谱系直觉**

```text
直接编码（VLM2Vec / GME / Qwen3-VL-Embedding）
    → 显式长 CoT（UME-R1 / TTE）
        → 检索对齐的 CoT / 自适应调用（Embed-RL / MMEmb-R1）
            → 检索反馈式 CoT（UniME-R1）
            → 成对判别推理 + 嵌入 / 重排统一（UMER）
            → 改写接口（RIME）或潜在推理（PLUME / LaME / DME 训练期）
            → 深度循环复用（ReLoop-UME）
    ↔ 不加推理、改加容量：专家扩容（MoEMB）
```

**最新进展**：问题已从「能不能想」变成「想什么、何时想、要不要说出来、是否成对比较、能否用检索反馈」。MoEMB 给出另一个方向：不在推理 token 上花算力，而是加专家。部署侧优先看自适应调用、训练期潜在推理（DME）、改写或深度循环，以及 UMER 的可调预算，不要无脑堆 CoT。

---

### 5.9 实例级与个性化检索

**任务**：认出「就是这一个」，而不是「同一类」；个性化检索还要求它出现在文字描述的场景里。

| 工作 | 出处 | 任务与规模 | 做法 / 结果 |
|---|---|---|---|
| [ILIAS](https://openaccess.thecvf.com/content/CVPR2025/html/Kordopatis-Zilos_ILIAS_Instance-Level_Image_retrieval_At_Scale_CVPR_2025_paper.html) | CVPR 2025 | 1,000 个物体实例，1,232 张干净查询图、4,715 张实拍正例，另有 1 亿张 YFCC 干扰图 | 领域专精模型跨领域失败；带背景杂乱时，局部描述子重排仍关键 |
| [MVEB / VisID](https://arxiv.org/abs/2608.01794) | 2026.08 | 视觉身份：同人、同物、Re-ID、AIGC 身份保持 | 身份感知采样联合训练；MMEB 高不等于会认同一个体 |
| [FoundYou](https://arxiv.org/abs/2608.29917) | ECCV 2026 | 个性化分割 + 检索；提出多参考图的 few-shot 个性化检索 | 冻结 SAM 2-small + 5.9M 可训参数；ILIAS 上参考图 1→4 张，mAP 30.1→34.2 |
| [POLAR](https://openaccess.thecvf.com/content/CVPR2025/papers/Ryan_Improving_Personalized_Search_with_Regularized_Low-Rank_Parameter_Updates_CVPR_2025_paper.pdf) | CVPR 2025 Highlight | 个人概念 + 场景文本（PerVL-DeepFashion2、ConCon-Chi） | 每个概念在 CLIP 文本编码器末层学一个 rank-1 LoRA；场景 mRR 40.72 / 46.33 |
| [π-map](https://arxiv.org/abs/2510.05411) | arXiv 2025 | 同上，含 This-Is-My 视频 | 每个实例把图像嵌入映射成文本 token |
| [This-Is-My](https://openaccess.thecvf.com/content/CVPR2023/html/Yeh_Meta-Personalizing_Vision-Language_Models_To_Find_Named_Instances_in_Video_CVPR_2023_paper.html) | CVPR 2023 | 视频中的命名实例 + 场景描述 | 测试时为每个实例学 token |

相关基准：[PerVL-DeepFashion2](https://github.com/nvlabs/PerVLBenchmark)（ECCV 2022）、[ConCon-Chi](https://openaccess.thecvf.com/content/CVPR2024/html/Rosasco_ConCon-Chi_Concept-Context_Chimera_Benchmark_for_Personalized_Vision-Language_Tasks_CVPR_2024_paper.html)（CVPR 2024）。

**最新进展**：只认实例已经有多参考图、推理时不训练的方案（FoundYou）。带场景文本的个性化检索仍以逐概念优化为主（POLAR、π-map），免优化的 MLLM 方案还没有出现在顶会。通用多模态嵌入在实例身份上明显偏弱（MVEB）。

---

### 5.10 效率与部署

| 方向 | 代表 | 要点 |
|---|---|---|
| 稠密 + 稀疏同前向 | [UEmbed](https://arxiv.org/abs/2608.02583)（2026.08） | 输入末尾挂 N 个可学习 token，各自预测词表一个子集的稀疏权重；只用约 394 万公开数据；9B dense 71.8 / sparse 71.0，可同时走 ANN 与倒排 |
| 事后稀疏 | [PUMA](https://arxiv.org/abs/2608.25780)（2026.08） | 稀疏自编码器挂在现成嵌入上，不重训骨干；存储约 8–16×，大候选池打分最高约 25× |
| 视觉 token 压缩 | Magic-MM、EVIE、Light-ColPali | 见 §5.1、§5.5 |
| 小模型与边缘 | [Eddy-VL 1.9B](https://arxiv.org/abs/2607.16316)、NeoMME-260M、ColNanoVDR-149M | Eddy-VL 剪枝 + 分层蒸馏，保留 2B 教师约 91.7%（MMEB-V2 63.2） |
| 深度复用 | [ReLoop-UME](https://arxiv.org/abs/2607.28751) | 扩深度、不扩 token |
| 维度弹性 | MRL + QAT（Qwen3-VL-Embedding）、Prefix-MRL（EVIE）、低秩分解（Ovis） | 一个模型多种向量长度 |
| 升级兼容 | [MKP-Adapter](https://arxiv.org/abs/2609.16875)（2026.09） | MLLM 嵌入升级时只训适配器，保持与旧向量索引兼容 |
| 自适应计算 | MoEMB | 按查询难度分配专家计算 |

**最新进展**：多向量检索的瓶颈已从精度转到存储；维度弹性成为标配；查询侧蒸馏与按难度分配计算开始出现。

---

### 5.11 垂直领域

| 领域 | 代表 | 要点 |
|---|---|---|
| 推荐 | [CAMIE](https://arxiv.org/abs/2608.30255)（Snap，2026.08） | 用户共互动 item 对做对称 InfoNCE，把内容嵌入拉向转化行为；线上 CTR / CVR 提升 |
| 驾驶日志 | [TraVEL](https://arxiv.org/abs/2608.13495)（2026.08） | 在 Qwen3-VL-Embedding 上以自车轨迹作 GRPO 特权监督，专打「左转 vs 右转」这类运动捷径 |
| 电商属性 | [HMGCLIP](https://arxiv.org/abs/2608.24467)（2026.08） | 异构超图挖结构感知难负例，双粒度推理 |
| 城市地理 | [Geo-Embed](https://arxiv.org/abs/2608.03826)（2026.08） | 地理多模态嵌入 |
| 遥感区域 | [VLM2GeoVec](https://arxiv.org/abs/2512.11490)（2025.12） | 用坐标指定区域的区域级嵌入 |
| 素描检索 | [SeCo-SBIR](https://arxiv.org/abs/2608.03120)（2026.08） | 零样本素描检索仍以 CLIP + 提示学习为主，MLLM 嵌入尚少 |
| 站内业务 | WeMM、DME | 微信 26 任务站内集与 14 组线上 A/B 一致提升；抖音站内离线约 +2.92% |

---

### 5.12 横向对照：主流 MLLM 嵌入方案

| 模型 | MLLM 骨干 | 表示 / 微调 | 训练数据公开度 | 最关键创新 |
|------|-----------|-----------|----------------|------------|
| [E5-V](https://github.com/kongds/E5-V) | LLaVA-NeXT-Llama3-8B | last token；QLoRA；纯文本对比 | **高** | 不用图文对也能迁移多模态 |
| [MM-Embed](https://huggingface.co/nvidia/MM-Embed) | LLaVA-1.6 + NV-Embed | projector + LoRA；模态感知难负例 | **中** | 纠正跨模态模态偏置 |
| [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) | Qwen2-VL 2B/7B | bi-encoder 对比 | **中高** | 大规模融合模态合成 |
| [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) / V2 | Phi/LLaVA/Qwen2-VL | 指令 + GradCache | **高** | 任务统一格式 + MMEB |
| [Omni-Embed-Nemotron](https://huggingface.co/nvidia/omni-embed-nemotron-3b) | Qwen2.5-Omni Thinker | 冻编码器；LLM LoRA | **中高** | 文/图/音/视频统一开放权重 |
| [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) | Qwen3-VL 2B/8B | EOS；MRL；Reranker 蒸馏 | **低至中** | 自举挖数 + 蒸馏，开放权重标杆 |
| [Magic-MM-Embedding](https://arxiv.org/abs/2602.05275) | InternVL3 + VTC | 视觉 token×1/4；三阶段粗到细 | 中 | 压缩仍可涨分的共设计 |
| [TSEmbed](https://arxiv.org/abs/2603.04772) | MLLM + MoE-LoRA | 专家解耦任务冲突 + EANS | 中 | 任务级缩放 |
| [UEmbed](https://arxiv.org/abs/2608.02583) | Qwen3.5 2B/4B/9B | 稠密 + SPLADE 式稀疏同前向 | **高**（公开数据） | 解码器也能做多模态稀疏检索 |
| [DME](https://arxiv.org/abs/2608.02148) | 工业骨干 2B/9B | 对比预训练 + 训练期 latent | **低**（工业） | MMEB-V2 78.4；线上近对比编码器 |
| [WeMM-Embedding](https://arxiv.org/abs/2608.24053) | Qwen3.5 2B/4B/9B | last-token + MRL；分级相关 + 跨尺度蒸馏 | **中**（权重开源，语料含站内） | MMEB-V2 80.6；2B 已达 77.9 |
| [MoEMB](https://arxiv.org/abs/2609.08663) | MoE MLLM（3B 激活） | 单向量、非自回归；专家扩容 + 自适应计算 | **高**（公开 MMEB 系数据） | 公开数据口径下 MMEB-V2 / MRMR 最好 |
| [Ovis-Embedding](https://arxiv.org/abs/2609.25165) | Qwen2.5-Omni-3B；Qwen3.5 2B/9B | last-token、无投影头；focal loss + 相似度蒸馏；低秩压维 | 中（称将开源权重与配方） | VL-9B MMEB-V2 81.13；Omni-3B MMEB-V3 58.46 |
| [UMER](https://arxiv.org/abs/2608.18504) | MLLM | 成对判别推理 + 嵌入 / 重排互蒸馏 | 中（方法公开） | 同一骨干可调预算地召回 + 精排 |
| [MMEmb-R1](https://arxiv.org/abs/2604.06156) / [PLUME](https://arxiv.org/abs/2604.02073) / [UniME-R1](https://arxiv.org/abs/2608.06060) | Qwen3-VL 等 | 推理 / 潜在 / 反馈式 CoT + 对比 | 中（方法公开） | 把中间计算引入嵌入 |

---

### 5.13 训练 / 推理算力速查（公开披露汇总）

<a id="513-训练--推理算力速查公开披露汇总"></a>

> 下列数字来自各论文 / 官方文档的**公开配置**，用来建立数量级直觉。卡型（A100 / A800 / H100 / H20 / Mi250x）、是否 GradCache、是否 LoRA、数据规模都会让墙钟时间差很多；**不能直接横比谁更省**，但可以判断这是「单机可玩」「小集群」还是「工业级大练」。

| 工作 | 训练硬件（公开） | 关键配置印象 | 推理 / 部署印象 |
|------|------------------|------------|---------------|
| [VLM2Vec](https://arxiv.org/html/2410.05160) | **8×H100** | LoRA + GradCache，全局 batch≈1024，~2k step | 7B 级需单卡 24–48GB+ |
| [VLM2Vec-V2](https://huggingface.co/papers/2507.04590) | **8×H100** | 2B 骨干约 **15h / 1k step** | 2B 更友好 |
| [UniME](https://arxiv.org/abs/2504.17432) | **8×A100 80GB** | ZeRO-2；阶段一 ~1–2h；阶段二 ~1k step | 4–7B LoRA 可单卡推理 |
| [UniME-V2](https://arxiv.org/abs/2510.13515) | **8×A800 80GB** | LoRA r=16，batch≈1024，~2k step | 336² 分辨率降低显存 |
| [E5-V](https://arxiv.org/abs/2407.12580) | 低（文本 QLoRA） | ~27 万对、batch 768、~1k step；成本约降 **95%** | 推理时再挂视觉塔 |
| [TARA](https://arxiv.org/abs/2512.13511) | **8×RTX A6000** | 7B，文本三元组，约 **1 小时** | 与同骨干视频嵌入相同 |
| [ColPali](https://arxiv.org/abs/2407.01449) | **8 GPU**；总实验 ~**1405 Mi250x GPU-h** | LoRA r=32，1 epoch，单次训练 ~40 GPU-h 量级 | 单卡（如 L4）可编码 |
| [mmE5](https://arxiv.org/abs/2502.08468) | **64×A100 40GB** | Llama-3.2-11B-Vision + LoRA r=8，1 epoch | 11B 推理需多显存或量化 |
| [RzenEmbed](https://arxiv.org/abs/2510.27350) | **16×A800 80GB** | bf16 + grad checkpoint；两阶段数据 | 2B/7B 分档部署 |
| [WAVE-7B](https://arxiv.org/abs/2509.21990) | 对齐 **128×H20**；主训 **192×H20 ≈36h / 1 epoch** | LoRA r=128，per-device batch=1，总 batch=192 | 7B Omni 骨干，显存偏高 |
| [UME-R1](https://arxiv.org/abs/2511.00405) | SFT **2336 H20 GPU-h** + RL **1344 H20 GPU-h**（对照判别式 DUME 约 1487） | SFT ~5k step，batch 1024；RL ~11k 对 | 生成式嵌入推理更慢（需出 CoT） |
| [Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) | 训练细节未完全公开（工业级三阶段 + 3 亿合成） | 对标：8×80GB 级 LoRA 微调常见 | **2B≥8GB / 8B≥24GB**（官方参考） |
| [Gemini Embedding 2](https://arxiv.org/abs/2605.27295) | 未公开 | — | **API / Vertex AI** 按量 |
| [MuCo](https://arxiv.org/abs/2602.06393) | 同档 7B 多卡；有效 batch ×7，额外算力 ≤3% | 用训练组织换负例数 | 与同骨干 MLLM 嵌入类似 |
| [Magic-MM](https://arxiv.org/abs/2602.05275) | 视觉 token≈1/4，显著降训推成本 | 三阶段补回压缩损失 | 同骨干下延迟 / 显存更友好 |
| [UEmbed](https://arxiv.org/abs/2608.02583) | 公开数据 + LoRA；2B/4B/9B | 稠密 + 稀疏同前向 | 稀疏走倒排，稠密走 ANN |
| [WeMM-Embedding](https://arxiv.org/abs/2608.24053) | 工业级两阶段（卡时未完全公开） | 数亿对齐 + 精炼蒸馏 / 9B merging | 官方支持 HF / vLLM / SGLang |
| [DME](https://arxiv.org/abs/2608.02148) | 工业未公开卡时 | 第二阶段 latent 仅训练期 | 推理近标准对比编码器 |
| [NeoMME](https://arxiv.org/abs/2609.01657) | 从零训 260M/800M | 掩码离散扩散 + 检索双头 | 文档页编码吞吐高于改 VLM 的 Col 系 |
| [Eddy-VL](https://arxiv.org/abs/2607.16316) | 教师 2B → 学生 1.9B 蒸馏 | 剪 4 层 + 分层蒸馏 | 边缘 / 离线友好（~3.85GB bf16） |
| [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2) | 双塔 ~0.9B，远低于 7B MLLM | 三阶段分辨率课程 | 消费级 / 中小 GPU 可服务 |
| [CLIP](https://arxiv.org/abs/2103.00020) 原始预训练 | 大规模（数百 GPU 量级，历史配置） | batch 32768 | 推理很轻；预训练不可轻易复现 |

**读表建议**

1. **复现 VLM2Vec / UniME 级工作**：先准备 **8×80GB**（H100 / A100 / A800 一类），上 LoRA + GradCache。
2. **只想用、不想训**：优先 API（Gemini）或直接拉 HF 权重；注意 7B/8B 推理显存。
3. **预算极紧**：从 E5-V、TARA、Jina CLIP v2、ColPali 小骨干入手。
4. **视频 + 音频统一（WAVE）或生成式推理嵌入（UME-R1）**：算力跳到百卡级或数千 GPU-hours。
5. **合成数据流水线**（Qwen3-VL-32B 标数、GPT-4o 合成）往往比最后那次嵌入微调更吃钱，算总账时别漏掉。

---

### 5.14 章节小结：怎么读下一篇新论文

遇到新的多模态嵌入工作，可以用七问快速读透：

0. **任务**：它在本章哪一类任务里比？总榜分数高不代表视频、文档、推理或实例身份也强。
1. **范式**：双塔 / MLLM 适配 / 原生融合 / 多向量 / 稠密 + 稀疏 / 事后稀疏 / MoE 扩容 / 推理增强（显式、潜在、改写、反馈、成对、深度循环）？
2. **损失**：InfoNCE、SigLIP，还是任务混合 / RL 奖励 / 重建辅助 / Rank-KL / 分级 CoSENT / focal loss？
3. **数据**：真数据、合成、难负例、课程；**训练语料公开到什么程度**？（工业 DME / WeMM 与公开数据的 UEmbed / MoEMB 不可盲比）
4. **表示**：单向量还是多向量？是否 MRL / 量化 / 稀疏？推理时要不要生成中间文本或成对比较？
5. **评测**：只看 MMEB 匹配，还是也看了 MR²、ViDoRe、CMDR、视觉身份 MVEB、组合性（CORE）、长上下文、MMEB-V3？
6. **效率**：相对基线的延迟、token 数、存储（尤其多向量、长 CoT、视觉 token 压缩、事后稀疏）？

把这七问答清，一篇方法文的主线通常就出来了。

---



## 6. 开源训练框架与后训练上手

> 面向「已有开源 Embedding 权重，要在业务/领域数据上做后训练（继续对比微调）」的读者。先选对仓库，再按最小闭环跑通：数据 → LoRA InfoNCE → 业务检索集评测。

### 6.1 通用训练框架（优先看这些）

| 框架 / 仓库 | 链接 | 模态重心 | 适合做什么后训练 | 上手成本 |
|-------------|------|----------|------------------|----------|
| **[ms-swift](https://github.com/modelscope/ms-swift)** | [Embedding 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md) · [Qwen3 示例](https://github.com/modelscope/ms-swift/tree/main/examples/train/embedding/qwen3) · [GME 脚本](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding/train_gme.sh) | 文本 + 多模态（官方已接 Qwen3 / Qwen3-VL / GME / GTE） | **生产向后训练首选**：LoRA/全参、`infonce`、多卡 DeepSpeed | 低 |
| **[VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec)** | 训练示例：`experiments/public/train` | 图 / 视频 / 视觉文档 | 复现 MMEB 管线、自定义 DataLoader、指令对比学习 | 中 |
| **[Tevatron 2.0](https://github.com/texttron/tevatron)** | [论文](https://arxiv.org/abs/2505.02466) | 文本为主，扩展多模态 | 大规模检索训练/评测统一管线；OmniEmbed 基线 | 中 |
| **[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)** | BGE / [BGE-VL](https://huggingface.co/collections/BAAI/megapairs-67c6bbe49c15a9e7a7c69d92) / [MegaPairs](https://github.com/VectorSpaceLab/MegaPairs) | 文本 + 图文（BGE-VL） | BGE 系微调与难负例挖掘；视觉检索可看 MegaPairs | 中 |
| **[Sentence-Transformers](https://www.sbert.net/)** | 多模态 finetune 文档 | 文本 / 图 / 音视频（视模型） | API 友好、小规模实验；已支持加载部分 VL Embedding | 低 |
| **[colpali-engine](https://github.com/illuin-tech/colpali)** | `pip install "colpali-engine[train]"` · [cookbook](https://github.com/tonywu71/colpali-cookbooks) | 视觉文档（多向量） | ColPali / ColQwen 族 LoRA 后训练 | 中 |
| **[OpenCLIP](https://github.com/mlfoundations/open_clip)** / [big_vision](https://github.com/google-research/big_vision) | SigLIP / SigLIP 2 配置 | 双塔图文 | CLIP 系继续预训练/微调，偏吞吐与索引 | 中高 |

> 说明：ms-swift 当前明确支持的是 **LLM/MLLM 系 Embedding**（含把任意 chat MLLM 用 `--task_type embedding` 改造成编码器）；**不覆盖经典 CLIP 双塔训练**。双塔请走 OpenCLIP / big_vision。

### 6.2 代表模型自带训练仓库（按任务抄作业）

| 工作 | 训练代码 | 数据公开度 | 备注 |
|------|----------|------------|------|
| [VLM2Vec / V2](https://github.com/TIGER-AI-Lab/VLM2Vec) | 完整 train + eval | **高**（MMEB-train 等） | 社区事实标准；GradCache + LoRA |
| [UniME](https://github.com/deepglint/UniME) / [UniME-V2](https://github.com/GaryGuTC/UniME-v2) | V2 有完整训练脚本 | 中高 | 蒸馏 + 硬负例；基于 VLM2Vec/ms-swift 生态 |
| [E5-V](https://github.com/kongds/E5-V) | 文本对 QLoRA 脚本 | **高** | 极低成本入门；验证「文本判别力迁移」 |
| [mmE5](https://github.com/haon-chen/mmE5) | 合成 + 训练 | 中 | 学合成数据质量流水线 |
| [RzenEmbed](https://github.com/360CVGroup/RzenEmbed) | 公开代码 | 中 | 视觉文档向 |
| [UME-R1](https://github.com/XMUDeepLIT/UME-R1) / [UniME-R1](https://arxiv.org/abs/2608.06060) | SFT+RL / Embedder–Adviser | 中 | 生成式 / 检索反馈式推理嵌入 |
| [Embed-RL](https://github.com/ZoengHN/Embed-RL) | EG-RL 训练 | 中 | Reasoner–Embedder 解耦 |
| [UEmbed](https://github.com/Alibaba-NLP/UEmbed) | 公开训练配方 + 权重 | **高** | 稠密+稀疏统一；公开数据复现友好 |
| [WeMM-Embedding](https://github.com/Tencent/WeMM-Embedding) | 官方侧重推理/权重 | 中（站内+合成未完整公开） | **开放权重新标杆**；后训练仍可走 ms-swift |
| [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) | 官方侧重推理/权重；**后训练推荐走 ms-swift** | 训练语料未完整公开 | 权重开源，配方可复用 |
| [Nemotron ColEmbed V2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) | 权重开源；训练细节见论文 | 中 | ViDoRe 向 late interaction 标杆 |
| [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) | 可用 [ms-swift train_gme.sh](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding/train_gme.sh) | 中高 | 融合模态合成思路可借鉴 |
| [MM-Embed](https://huggingface.co/nvidia/MM-Embed) | 权重开源，完整混合训练包未全公开 | 中 | 可参考模态感知难负例设计 |
| [Omni-Embed-Nemotron](https://huggingface.co/nvidia/omni-embed-nemotron-3b) | **训练代码未公开** | 中高（数据列表有） | 只适合当权重基线，不适合抄训练 |
| [DME](https://arxiv.org/abs/2608.02148) | **工业未开源权重** | 低 | 抄「训练期 latent、推理期对比」思路即可 |
| [NeoMME](https://huggingface.co/collections/Hcompany/neomme) | 预训练骨干 + Retriever 权重 | **高** | 文档向小单塔；稠密+多向量同前向 |
| ColPali / ColQwen | [illuin-tech/colpali](https://github.com/illuin-tech/colpali) | 中 | PDF/页级多向量后训练 |

### 6.3 怎么选框架（决策表）

| 你的目标 | 推荐起点 | 基座权重建议 |
|----------|----------|--------------|
| 业务域多模态 RAG / 图文检索，尽快出效果 | **ms-swift** + InfoNCE + LoRA | 直接用 [WeMM-Embedding-2B/9B](https://github.com/Tencent/WeMM-Embedding) 作召回；若要继续后训练，[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 生态更熟 |
| 对齐论文 / 刷 MMEB 子任务 | **VLM2Vec** | VLM2Vec-V2 或自选 Qwen2-VL/Qwen3-VL |
| 纯 PDF / 扫描件检索 | **colpali-engine**、NeoMME 或 [EVIE 官方仓库](https://github.com/Tencent/EVIE) | [EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B) / ColQwen3 / [Nemotron ColEmbed V2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) / [NeoMME-Retriever](https://huggingface.co/Hcompany/NeoMME-260M-Retriever) |
| 需要倒排 + 向量混合检索 | **UEmbed** 权重直用或仿其稀疏头；已有稠密向量可试 [PUMA](https://arxiv.org/abs/2608.25780) | [Alibaba-NLP/UEmbed](https://github.com/Alibaba-NLP/UEmbed) 2B/4B/9B |
| 只要文本检索后训练 | ms-swift（Qwen3-Embedding）或 FlagEmbedding / SBERT | Qwen3-Embedding-0.6B/4B |
| 大规模离线图库双塔 | OpenCLIP / SigLIP 2 | SigLIP 2 So400m/g |
| 边缘 / 离线部署 | 蒸馏小模型 | [Eddy-VL 1.9B](https://arxiv.org/abs/2607.16316) 思路或 2B + 量化 |
| 研究推理增强嵌入 | UME-R1 / Embed-RL / UniME-R1 / UMER | 对应论文骨干 |

### 6.4 快速上手：用 ms-swift 后训练 Qwen3-VL-Embedding（推荐路径）

下面是一条可在 **1–8×80GB GPU** 上跑通的最小闭环（默认 LoRA + InfoNCE）。细节以 [ms-swift Embedding 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md) 与 [examples/train/embedding/qwen3](https://github.com/modelscope/ms-swift/tree/main/examples/train/embedding/qwen3) 为准。

#### Step 0：环境

```bash
pip install ms-swift -U
pip install transformers -U deepspeed
# 可选但强烈建议（加速注意力）
pip install flash-attn --no-build-isolation
```

硬件经验：2B + LoRA 单卡 24–48GB 可试；有效 batch 靠 `gradient_accumulation`；对比学习仍吃大 batch，多卡更稳。

#### Step 1：准备领域数据（JSONL）

InfoNCE 最小样本（query–positive；负例可先靠 in-batch）：

```json
{"messages": [{"role": "user", "content": "<image>找出与描述匹配的商品图"}], "images": ["/data/q1.jpg"], "positive_messages": [[{"role": "user", "content": "红色陶瓷马克杯，木桌上，柔和晨光"}]]}
{"messages": [{"role": "user", "content": "发票总金额是多少？"}], "positive_messages": [[{"role": "user", "content": "<image>"}], "positive_images": [["/data/invoice_p3.png"]]]}
```

有难负例时加上 `negative_messages` / `negative_images`（list-of-list）。规则摘要：

- `messages` = anchor（query）
- `positive_messages` 外层长度必须为 **1**
- `<image>` 个数与 `images` / `positive_images` 对齐
- 可用 `system` 写入任务 instruction（Qwen3 系会拼进 query）

**数据量建议**：先 **3k–20k** 高质量对跑通；宁可少而难，也不要海量易负例。业务评测集务必 **与训练 query 隔离**。

#### Step 2：启动后训练

```bash
export INFONCE_TEMPERATURE=0.07
export INFONCE_USE_BATCH=True
# 可选：缓解假负例
# export INFONCE_MASK_FAKE_NEGATIVE=True

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --task_type embedding \
  --train_type lora \
  --dataset /path/to/train.jsonl \
  --split_dataset_ratio 0.05 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --loss_type infonce \
  --max_length 2048 \
  --dataloader_drop_last true \
  --deepspeed zero2 \
  --output_dir output/qwen3vl-emb-2b-lora \
  --save_steps 200 \
  --eval_steps 200 \
  --save_total_limit 3
```

纯文本 Qwen3-Embedding 时增加 `--model_type qwen3_emb`，并把 `--model` 换成对应文本权重。GME 可直接参考官方 `train_gme.sh`。

#### Step 3：合并 LoRA 并做「真变化」自检

```bash
swift export \
  --adapters output/qwen3vl-emb-2b-lora/checkpoint-xxx \
  --merge_lora true \
  --output_dir output/qwen3vl-emb-2b-merged
```

用同一批 query 比较 **基座 vs 合并后** 的向量余弦：若几乎恒为 1.0，多半是 **adapter 没加载/没合并成功**（ms-swift Issue 里常见坑），不要先怀疑数据。

#### Step 4：评测闭环（比只看 train loss 重要）

1. **业务集**：固定候选库，报 Recall@K / nDCG@K（与线上一致的指令模板）。  
2. **回归集**：抽一小撮 MMEB / 通用检索子集，防止灾难性遗忘。  
3. **文档场景**：另报 ViDoRe 或自建页级集；多向量模型用 colpali 评测脚本。

### 6.5 备用快速路径（按场景）

**A. 复现论文管线（VLM2Vec）**

```bash
git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git
cd VLM2Vec
# 按 README 安装；训练脚本见 experiments/public/train
# 数据：TIGER-Lab/MMEB-train + 自有 json 注册到 src/dataset/
```

适合：要 GradCache、多任务 instruction、和 MMEB 完全对齐时。

**B. 视觉文档（ColQwen / ColPali）**

```bash
pip install "colpali-engine[train]"
# 配置见仓库 scripts/configs/；多卡示例：
accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py
```

适合：PDF/PPT 页图检索，而不是通用 MMEB 全能嵌入。

**C. 从聊天 MLLM「冷启动」成 Embedder**

若不想从现成 Embedding 权重出发，可用 ms-swift：`--model <Qwen2.5-VL/Qwen3-VL-Instruct> --task_type embedding`，再 InfoNCE。代价是要比「已对齐的 Embedding 2B/8B 继续训」更吃数据与步数。

### 6.6 后训练实操清单（建议打印）

1. **先定任务**：文找图 / 图找文 / 融合 query / PDF 页检索——任务决定框架（ms-swift vs colpali）。  
2. **先定基座**：通用多模态优先 [WeMM-Embedding](https://github.com/Tencent/WeMM-Embedding)（要继续后训练则 Qwen3-VL-Embedding 更熟）；文档优先 ColQwen / NeoMME。  
3. **数据**：instruction 固定；正例准；难负例逐步加；剔除假负例。  
4. **训练**：LoRA → 有效 batch（积充分步）→ `infonce` → 1 个 epoch 探路。  
5. **验证**：合并权重后向量必须相对基座有偏移；业务集 + 回归集双看。  
6. **上线**：需要维度压缩时再开 MRL/截断；召回不够再叠 Reranker，而不是无限加大 Embedding。

### 6.7 常见坑

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 训完召回几乎不变 | LoRA 未合并 / 推理未挂 adapter | `swift export --merge_lora true`；核对 key 前缀 |
| Loss 很好、业务集崩 | 指令模板不一致；过拟合训练域 | 推理用同一 system/instruction；加通用数据混训 |
| OOM | 图分辨率/视频帧 × batch | 降分辨率、降 `max_length`、升 grad accum、ZeRO-2/3 |
| 负例越挖越差 | 假负例被当负例推远 | `INFONCE_MASK_FAKE_NEGATIVE` 或 Judge 过滤 |
| 多模态涨、纯文本掉 | 对齐税 | 混入文本检索数据持续微调（MM-Embed 思路） |

---



## 7. 趋势与展望



### 7.1 当前关键挑战

1. **推理鸿沟**：[MR²-Bench](https://arxivlens.com/paperview/details/mr-2-bench-going-beyond-matching-to-reasoning-in-multimodal-retrieval-4166-ffb0e88d) 揭示的表面匹配（MMEB ~77–78）与推理检索（~9.9）之间约 68pp 差距，仍是最紧迫问题；推理增强嵌入刚开始缩小缺口，远未「解决」。
2. **模态指令不可靠**：[MMEB-V3 / OmniSET](https://arxiv.org/abs/2604.23321) 显示模型难以按指令约束目标模态，跨模态检索高度不对称。
3. **组合性理解**：[Winoground](https://huggingface.co/datasets/facebook/winoground) 分数（30–40%）表明空间关系、属性绑定仍弱；[CORE](https://arxiv.org/abs/2609.04083) 说明 reranker 已经会、嵌入还不会，缺口可以蒸馏。
4. **实例身份 vs 语义匹配**：[MVEB](https://arxiv.org/abs/2608.01794) 表明通用 UME 不等于会做 VisID / Re-ID。
5. **长上下文退化**：[MMLongEmbed](https://arxiv.org/abs/2606.14747) 发现上下文越长、性能越差的系统性衰变；文档侧 [CMDR](https://arxiv.org/abs/2607.05927) 进一步暴露跨页聚合缺口。
6. **多模态对齐税**：增加模态支持通常伴随单一模态（尤其纯文本）性能下降。
7. **存储与推理效率**：多向量（[ColPali](https://huggingface.co/vidore/colpali) / [Nemotron ColEmbed](https://arxiv.org/abs/2602.03992)）与长 CoT 生成式嵌入分别在存储与延迟上承压；[Magic-MM](https://arxiv.org/abs/2602.05275) token 压缩、[OmniRet](https://arxiv.org/abs/2603.02098) 重采样、[PLUME](https://arxiv.org/abs/2604.02073)/[DME](https://arxiv.org/abs/2608.02148) 潜在推理、[ReLoop-UME](https://arxiv.org/abs/2607.28751) 深度循环、[PUMA](https://arxiv.org/abs/2608.25780) 事后稀疏、[NeoMME](https://arxiv.org/abs/2609.01657) 小单塔是几类对冲思路。



### 7.2 未来方向

1. **推理增强且可部署**：[MMEmb-R1](https://arxiv.org/abs/2604.06156) 自适应调用、[PLUME](https://arxiv.org/abs/2604.02073) 潜在推理、[RIME](https://arxiv.org/abs/2604.22280) 改写、[DME](https://arxiv.org/abs/2608.02148) 训练期-only latent、[UniME-R1](https://arxiv.org/abs/2608.06060) 检索反馈式 RC-CoT、[UMER](https://arxiv.org/abs/2608.18504) 成对判别 + 可调预算，比「永远生成长 CoT」更贴近线上。
2. **生成式嵌入与判别式对齐**：CMA / 蒸馏 / Refine-RL / 交叉条件重建一类机制，让「想清楚」与「嵌得准」可互相检索、互为锚点。
3. **稠密–稀疏统一与混合检索**：[UEmbed](https://arxiv.org/abs/2608.02583) 证明解码器可一次出两套表示；[PUMA](https://arxiv.org/abs/2608.25780) 证明已有稠密向量也能事后变稀疏。
4. **全模态（omni）与 Agent 嵌入**：音频、工具检索、GUI、记忆检索（MMEB-V3）；[Omni-Embed-Nemotron](https://arxiv.org/abs/2510.03458) / [WAVE](https://arxiv.org/abs/2509.21990) / [OmniRet](https://arxiv.org/abs/2603.02098) 是开源参照。注意 [WeMM](https://arxiv.org/abs/2608.24053) **不含音频**。
5. **任务缩放与冲突消解**：[TSEmbed](https://arxiv.org/abs/2603.04772) 的 MoE-LoRA + EANS 指向「加任务不必互相伤害」。
6. **合成数据与挖负闭环**：[Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720)、[GME](https://arxiv.org/abs/2412.16855)、[PolyGen](https://arxiv.org/abs/2602.01370)、Magic-MM 的 Judge 精筛、WeMM 的 Semantic-ID 重采样证明数据配方常比单纯堆参数更决定排名。
7. **原生深度融合**：[Gemini Embedding 2](https://arxiv.org/abs/2605.27295) 的早期融合仍是能力上限参考。
8. **量化与压缩**：MRL + QAT，以及 [SMEC](https://aclanthology.org/2025.emnlp-main.1332/)、[CSR](https://icml.cc/virtual/2025/oral/47166)、[MIPIC](https://github.com/tmp0810/MIPIC)、视觉 token 压缩、边缘蒸馏（Eddy-VL）、事后稀疏（PUMA）、文档多向量 255× 压缩（NeoMME）。
9. **专业领域**：数学图、科学插图、化学结构、城市地理（[Geo-Embed](https://arxiv.org/abs/2608.03826)）、驾驶视频（[TraVEL](https://arxiv.org/abs/2608.13495)）、电商属性（[HMGCLIP](https://arxiv.org/abs/2608.24467)）等垂直多模态嵌入仍有明显空间。
10. **组合性专项**：不要只靠 MMEB 总分；[CORE](https://arxiv.org/abs/2609.04083) 把 reranker 蒸馏当作组合性的可复制配方。



### 7.3 选型建议


| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 通用多模态检索（开放权重） | [WeMM-Embedding-9B](https://arxiv.org/abs/2608.24053) | MMEB-V2 **80.6**；2B/4B 亦已很强；MRL。[Ovis-Embedding-VL-9B](https://arxiv.org/abs/2609.25165) 自报 81.13，开源后复测再换 |
| 只用公开数据复现 / 研究扩容 | [MoEMB](https://arxiv.org/abs/2609.08663) / [UEmbed](https://arxiv.org/abs/2608.02583) | 公开数据口径下最好；专家扩容保持单向量 |
| 工业上限 / 视频+文档极强 | [DME](https://arxiv.org/abs/2608.02148)（若可合作调用） | 报告 78.4，活榜 DME-Large≈80.2；训练期 latent、线上可服务 |
| 预算有限的高精度 | [WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) / [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) / DME-2B | WeMM-2B 总分已≈原 8B 标杆 |
| 文档/PDF 检索 | [EVIE-8B / 4.5B](https://huggingface.co/tencent/EVIE-8B) / [Nemotron ColEmbed V2](https://arxiv.org/abs/2602.03992) / [NeoMME](https://arxiv.org/abs/2609.01657) / [Argus-9B](https://arxiv.org/abs/2606.04300) | ViDoRe V3 榜首为 EVIE（每页可压到 32/64 个向量）；小参数选 NeoMME；查询延迟敏感可配 [ColNanoVDR](https://arxiv.org/abs/2609.34899) 蒸馏的查询编码器 |
| 跨页文档问答检索 | [CMDR-Embed](https://arxiv.org/abs/2607.05927) 思路 | 需要页间上下文时不要只用单页 Col 模型 |
| 混合检索（稠密+稀疏） | [UEmbed](https://arxiv.org/abs/2608.02583) / [PUMA](https://arxiv.org/abs/2608.25780) | 训练期原生稀疏 vs 事后稀疏 |
| 视频理解 | [WAVE-7B](https://arxiv.org/abs/2509.21990) / WeMM / DME / Qwen3-VL-Embedding | 视频仍是普遍短板，优先专项或二次微调 |
| 视频细粒度（时序、否定） | [TARA](https://arxiv.org/abs/2512.13511) / [MARS](https://arxiv.org/abs/2609.02565) | TARA 只用文本难负例，8 卡约 1 小时；MARS 用多层多槽替代单 token |
| 文/图/音/视频统一 | [Ovis-Embedding-Omni-3B](https://arxiv.org/abs/2609.25165) / [OmniRetriever-7B](https://arxiv.org/abs/2605.26641) / [Omni-Embed-Nemotron](https://huggingface.co/nvidia/omni-embed-nemotron-3b) / [OmniRet](https://arxiv.org/abs/2603.02098) | Ovis-Omni-3B 在 MMEB-V3 领先 5 分以上（称将开源）；WeMM、Ovis-VL 无音频 |
| 组合图像检索 | [ReCALL](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_ReCALL_Recalibrating_Capability_Degradation_for_MLLM-based_Composed_Image_Retrieval_CVPR_2026_paper.html) / 免训练 [G-MIXER](https://openaccess.thecvf.com/content/CVPR2026/papers/Lim_G-MIXER_Geodesic_Mixup-based_Implicit_Semantic_Expansion_and_Explicit_Semantic_Re-ranking_CVPR_2026_paper.pdf) | 单向量召回 + MLLM 重排更稳 |
| 推理密集检索（研究） | [Embed-RL](https://arxiv.org/abs/2602.13823) / [MMEmb-R1](https://arxiv.org/abs/2604.06156) / [PLUME](https://arxiv.org/abs/2604.02073) / [UniME-R1](https://arxiv.org/abs/2608.06060) / [UMER](https://arxiv.org/abs/2608.18504) | 先评估延迟；部署侧看好 latent/自适应/成对可调预算 |
| 组合性 / 属性绑定 | [CORE](https://arxiv.org/abs/2609.04083) 思路 | 不要只看 MMEB 总分 |
| 实例身份 / 同款检索 | VisID 微调 + [MVEB](https://arxiv.org/abs/2608.01794) 回归 | 不要只看 MMEB 总分 |
| 边缘离线 | [Eddy-VL](https://arxiv.org/abs/2607.16316) / [NeoMME-260M](https://huggingface.co/Hcompany/NeoMME-260M-Retriever) 或 2B+量化 | 参数与延迟优先 |
| 多语言商用 API | [Gemini Embedding 2](https://arxiv.org/abs/2605.27295) | 闭源综合上限 |
| 大规模离线索引 | [SigLIP 2](https://arxiv.org/abs/2502.14786) / [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2) | 双塔吞吐高、易 ANN |


---

## 8. 附录：两份原稿整合说明与新增文献

### 8.1 整合策略

| 原稿 | 角色 | 主要并入内容 |
|------|------|--------------|
| 系统综述版（本文件前身） | **主骨架** | 数据集 Demo、评测榜、训练 blog、多数代表模型、算力表 |
| 代表作解析版（已删除） | **深度补丁** | SigLIP 2、GME/UMRB、Omni-Embed-Nemotron、训练数据披露边界、WeMM 榜位、横向对照表 |

现仅保留本文件为唯一报告。

### 8.2 本次补充的新工作（相对原两稿）

| 工作 | 时间/出处 | 为何值得关注 |
|------|-----------|--------------|
| [MMEB-V3 + OmniSET](https://arxiv.org/abs/2604.23321) | 2026.04 / COLM 2026 | 全模态评测 + 模态偏置诊断 |
| [OmniRet](https://arxiv.org/abs/2603.02098) | CVPR 2026 | 文–视–音复合检索 + 高效池化 |
| [Embed-RL](https://arxiv.org/abs/2602.13823) | ECCV 2026 | 检索对齐的 Traceability CoT |
| [MMEmb-R1](https://arxiv.org/abs/2604.06156) | 2026.04 | 成对选推理 + 自适应；4B→71.2 |
| [PLUME](https://arxiv.org/abs/2604.02073) | 2026.04 | 潜在推理，大幅降延迟 |
| [RIME](https://arxiv.org/abs/2604.22280) | 2026.04 / 微信视觉 | rewrite 替代冗长 CoT |
| [TTE](https://arxiv.org/abs/2510.05014) | 2025–2026 | Reasoner–Embedder 解耦范式参照 |
| [SigLIP 2](https://arxiv.org/abs/2502.14786) | 2025（原稿 B 已有，系统版补全） | 现代双塔默认骨干 |
| [Magic-MM-Embedding](https://arxiv.org/abs/2602.05275) | 2026.02 | 视觉 token×1/4 + 三阶段恢复 |
| [Nemotron ColEmbed V2](https://arxiv.org/abs/2602.03992) | 2026.02 | ViDoRe V3 开放权重榜首级 |
| [TSEmbed](https://arxiv.org/abs/2603.04772) | 2026.03 | MoE-LoRA 解任务冲突 + EANS |
| [CMDR / CMDR-Bench](https://arxiv.org/abs/2607.05927) | 2026.07 | 跨页上下文文档检索 |
| [Eddy-VL 1.9B](https://arxiv.org/abs/2607.16316) | 2026.07 | 边缘可部署剪枝蒸馏 |
| [ReLoop-UME](https://arxiv.org/abs/2607.28751) | 2026.07 | 深度循环 + Retrieval Registers |
| **[DME](https://arxiv.org/abs/2608.02148)** | **2026.08 / 抖音** | MMEB-V2 **78.4**；训练期 latent |
| **[UEmbed](https://arxiv.org/abs/2608.02583)** | **2026.08** | 稠密+稀疏统一；公开数据可复现 |
| **[MVEB / VisID](https://arxiv.org/abs/2608.01794)** | **2026.08** | 视觉身份专项基准 |
| **[UniME-R1](https://arxiv.org/abs/2608.06060)** | **2026.08** | 检索反馈式 RC-CoT（Learning from Failures） |
| **[WeMM-Embedding](https://arxiv.org/abs/2608.24053)** | **2026.08 / 微信** | 开源 2B/4B/9B；MMEB-V2 **80.6** |
| **[UMER](https://arxiv.org/abs/2608.18504)** | **2026.08** | 成对判别推理 + 嵌入/重排统一 |
| **[PUMA](https://arxiv.org/abs/2608.25780)** | **2026.08** | 事后稀疏化，不重训骨干 |
| **[CORE](https://arxiv.org/abs/2609.04083)** | **2026.09** | Rank-KL 蒸馏组合性 |
| **[NeoMME](https://arxiv.org/abs/2609.01657)** | **2026.09** | 从零训小单塔；ViDoRe 高吞吐 |
| **[MoEMB](https://arxiv.org/abs/2609.08663)** | **2026.09** | MoE 专家扩容；公开数据口径 MMEB-V2 / MRMR 最好 |
| **[Ovis-Embedding](https://arxiv.org/abs/2609.25165)** | **2026.09** | VL-9B MMEB-V2 81.13；Omni-3B MMEB-V3 58.46 |
| **[EVIE](https://huggingface.co/tencent/EVIE-8B)** | **2026.09 / 腾讯** | ViDoRe V3 榜首；Prefix-MRL + 免训练层次聚类压缩 |
| **[ColNanoVDR](https://arxiv.org/abs/2609.34899)** | **2026.09** | 不编码文档页的多向量查询侧蒸馏 |
| **[MARS](https://arxiv.org/abs/2609.02565)** | **2026.09 / EMNLP 2026** | 多层多槽视频检索表示 |
| **[SCALAR](https://arxiv.org/abs/2609.15335)** | **2026.09** | 查询条件化的多模态加权聚合 |
| **[MKP-Adapter](https://arxiv.org/abs/2609.16875)** | **2026.09** | 嵌入升级时只训适配器保持旧索引兼容 |
| **[TARA](https://arxiv.org/abs/2512.13511)** | **ECCV 2026** | 纯文本难负例训练的细粒度视频检索 |
| **[FoundYou](https://arxiv.org/abs/2608.29917)** | **ECCV 2026** | 多参考图、免训练的个性化实例检索 |
| **[ReCALL](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_ReCALL_Recalibrating_Capability_Degradation_for_MLLM-based_Composed_Image_Retrieval_CVPR_2026_paper.html)** / **[G-MIXER](https://openaccess.thecvf.com/content/CVPR2026/papers/Lim_G-MIXER_Geodesic_Mixup-based_Implicit_Semantic_Expansion_and_Explicit_Semantic_Re-ranking_CVPR_2026_paper.pdf)** | **CVPR 2026** | MLLM 组合检索的能力退化修复；免训练组合检索 |
| **[OmniRetriever](https://arxiv.org/abs/2605.26641)** | **2026.05** | 融合嵌入当教师的视听文检索 |

> 检索截止约 **2026-10-08**。之后若有新的 MMEB 登顶开放模型，请以 [MMEB Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard) 为准更新第 3 章表格。

---

> **报告生成日期**：2026 年 7 月 5 日  
> **修订**：2026-07-24（第 2–5 章扩写与算力对照）；2026-08-01（两稿整合 + 推理增强/全模态新工作）；2026-08-01（新增第 6 章：开源训练框架与后训练上手）；2026-08-12（补充 DME / UEmbed / Magic-MM / TSEmbed / ReLoop-UME / Eddy-VL / CMDR / MVEB / UniME-R1，并刷新 MMEB-V2 与选型表）；2026-09-08（补充 WeMM-Embedding / UMER / CORE / PUMA / NeoMME，并刷新 MMEB-V2 与选型表）；**2026-10-09（第 5 章改为按任务类别组织代表模型与最新进展；补充 Ovis-Embedding / MoEMB / EVIE / ColNanoVDR / ReCALL / G-MIXER / TARA / MARS / SCALAR / OmniRetriever / FoundYou 等，并刷新 MMEB-V2、ViDoRe V3 与选型表）**  
> **覆盖时段**：2025 年 1 月 – 2026 年 10 月初  
>
> 本文基于公开研究文献整理；各工作名称已在正文中直接链接至论文 / 代码 / 官方页面。第 3 章各榜为公开论文/模型卡/榜单快照，**活榜会持续刷新**；请以 [MMEB Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)、[MTEB](https://huggingface.co/spaces/mteb/leaderboard)、[ViDoRe](https://mteb-leaderboard.hf.space/benchmark/ViDoRe(v3))、[MR²-Bench](https://github.com/VectorSpaceLab/MR2-Bench) 等官方页面为准。

