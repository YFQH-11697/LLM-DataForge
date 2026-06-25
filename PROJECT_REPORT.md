# LLM-DataForge 项目解析报告

## 1. 项目背景

大模型数据算法工程师岗位关注的重点通常不是普通 RAG Demo 或 Agent 应用，而是预训练/后训练数据层能力。一个模型在训练前需要经历数据源接入、格式统一、文本解析、质量清洗、隐私过滤、去重、token 统计、质量评估、采样配比和报告反馈等流程。

LLM-DataForge 的定位是“轻量级可复现的 LLM 数据质量流水线”。它不声称处理真实万亿 Token，也不模拟完整工业平台，而是用单机 Python 实现一个小而完整的数据构建闭环，展示对 LLM 数据工程核心模块的理解。

## 2. 需求分析

岗位职责可以映射到本项目功能：

- 预训练数据集构建：通过 `schema.py` 和 `ingest.py` 将网页、文档、代码、指令数据统一为 JSONL schema
- 数据处理清洗流程：通过 `parse.py`、`clean.py` 完成 HTML 解析、instruction 拼接、文本标准化、语言检测和隐私过滤
- 万亿 Token 规模意识：通过 `tokenize_stats.py` 统计 token 数、长度分布和 source token 占比，并在报告中说明分片、Parquet、Spark/Ray 扩展方向
- 数据对模型性能影响：通过 `quality.py` 构造规则版质量评分，并在 `data_quality_report.md` 中展示质量分布和样本案例
- 采样加权去偏：通过 `sampler.py` 支持按质量阈值过滤和按 source ratio 配比采样
- 闭环反馈：通过 `report.py` 输出 Markdown、JSON、CSV 和可选图表，为下一轮规则调参和人工抽检提供依据

## 3. 技术架构

- `ingest.py`：生成样例 raw data，读取目录或单个 JSONL 文件，给缺失 id 的记录补充稳定 id
- `parse.py`：将不同来源的原始记录解析为统一 schema，包括 HTML 提取正文和 instruction/input/output 拼接
- `clean.py`：文本标准化、语言检测、PII/secret 检测和基础过滤规则
- `dedup.py`：使用 MD5 做精确去重，使用字符 n-gram Jaccard 做小规模近似去重演示
- `tokenize_stats.py`：提供离线 fallback tokenizer，并可选使用 Hugging Face tokenizer，失败时自动回退
- `quality.py`：提取质量特征并计算规则版 `quality_score`
- `sampler.py`：按质量阈值和 source 比例采样，导出训练 JSONL
- `report.py`：生成数据质量报告、metrics.json 和多个 CSV 统计文件
- `cli.py`：提供 `generate-sample-data`、`run`、`clean`、`report` 命令，并编排完整 pipeline

## 4. 核心算法说明

### 文本标准化

`normalize_text` 统一 CRLF/LF 换行，删除控制字符，合并连续空格和过多空行，但不会删除正常中文、英文、数字、常见标点和代码符号。清洗策略强调“质量和多样性的平衡”：过度清洗可能删除长尾知识、代码结构和真实语料表达。

### PII 检测

`detect_pii` 使用正则检测邮箱、手机号、身份证号、API key、token、password、secret 等风险模式。检测结果写入 `safety_flags`。当配置 `enable_pii_filter: true` 时，命中隐私风险的文本会被标记为 `filtered=true`，原因是 `pii_detected`。

### 精确去重

`exact_dedup` 先用 `normalize_for_hash` 将文本转为小写并合并空白，再计算 MD5 签名。签名相同的记录被视为精确重复，只保留第一条，同时在保留记录中写入 `dedup_signature`。

### 近似去重

`approximate_dedup_small` 将文本转换为字符 n-gram 集合，使用 Jaccard similarity 判断近似重复。该方法实现简单、可解释，但需要全量两两比较，只适合小规模演示。生产环境应使用 MinHash/SimHash/LSH 或向量索引来降低复杂度。

### Token 统计

`SimpleTokenizer` 是离线 fallback tokenizer：中文按字符近似计数，英文按单词计数，代码按单词和符号计数。`HFTokenizerWrapper` 在配置启用时尝试加载 Hugging Face tokenizer，如果模型下载失败或环境没有 transformers，会自动回退到 `SimpleTokenizer`，保证 pipeline 不因网络问题中断。

### 质量评分

`quality.py` 提取 token_count、文本长度、URL 数量、连续重复字符比例、PII 标志、语言类型、来源类型、代码符号比例、字母数字比例和 instruction 结构等特征。`compute_quality_score` 使用可解释规则扣分或加分：过短、URL 过多、重复字符过高、PII、unknown language 会扣分；instruction 数据包含 Instruction/Output 结构会加分；代码数据对符号比例有更宽容的判断。

### 采样策略

`sample_by_quality` 先过滤低质量样本，`sample_by_source_ratio` 再按配置中的 source ratio 选择样本，并优先保留质量分和 token 数更高的记录。这样可以模拟训练前的数据配比控制，避免某一来源过度占比。

## 5. 关键代码解读

1. `parse_html_to_text(html: str) -> str`

使用 BeautifulSoup 删除 `script/style/nav/footer/header` 等噪声标签，再提取正文并合并空白。它展示了网页数据进入 LLM 语料前的基础解析思路。

2. `clean_record(record: dict, config: dict) -> dict`

这是清洗模块的入口。它会标准化文本、检测语言、检测 PII、填充 `safety_flags`，并根据 `basic_quality_filter` 写入 `filtered` 和 `filter_reason`。

3. `exact_dedup(records: list[dict]) -> tuple[list[dict], dict]`

使用 MD5 签名做精确去重，返回去重后的记录和统计信息，包括输入数量、输出数量、删除重复数和重复 id。它是预训练数据构建中最常见的基础模块之一。

4. `add_token_counts(records: list[dict], config: dict) -> tuple[list[dict], dict]`

为每条记录增加 `token_count`，同时生成总 token、平均 token、最大/最小 token、按 source 的 token 总量和长度分布。token 统计用于评估训练成本、长度异常和配比。

5. `generate_report(...) -> dict[str, Path]`

统一输出 `data_quality_report.md`、`metrics.json`、`source_distribution.csv`、`filtered_reasons.csv` 和 `quality_distribution.csv`。报告把 pipeline 的运行结果变成可审计的质量反馈。

## 6. 实验与输出

运行：

```bash
python -m llm_dataforge.cli generate-sample-data --output data/raw
python -m llm_dataforge.cli run --config configs/pipeline.yaml
```

会生成：

- `data/interim/clean.jsonl`：解析与清洗后的记录，包含过滤标记
- `outputs/processed/processed.jsonl`：去重、token 统计、质量评分后的候选数据
- `outputs/processed/train.jsonl`：最终可用于训练或后续实验的 JSONL
- `outputs/reports/data_quality_report.md`：人类可读数据质量报告
- `outputs/reports/metrics.json`：完整结构化运行指标
- `outputs/reports/source_distribution.csv`：各 source 的记录数、token 数和 token 占比
- `outputs/reports/filtered_reasons.csv`：过滤原因分布
- `outputs/reports/quality_distribution.csv`：质量分桶统计

样例数据包含高质量文本、过短文本、URL 过多文本、重复文本、HTML 噪声、手机号/邮箱/API key、代码文本、instruction-output 数据和中英文混合文本。因此报告中能看到过滤、去重、token 统计和采样的完整闭环。

## 7. 项目局限性

- 没有真实处理万亿 Token，只是轻量级教学和面试项目
- 近似去重只是小规模演示，全量两两比较不能用于真实大规模语料
- 没有真正接入训练框架，`train.jsonl` 只是训练数据导出结果
- `quality_score` 是规则版，不是训练反馈版或学习型质量模型
- 没有复杂安全分类器，PII 检测依赖正则规则
- 没有分布式调度，也没有任务失败重试、资源隔离和作业监控
- 没有实现数据版权治理、来源授权检查和人工审核系统

## 8. 下一步优化

- 使用 Spark/Ray 做分布式清洗、去重和 token 统计
- 使用 MinHash + LSH 做大规模近似去重
- 使用 Parquet 做列式存储和分区管理
- 接入真实 tokenizer，并记录 tokenizer version
- 接入困惑度模型或质量分类器
- 增加人工抽检界面，支持样本标注和规则回放
- 增加数据版本 manifest、source manifest 和处理 lineage
- 接入训练评估反馈，把模型 loss、benchmark、人工偏好结果反馈到数据规则和采样权重

## 9. 简历写法

LLM-DataForge：面向大模型预训练/后训练数据构建的数据质量流水线。使用 Python 构建 JSONL 数据处理 pipeline，支持网页、文档、代码和指令数据统一格式化，实现文本清洗、隐私过滤、精确去重、近似去重、Tokenizer 统计、质量评分、采样和 Markdown 质量报告生成。项目模拟大模型数据工程核心流程，强调数据质量、去重、Token 级统计和反馈迭代。

## 10. 面试讲解稿

我做了一个叫 LLM-DataForge 的项目，定位是轻量级的大模型数据质量流水线。它不是 RAG 或聊天机器人，而是模拟预训练和后训练数据进入训练前的处理流程。

项目支持网页、文档、代码和指令四类数据。第一步会把不同来源的数据解析成统一 JSONL schema，比如网页会用 BeautifulSoup 去掉 script、style、nav、footer 等噪声，指令数据会统一拼成 Instruction/Input/Output 格式。然后进入清洗阶段，做文本标准化、语言检测、基础质量过滤和 PII 检测，比如邮箱、手机号、身份证号、API key、token、password 这类信息会被标记并过滤。

清洗后项目会做两层去重：精确去重用归一化文本的 MD5，近似去重用字符 n-gram 的 Jaccard similarity。这里我明确把近似去重写成小规模演示，因为生产环境应该换成 MinHash、SimHash 或 LSH。之后 pipeline 会统计 token 数，默认使用一个离线 fallback tokenizer，中文按字符、英文按词、代码按词和符号粗略计数；如果配置启用 Hugging Face tokenizer，加载失败也会自动回退，不会让 pipeline 因为网络问题中断。

最后，项目会根据 token_count、URL 数量、重复字符比例、PII 标志、语言类型、source 类型和 instruction 结构计算规则版 quality_score，再按质量阈值和 source 配比采样，导出 train.jsonl，并生成 Markdown、JSON 和 CSV 数据质量报告。这个项目的重点是展示我理解 LLM 数据层的关键问题：数据质量、隐私风险、重复率、token 分布、数据配比和质量反馈闭环。它不是生产级万亿 token 系统，但模块可以扩展到 Spark/Ray、Parquet、MinHash/LSH、真实 tokenizer、质量分类器和训练反馈。
