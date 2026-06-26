# LLM-DataForge

LLM-DataForge 是一个面向“大模型数据算法工程师 / LLM 数据工程 / 预训练数据构建”岗位的轻量级实战项目。它不是普通 RAG Demo，也不是聊天机器人，而是一个可复现的大模型预训练/后训练数据质量流水线，用 Python 模拟企业级数据构建中的核心模块。

项目目标是展示数据层能力：多源数据读取、统一 JSONL schema、文本解析、清洗、隐私过滤、精确去重、近似去重、Tokenizer 统计、质量评分、数据配比采样、训练集导出和质量报告生成。

## 为什么做这个项目

很多 LLM 应用项目只展示“调用模型回答问题”，但大模型数据算法工程师更关心数据如何进入训练或后训练流程：数据来源是否可控、重复率是否过高、隐私信息是否被过滤、token 分布是否合理、不同来源如何配比、质量问题如何反馈到下一轮数据构建。

LLM-DataForge 用轻量级方式复现这些关键环节，适合作为面试、课程项目和 GitHub 展示项目。

## 适合岗位

- 大模型数据算法工程师实习生
- AI 数据工程 / LLM Data Engineer
- 预训练数据处理 / 后训练数据处理
- 数据清洗、数据质量、数据闭环方向
- 机器学习工程师转 LLM 数据工程方向

## 功能特性

- 支持 `web`、`document`、`code`、`instruction` 四类数据
- 统一 JSONL 数据 schema
- HTML 解析和 instruction/input/output 拼接
- 文本标准化、语言检测、基础质量过滤
- 邮箱、手机号、身份证号、API key、secret/password 类隐私风险检测
- MD5 精确去重
- 字符 n-gram + Jaccard 小规模近似去重演示
- 离线 fallback tokenizer，避免无网络时失败
- 可选 Hugging Face tokenizer，失败时自动回退
- 规则版 `quality_score`
- 按质量分和 source 配比采样
- 导出 `train.jsonl`
- 生成 Markdown、JSON、CSV 和可选 PNG 质量报告
- 提供 CLI 和 pytest 单元测试

## Pipeline 流程图

```text
raw data
  -> ingest / parse
  -> clean
  -> exact dedup
  -> approximate dedup
  -> tokenize stats
  -> quality score
  -> quality + source-ratio sample
  -> export train.jsonl
  -> generate report
```

## 目录结构

```text
LLM-DataForge/
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── pipeline.yaml
├── data/
│   ├── raw/
│   │   ├── web.jsonl
│   │   ├── document.jsonl
│   │   ├── code.jsonl
│   │   └── instruction.jsonl
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── processed/
│   └── reports/
├── src/
│   └── llm_dataforge/
│       ├── __init__.py
│       ├── schema.py
│       ├── ingest.py
│       ├── parse.py
│       ├── clean.py
│       ├── dedup.py
│       ├── tokenize_stats.py
│       ├── quality.py
│       ├── sampler.py
│       ├── report.py
│       ├── cli.py
│       └── utils.py
└── tests/
    ├── test_clean.py
    ├── test_dedup.py
    ├── test_quality.py
    └── test_pipeline.py
```

## 安装方式

需要 Python 3.10+。

```bash
cd LLM-DataForge
pip install -r requirements.txt
```

`requirements.txt` 中包含 `-e .`，安装后可以直接运行：

```bash
python -m llm_dataforge.cli --help
```

## 快速运行

```bash
python -m llm_dataforge.cli generate-sample-data --output data/raw
python -m llm_dataforge.cli run --config configs/pipeline.yaml
```

只运行清洗：

```bash
python -m llm_dataforge.cli clean --input data/raw --output data/interim/clean.jsonl --config configs/pipeline.yaml
```

从已有训练集生成报告：

```bash
python -m llm_dataforge.cli report --input outputs/processed/train.jsonl --output outputs/reports
```

## 输出文件说明

完整 pipeline 运行后生成：

- `data/interim/clean.jsonl`：解析和清洗后的全量记录，包含 filtered 和 filter_reason
- `outputs/processed/processed.jsonl`：清洗保留、去重、token 统计、质量评分后的记录
- `outputs/processed/train.jsonl`：最终训练 JSONL
- `outputs/reports/data_quality_report.md`：Markdown 数据质量报告
- `outputs/reports/metrics.json`：结构化指标
- `outputs/reports/source_distribution.csv`：source 记录数和 token 占比
- `outputs/reports/filtered_reasons.csv`：过滤原因分布
- `outputs/reports/quality_distribution.csv`：质量分桶分布
- `outputs/reports/*.png`：如果 matplotlib 可用，会生成简单图表

## 数据 Schema

每条数据统一为 JSONL，一行一个 JSON 对象：

```json
{
  "id": "doc_000001",
  "text": "清洗后的正文内容",
  "source": "web|document|code|instruction",
  "language": "zh|en|mixed|code|unknown",
  "domain": "technology|general|code|education|unknown",
  "token_count": 0,
  "quality_score": 0.0,
  "dedup_signature": "",
  "safety_flags": [],
  "filtered": false,
  "filter_reason": "",
  "metadata": {
    "url": "",
    "title": "",
    "parser": "",
    "version": "clean_v1"
  }
}
```

## 模块说明

- `schema.py`：定义 `DataRecord`、schema 校验、schema 标准化和 id 生成
- `ingest.py`：读取 raw JSONL，生成样例数据
- `parse.py`：HTML 解析、instruction 记录拼接、多源记录标准化
- `clean.py`：文本标准化、语言检测、PII 检测、基础质量过滤
- `dedup.py`：MD5 精确去重、字符 n-gram 近似去重、SimHash 演示函数
- `tokenize_stats.py`：fallback tokenizer、HF tokenizer wrapper、token 统计
- `quality.py`：特征提取、规则版质量评分、质量分桶
- `sampler.py`：按质量过滤、按来源配比采样、导出 train.jsonl
- `report.py`：生成 Markdown、JSON、CSV 和可选图表
- `cli.py`：命令行入口和 pipeline 编排


## 局限性

- 这是轻量级教学/面试项目，不声称处理真实万亿 Token
- 当前 pipeline 是单机 Python，没有分布式调度
- 近似去重使用全量两两 Jaccard 比较，只适合小数据集演示
- 生产环境应使用 MinHash/SimHash/LSH，不应全量两两比较
- `quality_score` 是规则版，不是基于训练反馈或人工标注训练的质量模型
- PII 过滤是正则规则，不等同于完整安全分类器
- 没有直接接入真实训练框架

## 下一步计划

- 使用 Spark/Ray 做分布式清洗和统计
- 使用 Parquet 做列式存储，支持分片和增量处理
- 使用 MinHash + LSH 做大规模近似去重
- 接入真实模型 tokenizer，并保存 tokenizer version
- 接入困惑度模型或质量分类器
- 增加人工抽检界面和样本审核记录
- 增加数据版本 manifest、source manifest 和处理 lineage
- 接入训练评估反馈，形成数据闭环
