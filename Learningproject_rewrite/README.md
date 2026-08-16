# 知识库问答系统（重构版）

基于 **Streamlit + LangChain + LangGraph + Chroma + DeepSeek V4 Flash** 的本地知识库问答系统：
上传文档自动入库，支持多轮对话、来源引用、深度思考与 Agent 检索。

本目录是原项目 `Learningproject` 的可读性重构版（原项目保留不动），
功能保留 1–6 项（文档入库、两种问答流程、混合检索、实时工具、聊天体验、
Langfuse 可观测性），砍掉了离线评估（Ragas）。

## 快速开始

```bash
cd Learningproject_rewrite
# 直接复用原项目的虚拟环境（依赖相同），或新建：
# python -m venv .venv
# .venv\Scripts\activate
# pip install -r requirements.txt

# 配置 API Key（可从原项目复制 .env）
copy .env.example .env

# 启动
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

打开 `http://localhost:8501` 后：
1. 左侧上传文档（可用 `sample_data/员工手册.md` 体验）；
2. 点击「开始入库」；
3. 在聊天框提问，例如「员工请假需要提前几天申请？」

## 功能一览

| 功能 | 说明 |
| --- | --- |
| 文档入库 | txt / md / pdf / docx 解析、中英文标点智能分块、同名文件覆盖更新、删除/清空 |
| 两种问答流程 | 标准 RAG（检索→生成）与 Agent 检索（模型自主决定是否检索，ReAct 循环） |
| 混合检索 | 向量（Chroma）+ BM25（jieba 分词）+ RRF 融合，可选 Cross-Encoder 重排序 |
| 实时工具 | 当前时间 / IP 定位 / 实时天气（Open-Meteo，内置 356 个中国城市坐标） |
| 聊天体验 | 多轮对话、流式输出、深度思考、引用来源与工具调用展示 |
| 可观测性 | 可选 Langfuse 追踪，记录检索、工具、模型完整调用链 |

## 项目结构

```
Learningproject_rewrite/
├── streamlit_app.py          # 应用入口：页面组装与交互流程
├── requirements.txt          # 依赖
├── .env.example              # 配置模板（复制为 .env 使用）
├── sample_data/员工手册.md    # 示例文档
└── app/
    ├── __init__.py           # 包初始化：加载 .env + 设置离线环境变量
    ├── config.py             # 全局配置常量
    ├── models.py             # 领域模型（dataclass / TypedDict）
    ├── llm.py                # DeepSeek 模型封装（reasoning_content 兼容）
    ├── store.py              # 向量库：嵌入模型 + Chroma
    ├── ingestion.py          # 文档解析、分块与入库
    ├── retrieval.py          # 混合检索 + 重排序
    ├── tools.py              # Agent 实时工具（时间/地点/天气）
    ├── graphs.py             # LangGraph 流程 + 流式输出解析
    ├── tracing.py            # Langfuse 可观测性
    └── ui.py                 # Streamlit 界面组件（侧边栏/聊天渲染）
```

## 与原项目的主要差异

- **数据模型类型化**：新增 `app/models.py`，用 `SourceReference`、`ToolCallRecord`、
  `AssistantReply`、`CollectionStats`、`IngestionResult` 等 dataclass 取代散落的 dict；
- **模块职责更清晰**：`ingest.py` → `ingestion.py`（只做文档处理），
  `graph.py` → `graphs.py`（只做流程编排与流式解析），界面代码从入口拆到 `ui.py`；
- **命名更直观**：`get_vectorstore` → `create_vector_store`、`collection_stats` →
  `get_collection_stats`、`delete_source` → `delete_source_documents`、
  `invalidate_all` → `invalidate_retrieval_cache` 等；
- **去掉重复逻辑**：原先 `store.py` 与 `retrieval.py` 重复设置
  `HF_HUB_OFFLINE`，现统一收敛到 `app/__init__.py`；入库/删除/清空后的
  「失效缓存 + 刷新页面」合并为 `_refresh_knowledge_base` 一个入口；
- **全量类型标注与中文注释**：所有公开函数补充类型签名与职责说明。

## 常见问题

**首次启动较慢 / 卡在「加载本地嵌入模型」**

应用默认以离线模式加载嵌入模型（约 100 MB）。若本地没有该模型，
先联网下载一次：

```bash
huggingface-cli download BAAI/bge-small-zh-v1.5
```

或在 `.env` 中设置 `HF_HUB_OFFLINE=0` 后重启应用在线下载。

**多轮对话报 400 `reasoning_content` 错误**

这是 `langchain-deepseek` 1.0.x 的已知问题（GitHub #37713），
本项目通过 `app/llm.py` 的兼容子类在请求中回传 `reasoning_content`，
开启深度思考模式也可正常多轮对话。

**切换模型 / 嵌入模型**

模型下拉框支持 `deepseek-v4-flash`（默认）与 `deepseek-v4-pro`。
更换嵌入模型会改变向量维度，需删除 `chroma_db/` 目录后重新入库。
