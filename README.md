# ZenGraph-Agent: 基于 LangGraph 的自适应佛学智能体

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)
![RAG](https://img.shields.io/badge/Architecture-Advanced_RAG-green.svg)

**ZenGraph-Agent** 是一个专注于佛学古籍深度理解与交互的生产级 AI Agent 系统。它不仅是一个简单的问答工具，更是一个具备“自我反思、意图识别、指代消解”能力的进阶版 RAG 系统。

本项目旨在解决传统 RAG 在垂直古籍领域中面临的：**指代模糊、术语误解、多轮对话逻辑断裂**等核心痛点。

---

## 🚀 核心技术亮点

### 1. 语义路由分诊层 (Semantic Routing)
利用 **Router Node** 对用户意图进行精准分类：
- **Contextualize (重构模式)**：针对“怎么做？”等简短追问，利用 LLM 进行指代消解（Coreference Resolution），补全主语，确保检索质量。
- **HyDE (扩展模式)**：针对抽象概念，通过生成假设性文档（Hypothetical Document Embeddings）增强语义检索深度。
- **Direct Chat**：针对闲聊意图，实现快速短路响应，节省 Token 成本。

### 2. 两阶段检索流水线 (Two-Stage Retrieval)
- **海选阶段**：基于向量数据库实现语义初筛。
- **精排阶段**：集成 **BCE-Reranking** 深度排序模型，利用 Cross-Encoder 架构对海选结果进行二次评估，显著提升 Context Precision（上下文精确度）。

### 3. 基于 LangGraph 的自适应循环架构
不同于传统的顺序 Chain，本项目基于 **LangGraph** 构建了具备循环能力的图结构：
- **Self-Correction**：内置 Grader 节点，自动评估检索到的经文是否能回答用户问题，若质量不合格则自动触发 Query 重写。
- **Memory Persistence**：通过 Checkpointer 实现多轮对话状态持久化，使 Agent 具备长短期记忆。

### 4. 专业级 ETL 数据治理
- **古籍归一化 (Normalization)**：针对原始经文繁简混杂的情况，构建了基于 **OpenCC** 的双向语义对齐管线。
- **自动化预处理**：支持递归读取多级目录下的 txt 文档，自动提取文件夹分类作为元数据（Metadata）。

---

## 🛠️ 技术栈

- **核心框架**: LangGraph, LangChain / LlamaIndex
- **大语言模型**: DeepSeek-V3
- **Embedding/Rerank**: BGE-M3 / BCE-Reranker-base_v1
- **数据治理**: OpenCC, Python-docx/PyPDF2
- **向量数据库**: ChromaDB / Qdrant

---

## 📂 目录结构

```text
├── data/               # 存放经文数据（按部类分文件夹）
├── src/
│   ├── agents.py       # LLM 角色定义与调用逻辑
│   ├── nodes.py        # LangGraph 节点逻辑（Router, Contextualize, Rerank等）
│   ├── state.py        # 状态定义 (AgentState)
│   ├── utils.py        # 工具函数（繁简转换、模型初始化）
│   └── retriever.py    # 检索器与索引构建
├── main.py             # 项目入口与图构建
└── requirements.txt    # 依赖项
