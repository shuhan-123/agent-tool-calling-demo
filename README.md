# 论文问答 Agent（Tool Calling + RAG Demo）

## 项目说明
基于 LangChain 构建的 Tool-Calling Agent，能自动判断问题类型并调用对应工具：
- **lookup_metric**：结构化数据查询工具，从评测结果表中查询指定维度的幻觉率
- **retrieve_doc**：RAG 检索工具，基于 FAISS 向量索引在论文原文中检索相关段落

Agent 自主决策调用哪个工具，无需人工指定，并对每次工具调用记录耗时与成功状态（写入 `agent_log.txt`），实现基础的可观测性设计。

## 技术栈
`LangChain` `FAISS` `HuggingFace Embeddings` `Tool Calling`

## 使用方法
1. `pip install -r requirements.txt`
2. `python build_index.py 你的论文.pdf`（只需运行一次，建立向量索引）
3. 设置环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`（任选一个兼容 OpenAI 接口的大模型平台）
4. `python agent.py`，输入问题即可

## 示例问题
- "语言先验维度的幻觉率是多少？" → 自动调用 lookup_metric
- "论文里是怎么设计对抗性测试集的？" → 自动调用 retrieve_doc
