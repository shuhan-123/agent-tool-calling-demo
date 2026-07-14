"""
主程序：注册两个工具给 Agent，Agent 自己判断该查表还是查资料。
运行前需要设置环境变量（任选一个兼容 OpenAI 接口的大模型平台即可，
比如智谱、DeepSeek、Moonshot 等，它们都提供 OpenAI 兼容接口）：

Windows PowerShell 设置示例：
$env:OPENAI_API_KEY="你的key"
$env:OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4"  # 换成你用的平台地址

运行：
python agent.py
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import lookup_metric, retrieve_doc

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "glm-4-flash"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

tools = [lookup_metric, retrieve_doc]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个论文问答助手，能查询幻觉率数据表，也能检索论文原文。请根据问题判断该用哪个工具。重要规则：1. lookup_metric 只包含五个评测维度的跨模型平均幻觉率，不包含 Qwen2-VL 2B/7B 等单模型指标；lookup_metric 的输入只能是：目标存在性、计数与属性、OCR文本识别、文化常识、语言先验，严禁用“总体表现”等其他词作为输入；2. 如果用户问“最高/最低/哪一种”等跨维度比较，必须分别调用 lookup_metric 查询五个维度：目标存在性、计数与属性、OCR文本识别、文化常识、语言先验，再比较后回答；3. 如果用户问模型之间的总体表现或论文解释原因，应调用 retrieve_doc 检索论文原文；4. 回答具体模型总体表现时，必须优先依据总体幻觉率，幻觉率越低代表表现越好；如果论文说从 2B 到 7B 总体幻觉率上升，结论就是 2B 总体表现更好，同时可补充 7B 在某些局部类型上更好；5. 不要编造工具没有返回的数据。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print("Agent 已启动，输入问题（输入 exit 退出）")
    while True:
        q = input("\n你的问题：")
        if q.strip().lower() == "exit":
            break
        result = executor.invoke({"input": q})
        print("\n回答：", result["output"])
