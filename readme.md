# Agent Demo：基于 Python 的智能体应用实践

一个基于 Python 和大语言模型 API 实现的 Agent 项目，用于学习和实践智能体开发中的工具调用、多轮对话、上下文管理、摘要记忆、运行轨迹记录及本地知识库读取等核心技术。

本项目采用渐进式开发方式，从基础工具调用出发，逐步完善 Agent 的上下文管理、记忆机制和工程化能力。

## 1. 项目功能

### 1.1 基础 Agent 能力

* **大语言模型接入：** 通过 API 调用大语言模型。
* **工具调用（Tool Calling）：** 支持模型根据用户需求选择并调用 Python 函数。
* **多轮工具执行：** 将工具执行结果返回模型，支持模型继续推理并生成最终回答。
* **交互式对话：** 支持连续对话及会话重置、退出等基本操作。

### 1.2 上下文与记忆管理

* **对话持久化：** 保存和加载历史对话。
* **上下文裁剪：** 根据轮数和 Token 预算筛选历史消息，控制模型输入长度。
* **摘要记忆：** 对较早的对话内容进行摘要，保留关键信息。
* **上下文统计：** 记录消息数量及 Token 使用情况，辅助分析上下文管理效果。

### 1.3 工具与知识库

* `calculate`：执行数学计算。
* `get_current_time`：获取当前时间。
* `list_knowledge_files`：列出本地知识库中允许读取的文件。
* `read_knowledge_file`：读取指定的 Markdown 或 TXT 文件。

本地知识库目前采用按文件名读取的方式，为后续实现文本切分、相关内容检索及 RAG 知识库问答打下基础。

### 1.4 运行记录与评测

* 使用 JSONL 文件记录 Agent 运行轨迹。
* 通过测试用例检查工具选择和最终回答。
* 记录输入 Token、模型调用次数及执行耗时等指标。
* 对比不同上下文管理策略的初步测试结果。

## 2. 技术栈

* Python
* 大语言模型 API（Moonshot / Kimi）
* OpenAI Python SDK
* JSON / JSONL
* tiktoken（用于 Token 数量估算）

## 3. 项目结构

```text
agent_demo/
├── agent_tool_demo.py          # Agent 主程序
├── eval_cases.json             # 评测用例
├── run_eval.py                 # 自动化评测脚本
├── .env.example                # 环境变量配置模板（需自行创建）
├── .gitignore                  # Git 忽略规则
├── README.md                   # 项目说明
├── data/
│   └── knowledge/
│       └── agent_project_notes.md
└── logs/                       # 运行日志，不提交到 Git
```

实际文件结构可能因开发进度有所不同。

## 4. 环境配置

### 4.1 安装 Python 依赖

建议使用 Python 虚拟环境。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install openai python-dotenv tiktoken
```

### 4.2 配置 API

在项目根目录创建 `.env` 文件，填写自己的 API 配置：

```dotenv
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=your_model_name
```

将 `LLM_API_KEY` 替换为自己的 API Key，并将 `LLM_MODEL` 设置为账户当前可用的模型名称。

**安全提示：** `.env` 仅用于本地配置，不要上传真实 API Key 到 GitHub。

## 5. 运行项目

在项目根目录执行：

```powershell
python .\agent_tool_demo.py
```

启动后，可以输入问题与 Agent 交互。例如：

```text
帮我计算 12345 × 6789。
```

```text
查看一下知识库中有哪些文件。
```

```text
读取 agent_project_notes.md，告诉我项目目前已经实现了哪些功能。
```

实际可调用的工具由当前代码中的工具定义决定。

## 6. 运行评测

如果项目中已存在 `eval_cases.json` 和 `run_eval.py`，可以执行：

```powershell
python .\run_eval.py
```

评测脚本用于检查 Agent 在指定测试用例上的表现。具体输出文件名以脚本中的配置为准。

当前评测用例数量有限，测试结果主要用于初步验证功能和比较策略，不代表系统在所有任务上的表现。

## 7. 后续开发计划

* [ ] 完善工具参数校验及异常处理。
* [ ] 扩展自动化评测用例，增加失败案例分析。
* [ ] 优化长对话上下文管理和摘要记忆。
* [ ] 实现文本切分与相关内容检索。
* [ ] 构建完整的 RAG 知识库问答流程。
* [ ] 增加 API 接口和简单的 Web 交互界面。
* [ ] 完善日志追踪、执行监控和系统测试。

## 8. 项目目标

通过持续迭代，逐步掌握 Agent Runtime、Tool Calling、上下文管理、记忆机制、RAG 和自动化评测等技术，并提升将大语言模型能力转化为实际应用的工程实践能力。
