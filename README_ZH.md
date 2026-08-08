# LangGraph ReAct Agent 模板

[![CI](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent)

这个模板展示了使用 [LangGraph](https://github.com/langchain-ai/langgraph) 实现的 [ReAct 代理](https://arxiv.org/abs/2210.03629)，专为 [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) 设计。ReAct 代理是简单、原型化的代理，可以灵活地扩展到许多工具。

![LangGraph Studio UI 中的图形视图](./static/studio_ui.png)

核心逻辑在 `src/react_agent/graph.py` 中定义，演示了一个灵活的 ReAct 代理，它可以迭代推理用户查询并执行操作，展示了这种方法在复杂问题解决任务中的强大功能。

## 它的作用

ReAct 代理：

1. 将用户**查询**作为输入

2. 推理查询并决定执行操作

3. 使用可用工具执行所选操作

4. 观察操作结果

5. 重复步骤 2-4，直到它能够提供最终答案

默认情况下，它设置了一组基本工具，但可以使用自定义工具轻松扩展以适应各种用例。

## 入门

假设您已经[安装了 LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download)，请进行设置：

1. 创建一个 `.env` 文件。

```bash
cp .env.example .env
```

2. 在您的 `.env` 文件中定义所需的 API 密钥。

主要使用的 [搜索工具](./src/react_agent/tools.py) [^1] 是 [Tavily](https://tavily.com/)。创建 API 密钥 [此处](https://app.tavily.com/sign-in)。

<!--
由 `langgraph template lock` 自动生成的设置说明。请勿手动编辑。
-->

### 设置模型

`model` 的默认值如下所示：

```yaml
model: anthropic/claude-3-5-sonnet-20240620
```

按照以下说明进行设置，或选择其他选项之一。

#### Anthropic

要使用 Anthropic 的聊天模型：

1. 如果尚未注册，请注册 [Anthropic API 密钥](https://console.anthropic.com/)。
2. 获得 API 密钥后，将其添加到 `.env` 文件中：

```
ANTHROPIC_API_KEY=your-api-key
```
#### OpenAI

要使用 OpenAI 的聊天模型：

1. 注册 [OpenAI API 密钥](https://platform.openai.com/signup)。
2. 获得 API 密钥后，将其添加到 `.env` 文件中：
```
OPENAI_API_KEY=your-api-key
```

<!--
结束设置说明
-->

3. 在代码中自定义您想要的任何内容。
4. 打开文件夹 LangGraph Studio！

## 如何自定义

1. **添加新工具**：通过在 [tools.py](./src/react_agent/tools.py) 中添加新工具来扩展代理的功能。这些可以是执行特定任务的任何 Python 函数。
2. **选择其他模型**：我们默认使用 Anthropic 的 Claude 3 Sonnet。您可以通过配置使用 `provider/model-name` 选择兼容的聊天模型。示例：`openai/gpt-4-turbo-preview`。
3. **自定义提示**：我们在 [prompts.py](./src/react_agent/prompts.py) 中提供了默认系统提示。您可以通过工作室中的配置轻松更新它。

您还可以通过以下方式快速扩展此模板：

- 在 [graph.py](./src/react_agent/graph.py) 中修改代理的推理过程。
- 调整 ReAct 循环或向代理的决策过程添加其他步骤。

## 开发

在图表上迭代时，您可以编辑过去的状态并从过去的状态重新运行您的应用程序