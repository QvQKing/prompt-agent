# -*- coding: utf-8 -*-  # 指定源码编码，确保中文注释不乱码
"""
Multi-provider LLM Answer Tool (Chat Completions via OpenAI-compatible APIs)
支持多个 LLM 提供商（OpenAI/DeepSeek/APIYI 等），并可通过中转平台 base_url 访问。

放置路径: agent_r1/tool/tools/multi_llm_answer_tool.py
依赖安装: pip install --upgrade openai
环境变量（示例）:
  # 基本选择
  export LLM_PROVIDER=apiyi                 # 可选: openai / deepseek / apiyi (默认 openai)
  export LLM_MODEL=gpt-4.1-mini            # 模型名（示例为 apiyi 转发 OpenAI 模型）
  export LLM_API_KEY=sk-xxx                # 通用 API Key（若未设置则尝试各家专用变量）
  # 各家专用 Key（任选其一或都配上）
  export OPENAI_API_KEY=sk-openai-xxx
  export DEEPSEEK_API_KEY=sk-deepseek-xxx
  export APIYI_API_KEY=sk-apiyi-xxx
  # Base URL（若不显式设置，则依据 LLM_PROVIDER 给默认）
  export LLM_BASE_URL=https://api.apiyi.com/v1
  # 其他参数
  export LLM_TEMPERATURE=0.2
  export LLM_MAX_TOKENS=512
  export LLM_SYSTEM_PROMPT="You are a helpful assistant."
  # 可选：额外请求头（JSON 字符串），用于某些网关需要自定义 header 的情况
  export LLM_EXTRA_HEADERS='{"X-Custom-Header":"hello"}'
"""

from typing import Dict, List, Any  # 类型注解：Dict/List/Any
import os                          # 读取环境变量
import json                        # 序列化/反序列化 JSON
import ast                         # 安全地解析环境变量中的 JSON/字典字面量
import time                        # 简单重试时的 sleep（可选）

from agent_r1.tool.base import BaseTool  # 继承你项目的 BaseTool 抽象基类

from openai import OpenAI  # OpenAI 官方 SDK 客户端（支持自定义 base_url，从而接入兼容网关）


class MultiLLMTool(BaseTool):  # 定义多提供商 LLM 答题工具，继承 BaseTool
    """
    通过 OpenAI 兼容的 Chat Completions 接口，向不同提供商（或中转平台）发送对话请求并返回答案。
    """

    # ---- 工具的元信息（OpenAI function-calling 兼容）----
    name = "multi_llm_tool"  # 工具注册名，上层用这个名字调用
    description = "Answer questions by calling an OpenAI-compatible Chat Completions API (supports multiple providers and relay gateways)."  # 工具说明
    parameters = {  # JSON Schema：只需一个 query 字符串
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user question to answer"
            }
        },
        "required": ["query"]
    }

    def __init__(self):  # 构造函数：读取配置并初始化 OpenAI 客户端
        super().__init__()  # 调用父类构造，做 schema 合法性校验

        # ---------- 固定使用 apiyi 的 base_url ----------
        self.base_url = "https://api.apiyi.com/v1"  # apiyi 的 OpenAI 兼容 API 根路径

        # ---------- 读取 API Key ----------
        # 优先使用 APIYI_API_KEY；没有则尝试通用 LLM_API_KEY（便于和你们其它环境统一）
        self.api_key = os.environ.get("APIYI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
        # 这里不立即抛错，放到执行阶段更友好地返回错误信息

        # ---------- 推理相关超参数 ----------
        self.model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")       # 默认使用 gpt-4.1-mini
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))  # 默认温度 0.2（更稳定）
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "512"))      # 默认最大长度 512
        self.system_prompt = os.environ.get("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")  # 系统提示

        # ---------- 额外请求头（可选） ----------
        extra_headers_raw = os.environ.get("LLM_EXTRA_HEADERS", "").strip()  # 读取用户自定义 header
        self.extra_headers = {}  # 默认空字典
        if extra_headers_raw:  # 若设置了
            try:
                # 兼容 JSON 或 Python 字面量（{"k":"v"} / {'k':'v'}）
                self.extra_headers = (
                    ast.literal_eval(extra_headers_raw) if extra_headers_raw[0] in "{[" else json.loads(extra_headers_raw)
                )
                if not isinstance(self.extra_headers, dict):  # 必须是字典类型
                    self.extra_headers = {}
            except Exception:
                self.extra_headers = {}  # 解析失败则忽略

        # ---------- 初始化 OpenAI 客户端 ----------
        # 通过 base_url 指向 apiyi 的兼容接口；默认请求头可注入自定义 header
        self.client = OpenAI(
            api_key=self.api_key or None,                   # 若为空则传 None，执行时统一校验
            base_url=self.base_url,                         # 固定为 apiyi
            default_headers=self.extra_headers or None      # 需要附加头时传入
        )

        # ---------- 简单重试配置 ----------
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))     # 最大重试次数（失败后再试 N 次）
        self.retry_backoff = float(os.environ.get("LLM_RETRY_BACKOFF", "0.8"))  # 每次重试前等待秒数

    # -------------------- 单次调用 --------------------
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        通过 apiyi 的 Chat Completions 接口生成答案，并以统一格式返回：
        {"content": json.dumps({"results": [answer]}), "success": True}
        """
        try:
            # 1) 读取并校验参数
            query = (args.get("query") or "").strip()  # 获取 query 参数并去除首尾空白
            if not query:  # 若 query 为空
                return {"content": "Empty query.", "success": False}  # 返回失败信息
            if not self.api_key:  # 未配置 API Key
                return {"content": "APIYI_API_KEY (or LLM_API_KEY) not set.", "success": False}  # 返回失败信息

            # 2) 组织 Chat Completions 的消息体
            messages = []  # 初始化消息列表
            if self.system_prompt:  # 若设置了系统提示
                messages.append({"role": "system", "content": self.system_prompt})  # 添加 system 消息
            messages.append({"role": "user", "content": query})  # 添加用户消息（真正的问题）

            # 3) 调用 apiyi 的 Chat Completions 接口（带简单重试）
            last_err = None  # 记录最后一次异常
            for attempt in range(self.max_retries + 1):  # 尝试（最大重试次数 + 1）次
                try:
                    # 使用 SDK 的 chat.completions.create 与 apiyi 示例保持一致
                    resp = self.client.chat.completions.create(
                        model=self.model,             # 模型名（默认 gpt-4.1-mini，可通过环境变量覆盖）
                        messages=messages,            # 对话消息列表
                        temperature=self.temperature, # 采样温度
                        max_tokens=self.max_tokens,   # 最大输出 token 数
                    )
                    # 4) 解析第一条候选的文本内容
                    answer = (resp.choices[0].message.content if resp and resp.choices else "") or ""
                    # 5) 保持外层返回结构与旧工具一致（便于上层兼容）
                    return {"content": json.dumps({"results": [answer]}), "success": True}
                except Exception as e:
                    # 调用失败，若还可重试则等待后继续
                    last_err = e
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff)
                        continue
                    # 已到最后一次仍失败，返回错误信息
                    return {"content": f"Chat Completions call failed: {last_err}", "success": False}

        except Exception as e:
            # 兜底异常处理，确保工具本身不崩
            return {"content": f"Unexpected error: {e}", "success": False}

    # -------------------- 批量调用（串行最小实现） --------------------
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """
        简单串行地逐条调用 execute；如需更高吞吐，可自行并发/队列化。
        """
        results: List[Dict[str, Any]] = []  # 结果列表
        for args in args_list:              # 遍历每条请求
            results.append(self.execute(args, **kwargs))  # 调用单条接口
        return results                       # 返回所有结果