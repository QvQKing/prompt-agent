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


class MultiLLMAnswerTool(BaseTool):  # 定义多提供商 LLM 答题工具，继承 BaseTool
    """
    通过 OpenAI 兼容的 Chat Completions 接口，向不同提供商（或中转平台）发送对话请求并返回答案。
    """

    # ---- 工具的元信息（OpenAI function-calling 兼容）----
    name = "multi_llm_answer"  # 工具注册名，上层用这个名字调用
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

    def __init__(self):  # 构造函数：读取配置、初始化客户端
        super().__init__()  # 父类构造，会做 schema 合法性校验

        # ---- 选择提供商（决定默认 base_url 与 API Key 优先级）----
        self.provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()  # 从环境变量读取提供商，默认 openai

        # ---- 模型名、温度、最大 tokens ----
        self.model = os.environ.get("LLM_MODEL", "").strip() or self._default_model(self.provider)  # 若未设置则按 provider 给默认
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))  # 生成温度，默认 0.2（稳定/确定些）
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "512"))      # 最大生成长度，默认 512

        # ---- System Prompt（可选）----
        self.system_prompt = os.environ.get("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")  # 系统提示

        # ---- base_url（如果未显式设置，则根据 provider 选择默认）----
        explicit_base = os.environ.get("LLM_BASE_URL", "").strip()  # 显式指定的 base_url
        self.base_url = explicit_base or self._default_base_url(self.provider)  # 使用显式或默认 base_url

        # ---- API Key（优先用通用 LLM_API_KEY，否则按 provider 回退到专用变量）----
        self.api_key = (
            os.environ.get("LLM_API_KEY")
            or (os.environ.get("OPENAI_API_KEY") if self.provider == "openai" else None)
            or (os.environ.get("DEEPSEEK_API_KEY") if self.provider == "deepseek" else None)
            or (os.environ.get("APIYI_API_KEY") if self.provider == "apiyi" else None)
            or ""
        )

        # ---- 额外 Header（供某些网关需要自定义头的场景；JSON/字典字面量）----
        extra_headers_raw = os.environ.get("LLM_EXTRA_HEADERS", "").strip()  # 读取字符串
        self.extra_headers = {}  # 默认空字典
        if extra_headers_raw:  # 如果设置了
            try:
                # 兼容 JSON 或 Python 字面量（如 {"X-Token":"xxx"} 或 {'X-Token': 'xxx'}）
                self.extra_headers = ast.literal_eval(extra_headers_raw) if extra_headers_raw[0] in "{[" else json.loads(extra_headers_raw)
                if not isinstance(self.extra_headers, dict):  # 必须是字典
                    self.extra_headers = {}
            except Exception:
                self.extra_headers = {}

        # ---- 初始化 OpenAI 客户端（可设置 base_url、自定义请求头）----
        # OpenAI SDK 允许传入 base_url（用于兼容网关）和默认 headers（如需要）
        self.client = OpenAI(
            api_key=self.api_key or None,  # 若为空则传 None，后续执行时会校验
            base_url=self.base_url or None,  # 指定兼容 API 的 URL 前缀，如 https://api.apiyi.com/v1
            default_headers=self.extra_headers if self.extra_headers else None  # 需要自定义 Header 时传入
        )

        # ---- 简单的重试配置（可按需调节或移除）----
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))  # 失败时的最大重试次数
        self.retry_backoff = float(os.environ.get("LLM_RETRY_BACKOFF", "0.8"))  # 重试前的等待秒数

    # -------------------- 工具主逻辑：单次调用 --------------------
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        调用 OpenAI 兼容的 Chat Completions 接口并返回答案。
        统一返回结构：{"content": json.dumps({"results": [answer]}), "success": True}
        """
        try:
            # ---- 1) 读取并校验 query ----
            query = (args.get("query") or "").strip()  # 从参数里取出 query
            if not query:  # 如果 query 为空
                return {"content": "Empty query.", "success": False}  # 返回失败信息

            # ---- 2) 校验基础配置是否齐全 ----
            if not self.api_key:  # 缺少 API Key
                return {"content": "API key not set (LLM_API_KEY / provider-specific key).", "success": False}
            if not self.model:  # 缺少模型名
                return {"content": "Model name not set (LLM_MODEL).", "success": False}

            # ---- 3) 构造 messages（Chat Completions 风格）----
            messages = []  # 初始化消息列表
            if self.system_prompt:  # 若设置了系统提示
                messages.append({"role": "system", "content": self.system_prompt})  # 先放 system 指令
            messages.append({"role": "user", "content": query})  # 用户问题作为 user 消息

            # ---- 4) 调用 Chat Completions（带简单重试）----
            last_err = None  # 记录最后一次错误
            for attempt in range(self.max_retries + 1):  # 尝试 max_retries+1 次
                try:
                    # 使用 SDK 的 chat.completions.create（与中转平台示例一致）
                    resp = self.client.chat.completions.create(
                        model=self.model,                  # 模型名（例如 gpt-4.1-mini 或 deepseek-chat 等）
                        messages=messages,                 # 对话消息列表
                        temperature=self.temperature,      # 采样温度
                        max_tokens=self.max_tokens,        # 最大生成 tokens
                    )
                    # 解析返回的第一条 choice 的 message content
                    answer = (resp.choices[0].message.content if resp and resp.choices else "") or ""
                    # 统一包装并返回
                    return {"content": json.dumps({"results": [answer]}), "success": True}
                except Exception as e:
                    last_err = e  # 记录错误
                    if attempt < self.max_retries:  # 若还可重试
                        time.sleep(self.retry_backoff)  # 等待后重试
                        continue  # 进入下一次重试
                    else:
                        # 最后一次仍失败，返回错误信息
                        return {"content": f"Chat Completions call failed: {last_err}", "success": False}

        except Exception as e:  # 兜底异常（不影响外层框架）
            return {"content": f"Unexpected error: {e}", "success": False}

    # -------------------- 批量调用（简单串行） --------------------
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """
        逐条调用 execute 的最小实现；如需更快可自行并发/队列化。
        """
        results: List[Dict[str, Any]] = []  # 结果列表
        for args in args_list:              # 遍历每条请求参数
            results.append(self.execute(args, **kwargs))  # 直接复用单次 execute
        return results                       # 返回批量结果

    # -------------------- 内部工具：provider → 默认 base_url --------------------
    def _default_base_url(self, provider: str) -> str:
        """
        根据 provider 返回一个常见的默认 base_url。
        - openai   -> 官方默认（返回空串，SDK 自带）
        - deepseek -> https://api.deepseek.com/v1
        - apiyi    -> https://api.apiyi.com/v1
        也可以根据你们环境自行扩展/修改。
        """
        if provider == "openai":
            return ""  # 为空表示用 SDK 的默认（即官方 OpenAI）
        if provider == "deepseek":
            return "https://api.deepseek.com/v1"
        if provider == "apiyi":
            return "https://api.apiyi.com/v1"
        return ""  # 未知提供商则不设默认（需要显式 LLM_BASE_URL）

    # -------------------- 内部工具：provider → 默认模型 --------------------
    def _default_model(self, provider: str) -> str:
        """
        根据 provider 返回一个可用的默认模型名（可按需修改）。
        """
        if provider == "openai":
            return "gpt-4o-mini"        # OpenAI 家的性价比模型
        if provider == "deepseek":
            return "deepseek-chat"      # DeepSeek 常用模型名（如与你们网关配置一致）
        if provider == "apiyi":
            return "gpt-4.1-mini"       # 参考你给出的中转示例
        return ""                        # 未知提供商：要求显式设置 LLM_MODEL
