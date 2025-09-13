from typing import Dict, List, Any  # 类型注解支持
import os  # 操作系统路径、文件操作
from agent_r1.tool.base import BaseTool  # 导入Agent-R1定义的工具基类
import faiss  # Facebook AI的相似度搜索库，用于加载和查询向量索引
from FlagEmbedding import FlagAutoModel  # FlagEmbedding库，用于加载文本嵌入模型
import json  # JSON编解码
import asyncio
from openai import AsyncOpenAI  # 改用异步客户端

API_KEY = ""  # 替换为你的 OpenAI API Key
# MODEL = "gpt-4o-mini"  # 对应 mini 模型也是写 gpt-4o
# client = openai.OpenAI(api_key = API_KEY, base_url = "https://api.apiyi.com/v1")

class SearchTool(BaseTool):
    # 工具的基本元数据
    name = "prompt"  # 工具名称（Agent调用时识别用）
    description = "Give an explanation of the question in detail."       # 工具功能描述
    parameters = {
        "type": "object",  # 输入参数类型是对象
        "properties": {    # 对象的属性
            "prompt": {"type": "string", "description": "Give question's explanation"}  # 必须有一个prompt字段（字符串）
        },
        "required": ["prompt"]  # prompt是必填项
    }
        
    def __init__(self):
        super().__init__()
        # 初始化异步OpenAI客户端
        self.client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.apiyi.com/v1")
        # self.client = AsyncOpenAI(api_key=API_KEY)
        print("[DEBUG] ASYNC GPT CLIENT INITIALIZED")

    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        执行一次查询 -> 交给GPT-4o-mini
        """
        return asyncio.run(self._execute_async(args))
    
    async def _execute_async(self, args: Dict) -> Dict[str, Any]:
        """
        异步执行单次查询
        """
        try:
            prompt = args["prompt"]

            # 异步调用 GPT-4o-mini
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Please read the provided content (including the question to be answered and the solution), identify the question to be answered, and answer this question: "},
                    {"role": "user", "content": prompt}
                ]
            )

            # 提取回答文本
            answer = response.choices[0].message.content.strip()

            # 保持与原先一致的返回格式
            return {"content": json.dumps({"results": [answer]}), "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量执行 -> 每个prompt分别交给GPT
        """
        return asyncio.run(self._batch_execute_async(args_list))
    
    async def _batch_execute_async(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        异步批量执行，使用并发提升速度
        """
        async def _single_request(args):
            try:
                prompt = args["prompt"]
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Please read the provided content (including the question to be answered and the solution), identify the question to be answered, and answer this question: "},
                    {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content.strip()
                return {"content": json.dumps({"results": [answer]}), "success": True}
            except Exception as e:
                return {"content": str(e), "success": False}
        
        # 并发执行所有请求
        tasks = [_single_request(args) for args in args_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"content": str(result), "success": False})
            else:
                processed_results.append(result)
        
        return processed_results

# 未知提供商：要求显式设置 LLM_MODEL