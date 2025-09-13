from typing import Dict, List, Any
import os
from agent_r1.tool.base import BaseTool
import json
from openai import OpenAI  # 引入 OpenAI 库来调用 GPT-4o-mini API


class SearchTool(BaseTool):
    # 工具的基本元数据
    name = "prompt"  # 工具名称（Agent调用时识别用） 
    description = "Provide the relevant prompt for this question to a stronger assistant to answer it together with you."  
    # 工具功能描述
    parameters = {
        "type": "object",  # 输入参数类型是对象
        "properties": {    # 对象的属性
            "prompt": {"type": "string", "description": "Provide prompt"}  # 必须有一个query字段（字符串）
        },
        "required": ["prompt"]  # query是必填项
    }

    def __init__(self):
        super().__init__()
        print("[DEBUG] GPT-4o-mini API Client Loading")  # 调试输出，表示开始加载API客户端
        openai_key = ""
        # 用环境变量 OPENAI_API_KEY 进行鉴权
        self.client = OpenAI(api_key=os.getenv(openai_key))
        self.system_prompt = (
            "You are a helpful tool. Answer the user's query clearly and concisely. "
            "Do not include <tool_response> or any tool tags."
        )

    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        执行一次搜索
        Args:
            args: 包含搜索参数的字典（必须包含"query"字段）
        Returns:
            格式化后的搜索结果（字典形式，包含"content"和"success"）
        """
        try:
            query = args["query"]  # 获取查询字符串

            # 调用 GPT-4o-mini API 生成答案
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,  # 控制生成的多样性
            )

            # 获取 GPT-4o-mini 生成的文本
            text = resp.choices[0].message.content or ""
            return {"content": text, "success": True}  # 返回生成的内容和成功标志
        except Exception as e:
            return {"content": str(e), "success": False}  # 出错时返回错误信息

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量执行多条搜索
        Args:
            args_list: 参数字典的列表，每个字典包含一个"query"
        Returns:
            与输入等长的结果列表，每个元素是{"content":..., "success":...}
        """
        try:
            results = []
            for x in args_list:
                query = x["query"]

                # 调用 GPT-4o-mini API 生成答案
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,  # 控制生成的多样性
                )

                # 获取 GPT-4o-mini 生成的文本
                text = resp.choices[0].message.content or ""
                results.append({"content": text, "success": True})

            return results
        except Exception as e:
            return [{"content": str(e), "success": False} for _ in args_list]  # 出错时为每个输入返回错误信息

    def _format_results(self, results: List) -> str:
        """
        将搜索结果ID列表转为可读文本，并用JSON封装
        本方法不再需要，因为现在直接返回 GPT-4o-mini 生成的结果
        Args:
            results: 一个搜索结果ID的列表（例如[12, 45, 78, 90, 3]）
        Returns:
            JSON字符串，格式为 {"results": [文本1, 文本2, ...]}
        """
        results_list = []  # 用于保存实际文本结果
        
        # 现在不需要用 ID 查找 corpus 了，直接返回GPT生成的文本
        # 这里返回空列表，表示不再依赖 FAISS 语料库
        return json.dumps({"results": results_list})
