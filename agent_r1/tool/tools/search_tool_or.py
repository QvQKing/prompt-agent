"""
Search tool implementation for simulating internet searches
搜索工具的实现，用于模拟互联网搜索（实际上是基于本地维基百科/HotpotQA语料进行检索）
"""

from typing import Dict, List, Any  # 类型注解支持
import os  # 操作系统路径、文件操作

from agent_r1.tool.base import BaseTool  # 导入Agent-R1定义的工具基类

import faiss  # Facebook AI的相似度搜索库，用于加载和查询向量索引
from FlagEmbedding import FlagAutoModel  # FlagEmbedding库，用于加载文本嵌入模型
import json  # JSON编解码

class SearchTool(BaseTool):
    # 工具的基本元数据
    name = "search"  # 工具名称（Agent调用时识别用）
    description = "Search for information on the internet using Wikipedia as a knowledge source."  
    # 工具功能描述
    parameters = {
        "type": "object",  # 输入参数类型是对象
        "properties": {    # 对象的属性
            "query": {"type": "string", "description": "Search query"}  # 必须有一个query字段（字符串）
        },
        "required": ["query"]  # query是必填项
    }
    
    def __init__(self):
        # 调用父类构造函数初始化
        super().__init__()
        print("[DEBUG] EMBEDDINGS LOADING")  # 调试输出，标记嵌入模型加载开始
        
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 语料数据所在目录（这里直接写了绝对路径）
        # 注释掉的是相对路径写法，可以按需修改
        # data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/corpus/hotpotqa"))
        data_dir = os.path.abspath(os.path.join(current_dir, "/data/yichao/Agent-R1/data/corpus/hotpotqa"))
        
        # 1. 加载FAISS向量索引文件
        self.index = faiss.read_index(os.path.join(data_dir, "index.bin"))
        
        # 2. 加载文本嵌入模型
        self.model = FlagAutoModel.from_finetuned(
            'BAAI/bge-large-en-v1.5',  # 微调后的模型名称
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            # 给模型的查询提示，用于优化检索嵌入
            devices="cpu",   # 指定运行设备（cpu），如果不写会自动选择可用GPU或CPU
        )
        
        # 3. 加载文本语料（与索引向量对应的原文）
        self.corpus = []  # 保存文本的列表
        with open(os.path.join(data_dir, "hpqa_corpus.jsonl"), "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)  # 解析JSON行
                # 拼接标题和正文，存入corpus
                self.corpus.append(data['title'] + " " + data["text"])
        
        print("[DEBUG] EMBEDDINGS LOADING END")  # 调试输出，加载结束

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
            
            # 将查询编码为向量（输入必须是列表形式）
            embeddings = self.model.encode_queries([query])
            
            # 在FAISS索引中搜索最相似的5个结果
            dist, ids = self.index.search(embeddings, 5)  # ids的形状是 batch_size * 5
            
            # 格式化搜索结果（取第一条查询的结果id列表）
            result_str = self._format_results(ids[0])
            
            return {"content": result_str, "success": True}  # 成功返回
        except Exception as e:
            # 出错时返回错误信息
            return {"content": str(e), "success": False}
    
    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量执行多条搜索
        Args:
            args_list: 参数字典的列表，每个字典包含一个"query"
        Returns:
            与输入等长的结果列表，每个元素是{"content":..., "success":...}
        """
        try:
            # 提取所有查询字符串
            queries = [x["query"] for x in args_list]
            
            # 批量编码查询向量*38   进来的是这里？！
            embeddings = self.model.encode_queries(queries)
            
            # 批量搜索，每条查询返回top 5结果 ids.shape
            dist, ids = self.index.search(embeddings, 5)  # ids的形状是 batch_size * 5
            
            # 逐条格式化搜索结果
            results_str = [self._format_results(ids[i]) for i in range(len(ids))]
            
            # 返回成功标志的结果列表
            return [{"content": result_str, "success": True} for result_str in results_str]
        except Exception as e:
            # 出错时，为每个输入返回错误信息
            return [{"content": str(e), "success": False} for _ in args_list]

    def _format_results(self, results: List) -> str:
        """
        将搜索结果ID列表转为可读文本，并用JSON封装
        Args:
            results: 一个搜索结果ID的列表（例如[12, 45, 78, 90, 3]）
        Returns:
            JSON字符串，格式为 {"results": [文本1, 文本2, ...]}
        """
        results_list = []  # 用于保存实际文本结果
        
        for i, result in enumerate(results):
            # 根据索引ID取出corpus中的原文
            results_list.append(self.corpus[result])
        
        # 返回JSON字符串
        return json.dumps({"results": results_list})
