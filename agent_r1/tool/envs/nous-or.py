from agent_r1.tool.base import BaseToolEnv, BaseTool   # 引入工具环境与工具的抽象基类
from typing import List, Dict, Tuple, Any              # 类型注解
import re                                              # 正则表达式，用于解析 <tool_call> 标签
import json                                            # 解析/序列化 JSON

class NousToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tools = tools                                              # 保存外部传入的工具实例列表
        self.tool_map = {tool.name: tool for tool in self.tools}        # 以工具 name 建索引，便于快速查找
        self.tool_call_start = "<tool_call>"                            # LLM 输出中，工具调用的起始标记
        self.tool_call_end = "</tool_call>"                             # 工具调用的结束标记
        self.tool_response_start = "<tool_response>"                    # 工具响应的起始标记（用于返回给 LLM）
        self.tool_response_end = "</tool_response>"                     # 工具响应的结束标记
        self.eos_token = "<|im_end|>"                                   # 特定对话分隔/结束 token
        self.parallel_tool_calls = False                                # 是否允许一次并行多个工具调用（此处默认关闭）
        self.max_tool_response_length = max_tool_response_length        # 工具响应的最大字符长度（超出会截断）

    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:

    # ==== DEBUG：环境收到的模型原文 ====
        # print("\n[DEBUG] ENV.STEP RECEIVED RAW RESPONSE")
        # print(f"has_tool_call={'<tool_call>' in raw_response}")
        # print(raw_response)
        # print("-"*60)
    # ==================================
        # 从 LLM 的文本输出中解析出所有 <tool_call>...</tool_call> 片段（JSON）
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:                    # 若没有任何工具调用
            return "", [], False                    # 返回空响应、空成功列表、active=False（无需继续等待工具）
        if not self.parallel_tool_calls:            # 如果不支持并行，仅取第一条工具调用
            tool_calls = [tool_calls[0]]
        tool_responses = []                         # 收集每个工具调用的响应内容（字符串）
        tool_successes = []                         # 收集每个工具调用是否成功（布尔）

        for tool_call in tool_calls:
            if tool_call is None:                   # JSON 解析失败时会得到 None
                tool_responses.append("Error: JSONDecodeError")
                tool_successes.append(False)
            else:
                if "name" not in tool_call:         # 缺少工具名
                    tool_responses.append("Error: No tool name")
                    tool_successes.append(False)
                else:
                    tool_name = tool_call["name"]   # 取得工具名
                    if tool_name not in self.tool_map:      # 工具未注册/不存在
                        tool_responses.append("Error: ToolNotFoundError")
                        tool_successes.append(False)
                    else:
                        tool = self.tool_map[tool_name]     # 取得工具实例
                        # 校验参数是否符合该工具的 schema（通常由 BaseTool.validate_args 实现）
                        if not tool.validate_args(tool_call["arguments"]):
                            tool_responses.append("Error: Invalid tool arguments")
                            tool_successes.append(False)
                        else:
                            # 执行工具，并收集结果（统一要求返回 {"content": str, "success": bool}）
                            tool_result = tool.execute(tool_call["arguments"])
                            tool_responses.append(tool_result["content"])
                            tool_successes.append(tool_result["success"])

        # 将所有工具响应打包成一个带特殊标签的“用户消息”，再交回给 LLM 继续生成
        tool_response = self.format_tool_response(tool_responses)
        # 返回：格式化后的工具响应、每次工具调用是否成功列表、active=True（说明这轮确实触发了工具）
        return tool_response, tool_successes, True

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        # 为每个样本预先创建容器（注意：使用 [[]] * N 会共享引用，但后面每个 i 都整体替换为新列表，仍安全）
        batch_tool_responses = [[]] * len(raw_responses)  # 每个样本的工具响应（字符串列表）
        batch_tool_successes = [[]] * len(raw_responses)  # 每个样本的成功标记（布尔列表）
        batch_active = [True] * len(raw_responses)        # 每个样本这轮是否活跃（是否触发工具）
        success_tool_calls_arguments = {}                 # 记录可批量执行的工具调用参数：tool_name -> [arguments...]
        success_tool_calls_index = {}                     # 记录批量调用在原batch中的位置：tool_name -> [(i,j)...]

        for i, raw_response in enumerate(raw_responses):
            # 提取该样本的工具调用列表
            tool_calls = self.extract_tool_calls(raw_response)
            if len(tool_calls) == 0:                      # 没有工具调用
                batch_tool_successes[i] = []              # 对应该样本为空
                batch_active[i] = False                   # 标记该样本本轮不活跃
                batch_tool_responses[i] = []              # 空响应
                continue

            if not self.parallel_tool_calls:              # 不支持并行，仅取第一条
                tool_calls = [tool_calls[0]]

            tool_responses = []                           # 累积该样本每个调用的即时返回（可能是错误，或“Executing...”占位）
            tool_successes = []                           # 累积该样本每个调用的成功标记

            for j, tool_call in enumerate(tool_calls):    # 遍历该样本的工具调用
                if tool_call is None:                     # JSON 解析失败
                    tool_responses.append("Error: JSONDecodeError")
                    tool_successes.append(False)
                else:
                    if "name" not in tool_call:           # 缺少工具名
                        tool_responses.append("Error: No tool name")
                        tool_successes.append(False)
                    elif "arguments" not in tool_call:    # 缺少参数字段
                        tool_responses.append("Error: No tool arguments")
                        tool_successes.append(False)
                    else:
                        tool_name = tool_call["name"]
                        if tool_name not in self.tool_map:   # 工具未注册
                            tool_responses.append("Error: ToolNotFoundError")
                            tool_successes.append(False)
                        else:
                            tool = self.tool_map[tool_name]
                            if not tool.validate_args(tool_call["arguments"]):  # 参数校验失败
                                tool_responses.append("Error: Invalid tool arguments")
                                tool_successes.append(False)
                            else:
                                # 通过预聚合（同名工具合并）实现“跨样本批量执行”
                                if tool_name not in success_tool_calls_arguments:
                                    success_tool_calls_arguments[tool_name] = []  # 新建参数列表
                                    success_tool_calls_index[tool_name] = []      # 新建位置列表
                                tool_responses.append("Executing...")             # 先占位，执行完后回填
                                tool_successes.append(False)                      # 先标 False，执行完再写真值
                                success_tool_calls_arguments[tool_name].append(tool_call["arguments"])  # 记录参数
                                success_tool_calls_index[tool_name].append((i,j)) # 记录该调用在原batch中的(i=样本索引, j=该样本的第j个调用)

            batch_tool_responses[i] = tool_responses      # 写回该样本的即时响应列表
            batch_tool_successes[i] = tool_successes      # 写回该样本的即时成功标记
        # breakpoint()
        # === 统一“批量执行”阶段（按工具名聚合调用）===
        for tool_name, args_list in success_tool_calls_arguments.items():
            tool = self.tool_map[tool_name]               # 找到对应的工具实例
            batch_results = tool.batch_execute(args_list) # 一次性批量执行所有样本中该工具的调用
            # 将执行结果回填到对应的 (i,j) 位置  search_tool.py跳转到这里；batch_results是返回的数据
            for batch_result, (i,j) in zip(batch_results, success_tool_calls_index[tool_name]):
                assert batch_tool_responses[i][j] == "Executing..."     # 确保之前是占位
                batch_tool_responses[i][j] = batch_result["content"]    # 回填工具返回内容
                batch_tool_successes[i][j] = batch_result["success"]    # 回填成功标记

        # === 将每个样本的工具响应包装成 LLM 可继续消费的消息字符串 ===
        batch_tool_responses_ = []
        for i, tool_responses in enumerate(batch_tool_responses):
            if batch_active[i]:                                         # 如果该样本本轮活跃（存在工具调用）
                assert len(batch_tool_responses[i]) > 0
                batch_tool_responses_.append(self.format_tool_response(tool_responses))
            else:
                batch_tool_responses_.append("")                        # 不活跃样本返回空串（不追加任何消息）

        # 返回：每个样本的格式化工具响应字符串列表、每个样本的每次调用成功标记列表、每个样本是否活跃
        return batch_tool_responses_, batch_tool_successes, batch_active

    def stop(self, raw_response: str) -> bool:
        # 判断是否应当“停止”工具阶段：如果没有解析到任何工具调用，就返回 True（表示无需继续等待工具）
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return True
        else:
            return False

    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        # 解析 LLM 输出中所有 <tool_call>...</tool_call> 的内容（要求内容是 JSON）
        tool_calls = []
        # 用非贪婪匹配 .*?，并开启 re.DOTALL 让 . 能匹配换行
        pattern = re.compile(f"{re.escape(self.tool_call_start)}(.*?){re.escape(self.tool_call_end)}", re.DOTALL)
        for tool_call in re.findall(pattern, raw_response):  # 遍历每一段匹配到的“JSON文本”
            try:
                tool_call = json.loads(tool_call)            # 尝试解析 JSON
                tool_calls.append(tool_call)                 # 成功则加入列表
            except json.JSONDecodeError:
                tool_calls.append(None)                      # 失败则用 None 占位，后续会返回 JSONDecodeError

        return tool_calls

    def format_tool_response(self, tool_responses: List[str]) -> str:
        # 把工具的多个响应拼接成一段“用户角色”的消息，交给 LLM 继续生成
        tool_message = "<|im_end|>\n<|im_start|>user\n"      # 切到“user”角色，开始追加工具响应
        for i, tool_response in enumerate(tool_responses):
            # 对单条工具响应做长度限制（避免过长）
            if len(tool_response) > self.max_tool_response_length:
                tool_response = tool_response[:self.max_tool_response_length] + "..."
            # 用 <tool_response> 包住一条工具返回（LLM 在系统中会通过这些段落读取工具结果）
            tool_message += f"<tool_response>\n{tool_response}\n</tool_response>"
            if i < len(tool_responses) - 1:
                tool_message += "\n"                          # 多条之间换行分隔
        # 结束“user”，切回“assistant”，并进入 <think>（思考阶段，具体取决于上游约定的格式）
        tool_message += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return tool_message
