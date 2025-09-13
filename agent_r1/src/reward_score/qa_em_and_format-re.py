# Copyright 2024 Bytedance Ltd. and/or its affiliates
# 版权所有声明（版权归字节跳动及/或其关联公司所有）
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 采用 Apache 2.0 开源许可证
# you may not use this file except in compliance with the License.
# 使用本文件须遵守许可证条款
# You may obtain a copy of the License at
# 许可证链接如下
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     Apache 2.0 许可证的官方网址
#
# Unless required by applicable law or agreed to in writing, software
# 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,
# 否则按“现状”提供本软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何形式的保证（明示或暗示）
# See the License for the specific language governing permissions and
# 具体权利和限制见许可证
# limitations under the License.
# 许可证中的限制条款

import re             # 正则表达式模块，用于模式匹配（提取标签内容、替换等）
import random
import string         # 字符串工具，包含标点符号集合 string.punctuation
from eval import cal_em, cal_f1

def normalize_answer(s):
    # 归一化答案字符串：小写化、去标点、去冠词(a/an/the)、压缩空白
    def remove_articles(text):
        # 去掉英文冠词 a/an/the（\b 表示词边界，避免匹配到单词内部）
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # 将任意多空白压缩为单个空格，并去除首尾空白
        return " ".join(text.split())

    def remove_punc(text):
        # 去掉所有 ASCII 标点符号（来自 string.punctuation）
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 全部转为小写
        return text.lower()

    # 归一化流水线：小写 -> 去标点 -> 去冠词 -> 压缩空白
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    # 严格 EM（exact match）检查：预测与任一金标完全相等（在归一化后）
    if isinstance(golden_answers, str):
        # 若只传入一个字符串，转为列表以统一处理
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)  # 归一化预测
    score = 0.0                                           # 默认得分0（不匹配）
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)   # 逐个归一化金标答案
        if golden_answer == normalized_prediction:        # 精确匹配
            score = 1.0
            break                                         # 一旦匹配成功即可退出
    return score


def subem_check(prediction, golden_answers):
    # 子串匹配的“宽松 EM”：只要金标是预测的子串即可判为匹配
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)  # 归一化预测
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)   # 归一化金标
        if golden_answer in normalized_prediction:        # 子串包含判定
            score = 1.0
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # 从文本中抽取 <answer>...</answer> 标签包裹的答案内容
    answer_pattern = r'<answer>(.*?)</answer>'            # 非贪婪匹配 <answer> 与 </answer> 之间
    match = re.search(answer_pattern, solution_str, re.DOTALL)  # DOTALL 让 '.' 匹配换行
    
    if match:
        return match.group(1).strip()                     # 返回去首尾空白的内容
    return None                                           # 找不到则返回 None

def compute_score_format(solution_str):
    """The scoring function for format reward.
    计算“格式分”的函数：检查生成是否符合预期的对话/工具调用/思考标记结构
    Args:
        solution_str: the solution text
    """
    if solution_str is None:
        return 0.0                                       # 空输入直接返回0
    
    try:
        # 预期结构（示例意图）：
        # 若有工具：assistant 块包含 <think> 与 <tool_call>；其后可能出现 tool 的 <tool_response>
        # 最后一个 assistant 块包含 <answer>，整个段落被 <|im_start|>assistant ... <|im_end|> 包裹

        # 先提取所有 assistant 块（每个块是 <|im_start|>assistant 与 <|im_end|> 之间的内容）
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0                               # 初始化格式分
        
        # 如果一个块都没有，则格式不合格
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        
        # 对“非最后一个”assistant块：检查是否各含一次 <think> 和 <tool_call>（严格计数=1）
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if (assistant_block.count('<think>') == 1 and
                assistant_block.count('</think>') == 1 and
                assistant_block.count('<tool_call>') == 1 and
                assistant_block.count('</tool_call>') == 1):
                # 进一步用正则严格校验结构：必须是 <think>... </think> 后紧跟 <tool_call>... </tool_call>
                think_match = re.search(
                    r'^<think>(.*?)</think>(\s*)<tool_call>(.*?)</tool_call>$',
                    assistant_block,
                    re.DOTALL
                )
                # soft_think_match = ...（软匹配示例被注释掉）
                if think_match:
                    # 给这一轮（该非末尾 assistant 块）加格式奖励（0.5）
                    # （注：原注释提到过指数衰减 0.2 * 0.8**i，这里实际使用常量 0.5）
                    format_reward += 0.5

        # 对“最后一个”assistant块：检查是否包含 <think>... </think> 与 <answer>... </answer>
        last_assistant_block = assistant_blocks[-1]
        think_answer_match = re.search(
            r'^<think>(.*?)</think>(.*?)<answer>(.*?)</answer>$',
            last_assistant_block,
            re.DOTALL
        )
        if think_answer_match:
            format_reward += 0.5                          # 满足结构再加0.5，总体格式分上限接近1
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")  # 调试输出异常信息
        return 0.0

    return format_reward                                  # 返回格式得分（未做上限约束，但下游会 min(.,1.0)）


def compute_score_answer(solution_str, ground_truth):
    """The scoring function for exact match (EM) with format reward.
    计算“答案分”（更关注答案是否匹配），这里实现了子串匹配为1分、全局子串为0.2分的规则
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    Returns:
        float: Total reward score (format reward + answer reward)
        实际这里只返回答案分（格式分另由 compute_score_format 计算）
    """
    if solution_str is None:
        return 0.0
    
    try:
        # 和上面一样，先找所有 assistant 块
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            solution_str,
            re.DOTALL
        )
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        solution_str = assistant_blocks[-1]          # 只用“最后一个”assistant 块作为最终回答来源
        answer = extract_solution(solution_str)      # 从中提取 <answer> 标签内的文本

        answer_reward = 0.0
        
        if answer is not None:
            # 这里把“子串匹配”直接给 1.0 分（注：严格EM与半分方案被注释掉）
            # if subem_check(answer, ground_truth):
            #     answer_reward = 1.0
            answer_reward = cal_f1([ground_truth.tolist()],[answer])
        
        # 如果 <answer> 内未命中，再退而求其次：在“最后一个 assistant 块全文”里找子串，给 0.2 分
        # if answer_reward == 0.0:
        #     if subem_check(solution_str, ground_truth):
        #         answer_reward = 0.2
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")  # 异常时直接返回0
        return 0.0
    
    return answer_reward


def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for format reward.
    计算“总分”的函数（格式分 + 答案分 + 一个基线偏移），用于训练中的奖励塑形
    Args:
        solution_str: the solution text
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)  # 先算格式分
        answer_reward = compute_score_answer(solution_str, ground_truth)  # 再算答案分

        format_reward = min(format_reward, 1.0)             # 将格式分上限裁为1.0
        # if format_reward >= 0.5:
        #     # 当格式达标（≥0.5）时：总分 = -1 + 格式分 + 答案分
        #     # 这里的 “-1” 是一个基线（baseline/offset），利于RL稳定或奖励稀疏时的区分度
        #     return -1.0 + format_reward + answer_reward
        # else:
        #     # 格式不达标：不给答案分，只返回 -1 + 格式分（鼓励先学会正确的输出结构）
        #     return -1.0 + format_reward
        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -1.0                                       # 异常时给一个负分，表示无效/错误轨迹


def compute_score_em(solution_str, ground_truth):
    """The scoring function for exact match (EM).
    仅做 EM（这里用 subem_check，即子串匹配作为“命中”）的评分函数，返回 0/1
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        # 同样提取所有 assistant 块
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            solution_str,
            re.DOTALL
        )
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        solution_str = assistant_blocks[-1]          # 取最后一个
        answer = extract_solution(solution_str)      # 抽取 <answer>
        if answer is None:
            return 0.0
        # return float(subem_check(answer, ground_truth))  # 命中返回1.0，否则0.0
        return float(cal_em([ground_truth.tolist()], [answer]))
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_em: {e}")
        return 0.0

def compute_score_f1(solution_str, ground_truth):
    """The scoring function for exact match (F1).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        return float(cal_f1([ground_truth.tolist()],[answer]))
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_f1: {e}")
        return 0.0
