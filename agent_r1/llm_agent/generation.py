"""
Tool generation manager for LLM agents
工具生成/交互管理器：用于管理 LLM 在“带工具”的生成循环中的交互与状态维护
"""

import torch
import re
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .tensor_helper import TensorHelper, TensorConfig             # 本地辅助模块：封装padding/拼接/position_ids等张量处理
from agent_r1.tool.base import BaseToolEnv, BaseImageToolEnv      # 工具环境接口（文本工具环境/图像工具环境）

from verl import DataProto                                        # veRL 的通用批数据容器（tensors / non-tensors / meta）
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto # 将batch pad到分布式world_size的公倍数，再取消

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int                           # 在一次交互中允许的最大工具轮数
    max_prompt_length: int                   # 输入prompt（包含历史）的最大长度限制（token数）
    max_response_length: int                 # 整体响应tokens的最大长度上限（用于截断）
    max_response_length_single_turn: int     # 单轮（每次generate）所允许的最大生成长度
    use_batch_tool_calls: bool = False       # 是否启用批量工具调用（一次把batch里所有工具请求打包执行）

class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction
    工具生成主控：负责循环调用模型、解析工具调用、执行工具、拼接上下文、维护掩码与多模态输入等。
    """
    
    def __init__(
        self,
        tokenizer,                   # 文本分词/解码器（支持pad_token_id等）
        processor,                   # 多模态处理器（图像预处理、image token等；对纯文本也可为空）
        actor_rollout_wg,            # 负责推理/rollout的执行器（通常是分布式worker的句柄，带generate_sequences）
        config: ToolGenerationConfig,
        is_validation: bool = False, # 是否处于验证模式
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # 初始化张量帮助器：主要为了统一的pad/concat/position_ids等操作
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses.
        批量把字符串转成 input_ids（不加特殊token），按最长序列padding，返回['input_ids']张量
        """
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _example_level_pad(self, data: Union[List[Any], np.ndarray, torch.Tensor],
                           active_mask: torch.Tensor,
                           pad_value: Any = None) -> Union[List[Any], np.ndarray, torch.Tensor]:
        """Pad data according to active mask.
        按 active_mask（形如[True, False, True]）把一个“仅包含active样本的数据列表”扩充回 batch 大小，并在非活跃位填充 pad_value。
        
        Args:
            data: 仅对应 active=True 的条目（顺序与active_mask中的True顺序一致）
            active_mask: 逐样本布尔向量，标记哪些样本仍然活跃（需要接收对应的data）
            pad_value: 如果不提供，将根据首个元素类型自动选择：
                       - str -> ""
                       - list -> []
                       - torch.Tensor -> 与元素同形同dtype的pad填充值
        Returns:
            与 batch 大小等长的同类型容器（list/np.ndarray/torch.Tensor），非活跃位为 pad_value
        """
        batch_size = active_mask.shape[0]  # batch大小

        # 自动推断 pad_value
        if pad_value is None:
            if len(data) > 0:
                first_elem = data[0]
                if isinstance(first_elem, str):
                    pad_value = ""
                elif isinstance(first_elem, list):
                    pad_value = []
                elif isinstance(first_elem, torch.Tensor):
                    # 对张量：创建同形同dtype的pad token张量（值为tokenizer.pad_token_id）
                    pad_value = torch.full_like(first_elem, fill_value=self.tokenizer.pad_token_id, dtype=first_elem.dtype, device=first_elem.device)
                else:
                    raise NotImplementedError(f"[WARNING] Unsupported data type: {type(first_elem)}")
                
        # 先用 pad_value 填满
        padded_data = [pad_value] * batch_size
        
        # 将data依序填入 active=True 的位置
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_data[i] = data[s]
                s += 1
        
        # 若原始是 numpy，则转回 numpy；若原始是 tensor，则堆叠为tensor
        if isinstance(data, np.ndarray):
            padded_data = np.array(padded_data, dtype=object)
        elif isinstance(data, torch.Tensor):
            padded_data = torch.stack(padded_data, dim=0)
            
        return padded_data
    
    def _create_response_action_mask(self, responses_ids_list: List[List[int]], tool_responses_ids_list: List[List[int]]) -> List[List[int]]:
        """
        为“响应序列”创建action mask：区分哪些token是模型生成(1)，哪些是工具输出拼接(0)。
        
        Args:
            responses_ids_list: 每条样本，模型生成token ids列表
            tool_responses_ids_list: 每条样本，对应的工具响应token ids列表
            
        Returns:
            action_masks: 与“模型生成 + 工具响应”拼接后等长的掩码（模型=1，工具=0）
        """
        action_masks = []
        
        for model_ids, tool_ids in zip(responses_ids_list, tool_responses_ids_list):
            # 模型tokens -> 1，工具tokens -> 0
            action_mask = [1] * len(model_ids) + [0] * len(tool_ids)
            action_masks.append(action_mask)

        return action_masks
 
    def _update_rolling_state(self, rollings, responses_ids: torch.Tensor, 
                            tool_responses: List[str], tool_responses_images: List[List[Image.Image]]):
        """
        将“本轮模型生成 + 本轮工具响应（含图片）”拼接回 rollings（滚动的上下文状态），
        并维护：responses累积、action_mask累积、input_ids/attention/position_ids、多模态数据等。
        """
        is_multi_modal = "multi_modal_data" in rollings.non_tensor_batch.keys()  # 是否存在多模态流

        row_dict_list = []                 # 每条样本在本轮新增的多模态条目（若有）
        formatted_tool_responses = []      # 将工具响应文本做占位替换后的版本（如 <image> -> 视觉token序列）
        raw_tool_responses = []            # 工具响应的原文（不带替换）
        action_masks = []                  # 本轮的action mask（稍后计算）

        # 逐样本处理工具响应（含图像占位符替换）
        for i, (tool_response, tool_responses_image) in enumerate(zip(tool_responses, tool_responses_images)):
            row_dict={}
            if is_multi_modal and '<image>' in tool_response and tool_responses_image is not None:
                # 对于多模态：校验工具返回的图片数量是否与 <image> 占位符一致
                assert len(tool_responses_image) == tool_response.count('<image>'), f"[WARNING] TOOL RESPONSE IMAGE NUMBER NOT MATCH, {len(tool_responses_image)} != {tool_response.count('<image>')} for {tool_response}"
                # 原始文本中把 <image> 替换为视觉token边界串（稍后再用实际数量的占位符替换）
                raw_tool_responses.append(tool_response.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>'))
                # 记录本轮新增的图像数据
                row_dict['multi_modal_data'] = {'image': tool_responses_image}
                # 用processor将图像转成模型可接受的张量（如patches/grid信息）
                image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                image_grid_thw = image_inputs['image_grid_thw']  # 每张图像的grid形状信息（T, H, W）

                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2  # 一次merge的patch数
                    index = 0
                    # 将每个 <image> 替换为：vision_start + N个<|placeholder|> + vision_end
                    # 其中N = (T*H*W) // merge_length，最终再把placeholder统一替换成 image_token
                    while '<image>' in tool_response:
                        tool_response = tool_response.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        index += 1

                    # 真正替换占位符为模型定义的 image token（如 <image_token>）
                    tool_response = tool_response.replace('<|placeholder|>', self.processor.image_token)

            else:
                # 纯文本或无图像场景：原样记录
                raw_tool_responses.append(tool_response)
            formatted_tool_responses.append(tool_response)
            row_dict_list.append(row_dict)

        # 将（已处理占位）后的工具响应，转换为token ids
        tool_responses_ids = self._batch_tokenize(formatted_tool_responses)

        # 将“模型本轮生成 + 工具本轮响应”拼到 rollings.batch['responses'] 尾部（累积所有turns）
        if "responses" not in rollings.batch.keys():
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                responses_ids,
                tool_responses_ids
            ], pad_to_left=False)
        else:
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                rollings.batch['responses'],
                responses_ids,
                tool_responses_ids
            ], pad_to_left=False)

        # 截断responses总长度，防止越界
        rollings.batch['responses'] = rollings.batch['responses'][:, :self.config.max_response_length]

        # 将张量形式的 ids 去掉pad后，转成list，用于构建action mask
        responses_ids_list = []
        tool_responses_ids_list = []
        for i, (responses_ids_, tool_responses_ids_) in enumerate(zip(responses_ids, tool_responses_ids)):
            responses_ids_ = responses_ids_[responses_ids_ != self.tokenizer.pad_token_id].tolist()
            tool_responses_ids_ = tool_responses_ids_[tool_responses_ids_ != self.tokenizer.pad_token_id].tolist()
            responses_ids_list.append(responses_ids_)
            tool_responses_ids_list.append(tool_responses_ids_)

        # 为本轮（模型+工具）的拼接部分创建action mask（模型=1，工具=0）
        action_masks = self._create_response_action_mask(responses_ids_list, tool_responses_ids_list)

        # 将action mask累积到非张量批数据里（逐轮append）
        if "action_mask" not in rollings.non_tensor_batch.keys():
            rollings.non_tensor_batch['action_mask'] = np.array(action_masks, dtype=object)
        else:
            new_action_masks = []
            for i, action_mask in enumerate(rollings.non_tensor_batch['action_mask']):
                # 之前已有的action_mask + 本轮新增的action_mask
                new_action_masks.append(action_mask + action_masks[i])
            rollings.non_tensor_batch['action_mask'] = np.array(new_action_masks, dtype=object)

        # 把 input_ids 也同步累积（prompt + 历史responses + 本轮 new responses + 工具响应）
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            responses_ids,
            tool_responses_ids
        ])

        # 为新的 input_ids 创建 attention_mask
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        
        if is_multi_modal:
            # 取出历史的多模态流（图像数据与其张量表示）
            multi_modal_data = rollings.non_tensor_batch['multi_modal_data']
            multi_modal_inputs = rollings.non_tensor_batch['multi_modal_inputs']

            new_multi_modal_data = []
            new_multi_modal_inputs = []

            # 将本轮新增的图像数据/张量拼接到历史多模态数据上（逐样本）
            for row_dict, multi_modal_data_, multi_modal_inputs_ in zip(row_dict_list, multi_modal_data, multi_modal_inputs):
                if 'multi_modal_data' in row_dict.keys():
                    new_multi_modal_data.append({"image":multi_modal_data_['image'] + row_dict['multi_modal_data']['image']})
                else:
                    new_multi_modal_data.append({"image":multi_modal_data_['image']})
                if 'multi_modal_inputs' in row_dict.keys():
                    new_multi_modal_inputs.append({key: torch.cat((val,row_dict['multi_modal_inputs'][key]),dim=0) for key, val in multi_modal_inputs_.items()})
                else:
                    new_multi_modal_inputs.append({key: val for key, val in multi_modal_inputs_.items()})

            # 回写到non_tensor_batch
            rollings.non_tensor_batch['multi_modal_data'] = np.array(new_multi_modal_data, dtype=object)
            rollings.non_tensor_batch['multi_modal_inputs'] = np.array(new_multi_modal_inputs, dtype=object)

            # 对于多模态，需要根据图像grid和attention重新计算 position_ids（以Qwen2-VL的实现为例）
            from verl.models.transformers.qwen2_vl import get_rope_index
            new_postion_ids = []
            for i in range(len(new_multi_modal_data)):
                new_postion_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=new_input_ids[i],
                    image_grid_thw=new_multi_modal_inputs[i]['image_grid_thw'],
                    attention_mask=new_attention_mask[i],
                ))

            new_position_ids = torch.stack(new_postion_ids, dim=0)
        else:
            # 纯文本情况下，常规的position_ids创建
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # 更新 rollings 的batch张量
        rollings.batch['input_ids'] = new_input_ids
        rollings.batch['position_ids'] = new_position_ids
        rollings.batch['attention_mask'] = new_attention_mask

        # 维护 raw_prompt_ids（未pad的“真实输入序列”id列表）：用于后续越界检查/调试
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids'].tolist()
        new_raw_prompt_ids = []
        for raw_prompt_id, responses_ids_, raw_tool_response in zip(raw_prompt_ids, responses_ids_list, raw_tool_responses):
            if len(responses_ids_) > 0 or len(raw_tool_response) > 0:
                tool_response_ids = self.tokenizer.encode(raw_tool_response, add_special_tokens=False)
                new_raw_prompt_ids.append(raw_prompt_id + responses_ids_ + tool_response_ids)
            else:
                new_raw_prompt_ids.append(raw_prompt_id)

        rollings.non_tensor_batch['raw_prompt_ids'] = np.array(new_raw_prompt_ids, dtype=object)

        return rollings
    
    def run_llm_loop(self, gen_batch, env: Union[BaseToolEnv, BaseImageToolEnv]) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop.
        主循环：反复让模型生成 → 解析工具调用 → 执行工具 → 拼接结果 → 直到达到轮数/长度限制或无工具可调。
        """

        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)           # 哪些样本仍活跃（需要继续循环）
        turns = torch.zeros(batch_size, dtype=torch.int32)               # 每个样本已经经历的回合数
        active_num_list = [active_mask.sum().item()]                     # 记录每轮活跃样本数（调试用）
        rollings = gen_batch                                             # 将传入的batch作为滚动状态起点
        meta_info = gen_batch.meta_info  
                               # 保留元信息（数据来源等）
        prompts = gen_batch.batch['input_ids'][:, -self.config.max_prompt_length:].clone()
        # 保存截断后的prompt（最后max_prompt_length个token）

        # Main generation loop
        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break  # 无活跃样本，结束

            # 1) 基于attention_mask统计有效长度，超过prompt上限的样本直接置为不活跃
            effective_len = rollings.batch['attention_mask'].sum(dim=1)
            length_exceeded = effective_len > self.config.max_prompt_length
            if length_exceeded.sum() > 0:
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                active_mask[length_exceeded] = 0

            # 2) raw_prompt_ids 的越界检查（非pad版本），如果超长同样停用
            raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
            length_exceeded = [len(prompt_id) > self.config.max_prompt_length for prompt_id in raw_prompt_ids]
            if any(length_exceeded):
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                for prompt_id, length_exceeded_ in zip(raw_prompt_ids, length_exceeded):
                    if length_exceeded_:
                        print(f"[DEBUG] LENGTH {len(prompt_id)}: {self.tokenizer.decode(prompt_id)}")
                active_mask[length_exceeded] = 0
            
            if not active_mask.sum():
                print("[WARNING] NO ACTIVE SEQUENCES")
                break  # 再次确认：若没有活跃样本，提前结束
            
            # 3) 构建仅包含活跃样本的 DataProto（张量+非张量）。这会减少无用计算。
            if hasattr(rollings, 'non_tensor_batch') and rollings.non_tensor_batch:
                rollings_active = DataProto.from_dict(
                    tensors={ k: v[active_mask] for k, v in rollings.batch.items() },
                    non_tensors={ k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items() },
                    meta_info=meta_info
                )
            else:
                rollings_active = DataProto.from_dict(
                    tensors={ k: v[active_mask] for k, v in rollings.batch.items() },
                    meta_info=meta_info
                )
            # breakpoint()
            # # 打印当前输入给模型的 prompt（解码为可读文本）
            # input_ids = rollings.batch['input_ids']
            
            # # 解码整个batch的input_ids，打印每个样本对应的文本
            # decoded_prompts = [self.tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(batch_size)]
            
            # # 打印每个样本的prompt
            # for i, decoded_prompt in enumerate(decoded_prompts):
            #     print(f"\n[DEBUG] Input to Model (Prompt) for sample {i}:\n{decoded_prompt}")        
            # You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

            # # Tools

            # You may call one or more functions to assist with the user query.

            # You are provided with function signatures within <tools></tools> XML tags:
            # <tools>
            # {"type": "function", "function": {"name": "search", "description": "Search for information on the internet using Wikipedia as a knowledge source.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}}
            # </tools>

            # For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            # <tool_call>
            # {"name": <function-name>, "arguments": <args-json-object>}
            # </tool_call>
            # user
            # Question: Which was formed first, Noori or Test Icicles?
            # You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags.
            # assistant

            # 4) 为分布式world_size做padding，使batch能均匀切分到各GPU；生成后再unpad回原形
            # breakpoint()

            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)  # 模型生成：拿到本轮 raw responses
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            # 5) 从生成结果里取出“responses”字段，并交给env做必要的后处理（去噪、裁切、去特殊符号等）；这里的“raw_responses”是小模型的输出
            raw_responses_ids = gen_output.batch['responses']
            responses_ids = env.process_responses_ids(self.tokenizer, raw_responses_ids)
            raw_responses = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)  # 解码为字符串
            # '<think>\nTo determine which was formed first, Noori or Test Icicles, we need to look up 
            # information about each. Noori is a brand name, 
            # and Test Icicles is a type of ice cream. 
            # We can search for information on Wikipedia to find the formation dates of both.\n
            # </think>\n
            # <tool_call>\n\n{"name": "search", "arguments": 
            # {"query": "formation date Noori Test Icicles"}}
            # \n</tool_call>'


            # ==== 在这里开始插入 DEBUG 验证（紧接着 raw_responses 这一行之后）====
            # print("\n[DEBUG] === RAW MODEL RESPONSE (before tool env) ===")
            # for i, txt in enumerate(raw_responses):
            #     has_tool_call = "<tool_call>" in txt
            #     print(f"[{i}] has_tool_call={has_tool_call}\n{txt}\n{'-'*60}")
            # ===========================================================

            # 6) 解析工具调用并执行工具（文本工具或图像工具两类环境）
            if isinstance(env, BaseToolEnv):
                if self.config.use_batch_tool_calls:
                    # 批量工具：一次处理整个batch的工具调用
                    # breakpoint()
                    tool_responses, _, new_active_masks = env.batch_step(raw_responses)
                else:
                    # 逐条工具：对每条样本依次解析并执行
                    tool_responses = []
                    new_active_masks = []
                    for raw_response in raw_responses:
                        tool_response, _, active = env.step(raw_response)
                        tool_responses.append(tool_response)
                        new_active_masks.append(active)
                        # ==== DEBUG：工具环境返回的继续标记 & 注回上下文的工具响应预览 ====
                        # print("\n[DEBUG] new_active_masks:", new_active_masks)
                        # for i, tr in enumerate(tool_responses):
                        #     preview = (tr or "")[:200].replace("\n", " ")
                        #     print(f"[DEBUG] tool_responses[{i}] len={len(tr) if tr is not None else 0} preview={preview}")
                        # =================================================================

                tool_images = [[]] * len(raw_responses)  # 纯文本工具无图像
            elif isinstance(env, BaseImageToolEnv):
                if self.config.use_batch_tool_calls:
                    # 图像工具的批量接口：额外返回每条的图像列表
                    tool_responses, tool_images, _, new_active_masks = env.batch_step(raw_responses)
                else:
                    tool_responses = []
                    tool_images = []
                    new_active_masks = []
                    for raw_response in raw_responses:
                        # 对图像工具：可能返回 assistant_message（可选）、tool_message(要拼接的工具响应)、
                        # tool_image（图像数据）、success、stop（是否停止）
                        assistant_message, tool_message, tool_image, success, stop = env.step(raw_response)
                        tool_responses.append(tool_message)
                        tool_images.append(tool_image)
                        new_active_masks.append(stop)

            # 7) 将“仅包含活跃样本顺序”的outputs扩展回batch大小（用_example_level_pad）
            responses_ids = self._example_level_pad(responses_ids, active_mask)
            tool_responses = self._example_level_pad(tool_responses, active_mask, pad_value="")
            tool_images = self._example_level_pad(tool_images, active_mask, pad_value=[])

            # 8) 更新active_mask为这轮工具后的活跃状态（例如模型说停止/无工具 → False）
            active_mask[active_mask.clone()] = torch.tensor(new_active_masks, dtype=torch.bool)

            # 9) 活跃样本的回合计数+1
            turns[active_mask] += 1

            # 记录活跃样本数量（调试）
            active_num_list.append(active_mask.sum().item())

            # 10) 将“本轮生成 + 工具响应(文本/图像)”拼回rollings
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses,
                tool_images
            )
 
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # === 循环结束：组装最终输出 ===
        final_output = {}
        final_output['turns'] = turns
        final_output['prompts'] = prompts                         # 最初截取的prompt
        final_output['responses'] = rollings.batch['responses'].long()  # 全部累积responses（模型+工具拼接）
        final_output['input_ids'] = torch.cat([
            prompts,
            rollings.batch['responses'].long()
        ], dim=1)                                                # 最终输入（prompt + 全部responses）
        final_output['attention_mask'] = self.tensor_fn.create_attention_mask(final_output['input_ids'])

        # 计算最终 position_ids：多模态需按视觉grid重新计算，纯文本直接按mask生成
        if "multi_modal_data" in rollings.non_tensor_batch.keys():
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = []
            for i in range(len(rollings.non_tensor_batch['multi_modal_data'])):
                position_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=final_output['input_ids'][i],
                    image_grid_thw=rollings.non_tensor_batch['multi_modal_inputs'][i]['image_grid_thw'],
                    attention_mask=final_output['attention_mask'][i],
                ))
            position_ids = torch.stack(position_ids, dim=0)
            final_output['position_ids'] = position_ids
        else:
            final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])

        # === 构造 action_mask（训练时决定哪些token参与策略梯度） ===
        response_length = final_output['responses'].shape[-1]
        response_mask = final_output['attention_mask'][:, -response_length:]  # 只取responses部分的mask

        final_output['action_mask'] = response_mask.clone()  # 初始化：默认全1（即模型+工具都算）

        # 用我们在滚动过程中累积的 non_tensor_batch['action_mask']（模型=1 工具=0）覆盖
        for i, action_mask in enumerate(rollings.non_tensor_batch['action_mask']):
            mask_len = min(len(action_mask), response_mask.shape[1])
            final_output['action_mask'][i, :mask_len] = torch.tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
            # 注意：乘以response_mask是为了避免越界部分或padding位置被置1

        # 打包为 DataProto，沿用non_tensor_batch（包含multi_modal_data/inputs、action_mask等）
        final_output = DataProto.from_dict(final_output)
        final_output.non_tensor_batch = rollings.non_tensor_batch
        
        return final_output
