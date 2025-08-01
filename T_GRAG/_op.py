import re,os
import json
import asyncio
import tiktoken
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from difflib import get_close_matches
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from scipy.spatial.distance import cosine
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from datetime import datetime

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)# 使用 encode_string_by_tiktoken 将原始描述 description 编码为 tokens，以便确定描述的长度。
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    ) #将 tokens 的前部分解码成原始文本（llm_max_tokens 限制）。这个解码后的文本将作为生成摘要的输入。
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base) #通过 prompt_template 模板填充 context_base 字典，构造实际传递给模型的提示（prompt）
    logger.debug(f"Trigger summary: {entity_or_relation_name}")# 记录调试信息，输出正在处理哪个实体或关系的摘要。
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    '''
    如果记录格式不符合预期（即不是一个有效的实体记录），就返回 None，表示没有找到有效的实体。
    例如，如果记录没有包含足够的属性（至少需要 4 个），或者第一个字段不是 "entity"，则不处理该记录。
    '''
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper()) #entity_name 是从 record_attributes[1] 中提取的实体名称，并且调用了 clean_str() 函数对它进行清洗，upper() 方法将名称转换为大写字母。
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key 
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    already_node = await knwoledge_graph_inst.get_node(entity_name) #根据 entity_name 从知识图谱中获取已存在的节点数据。如果节点存在，便将节点的相关数据添加到上述的列表中。
    #加载节点
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(), #根据 entity_name 从知识图谱中获取已存在的节点数据。如果节点存在，便将节点的相关数据添加到上述的列表中。
        key=lambda x: x[1],
        reverse=True,
    )[0][0]#将统计结果按频率降序排列，取频率最高的实体类型。返回最常见的实体类型作为最终的 entity_type。
    
    for node in nodes_data:
        if "description" in node:
            node["description"] += f"-data from {global_config['time']}-"

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    #合并所有的描述信息。首先从 nodes_data 和 already_description 中提取所有描述，去重后排序，最后通过 GRAPH_FIELD_SEP 连接成一个字符串。
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    '''
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    '''
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        timestamp=global_config["time"],
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    """
    时间戳
    """
    for edge in edges_data:
        if "description" in edge:
            edge["description"] += f"-data from {global_config['time']}-"
    
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )

    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                    "timestamp": global_config["time"],
                },
            )
    '''
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    '''
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order, timestamp=global_config["time"],
        ),
    )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage, #存储和操作图形结构数据
    entity_vdb: BaseVectorStorage, #这是一个向量数据库实例，用于存储抽取出来的实体的向量表示。可能用来进行检索和相似度计算。
    global_config: dict,
) -> Union[BaseGraphStorage, None]: #Union 是 typing 模块中的一个类型提示，它允许函数返回多个类型中的任何一个。
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    current_time = datetime.now().isoformat()
    ordered_chunks = list(chunks.items()) #将 chunks 字典转换成一个列表，列表中的每个元素是一个元组 (chunk_id, chunk_data)。这一步通常是为了方便按顺序处理每个文本块。

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],# "<|>"
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],# "##"
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],# "<|COMPLETE|>"
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]), #["organization", "person", "geo", "event"] #它将 PROMPTS["DEFAULT_ENTITY_TYPES"] 中的所有实体类型通过逗号连接成一个字符串。
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"] #这个提示信息可能用于在抽取实体过程中进行多轮交互。它可能告诉模型在实体抽取过程中，如果需要继续抽取更多的实体，应该如何处理。例如，可能会提示模型在当前文档中继续查找未识别的实体。
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"] #这个提示信息可能用于控制抽取的循环流程，尤其是在需要判断是否存在更多实体时。如果模型在抽取时判断是否还需要继续循环提取实体，这个提示可能会控制是否进入下一轮的抽取。

    already_processed = 0 #用于记录已处理的文本块数量。
    already_entities = 0 #用于记录已抽取的实体数量（包括重复的实体）
    already_relations = 0 #用于记录已识别的关系数量。

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]): #内部处理单个文本块的异步函数
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        sum_dict={}
        content = chunk_dp["content"]
        sum_content=await use_llm_func(PROMPTS['summary'].format(text=content))
        sum_dict[chunk_key]=sum_content
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=sum_content)
        final_result = await use_llm_func(hint_prompt)
        '''
        使用 entity_extract_prompt 格式化提示语，将 context_base 和 content 插入其中，并调用大语言模型（use_llm_func）进行实体提取。
        '''
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result) #将初步的提示语和结果打包成对话历史，并开始循环补充提取（最多 entity_extract_max_gleaning 次）
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result) #在每次循环中，使用 continue_prompt 继续进行实体提取，将结果追加到 final_result 中。
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
            #使用 if_loop_prompt 判断是否继续进行循环。如果模型返回的结果不是 "yes"，则停止循环
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        #print("records",records)
        '''
        使用分隔符将 final_result 切分成多个记录。分隔符包括 record_delimiter 和 completion_delimiter。
        '''
        maybe_nodes = defaultdict(list) #defaultdict 是 Python 标准库中的一个字典类，它允许在访问不存在的键时返回一个默认值（这里是空列表）
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record) #从 record 字符串中提取圆括号 () 内的内容。
            #print(record)
            if record is None:
                continue
            record = record.group(1)
            '''
            record.group(1) 返回第一个匹配的子串，即括号中的内容。如果没有找到括号中的内容，record 将被设置为 None，这时 continue 会跳过当前循环，处理下一个 record。
            '''
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            #if_entities["entity_name"] 作为键，实体数据作为值。
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
            #如果成功提取到关系（即 if_relation 不是 None），则将关系存储到 maybe_edges 字典中，键为 (src_id, tgt_id)，值为关系对象
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges),sum_dict

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    ) #异步处理每个文本块
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    summary_data={}
    for m_nodes, m_edges,sum_dict in results:
        summary_data.update(sum_dict)
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v) #tuple(sorted(k)) 是为了确保关系是无向图的，即 src_id 和 tgt_id 的顺序不影响结果。
    sum_file = os.path.join(global_config['working_dir'], "chunk_sum.json")
    with open(sum_file, 'w', encoding='utf-8') as file:
        json.dump(summary_data, file, ensure_ascii=False, indent=4)
    #print("base node",maybe_nodes)
    #print("base edge",maybe_edges)
    '''
    # 统一批量处理所有节点的名称
    entity_names = list(maybe_nodes.keys())
    entity_names_string = ", ".join(entity_names)
    # Prepare the merge prompt
    entity_merge_prompt = PROMPTS["merge_extraction"].format(entity_names=entity_names_string)
    # 使用 LLM 进行批量合并处理

    merged_entity_names = await use_llm_func(entity_merge_prompt)
    #print(type(merged_entity_names),merged_entity_names)
    merged_entity_names = merged_entity_names.split("<SEP>")
    print(type(merged_entity_names),merged_entity_names)
    merger_new_name_dict={}
    for merger_name in merged_entity_names:
        if '---' in merger_name:
            merger_name_list=merger_name.split('---')
            merger_name_newlist=list(map(lambda item: f'"{item}"', merger_name_list))
            merger_new=merger_name_newlist[0]        
            for element in merger_name_newlist:
                merger_new_name_dict[element]=merger_new


    print("old entity",len(maybe_nodes))
    maybe_nodes=merge_nodes_by_names(merged_entity_names,maybe_nodes)
    print("new entity",len(maybe_nodes),maybe_nodes)
    
    print("old edge",maybe_edges)
    maybe_edges=merge_relation_by_names(merger_new_name_dict,maybe_edges)
    print("new edge",maybe_edges)
    '''
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    ) #使用 asyncio.gather() 异步执行 _merge_nodes_then_upsert()，将合并后的节点数据（实体）插入知识图谱。
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb) #这一行代码将构建好的 data_for_vdb 字典通过 upsert 方法插入或更新到向量数据库 entity_vdb 中。
    return knwoledge_graph_inst


def merge_nodes_by_names(
    merged_entity_names: list[str], maybe_nodes: defaultdict[list]
):
    for merger_name in merged_entity_names:
        #print("merger_name",merger_name)
        if '---' in merger_name:
            merger_name_list=merger_name.split('---')
            merger_name_newlist=list(map(lambda item: f'"{item}"', merger_name_list))
            merger_new=merger_name_newlist[0]
            #print("change name",merger_new)
            for i in range(1,len(merger_name_newlist)):
                for j in range (len(maybe_nodes[merger_name_newlist[i]])):
                    maybe_nodes[merger_name_newlist[i]][j]['entity_name']=merger_new
                maybe_nodes[merger_new].extend(maybe_nodes[merger_name_newlist[i]])
                del maybe_nodes[merger_name_newlist[i]]
            #print(merger_new,"done",maybe_nodes[merger_new])
    #print("merge",maybe_nodes_merge)
    #print("new entity",len(maybe_nodes),maybe_nodes)
    return maybe_nodes

def merge_relation_by_names(
    merger_new_name_dict: dict, maybe_edges: defaultdict[list]
):
    for key in list(maybe_edges.keys()):
        print(key)
        new_key = tuple(merger_new_name_dict[element] if element in merger_new_name_dict else element for element in key)
        print(new_key)
        if new_key == key:
            continue
        if new_key:
            if new_key in maybe_edges:
                maybe_edges[new_key].extend(maybe_edges.pop(key))
            else:
                maybe_edges[new_key] = maybe_edges.pop(key)
    for key, value in maybe_edges.items():
    # 遍历列表中的每个字典
        for item in value:
            # 如果'src_id'或'tgt_id'的值存在于 merger_new_name_dict 的键中
            if item['src_id'] in merger_new_name_dict or item['tgt_id'] in merger_new_name_dict:
                # 如果'src_id'的值存在于 merger_new_name_dict 的键中，则更新'src_id'的值
                if item['src_id'] in merger_new_name_dict:
                    item['src_id'] = merger_new_name_dict[item['src_id']]
                # 如果'tgt_id'的值存在于 merger_new_name_dict 的键中，则更新'tgt_id'的值
                if item['tgt_id'] in merger_new_name_dict:
                    item['tgt_id'] = merger_new_name_dict[item['tgt_id']]
    return maybe_edges

def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)

        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):

    text_units=[]
    for index,dp in enumerate(node_datas):
        text_units.append([])
        for des in dp["description"]:
            #print('dp["description"]',dp["description"])
            #print("des",des)
            text_units[-1].append(des[2])
    print("text_units",text_units)
    #获取一跳节点 (直接相连的节点)
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    #all_one_hop_nodes 是包含所有一跳节点的集合。
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges]) #这部分代码收集所有与当前实体直接相连的节点（一跳节点），并将它们存储在 all_one_hop_nodes 集合中
    all_one_hop_nodes = list(all_one_hop_nodes)
    
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    ) #获取一跳节点数据：

    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    } #构建一跳文本单元查找表：
    #all_one_hop_text_units_lookup 是一个字典，方便后续快速查询文本单元与一跳节点的关系。

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None #转换为列表
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units_id=[k['id'] for k in all_text_units]
    print(f"chunk排序:{all_text_units_id}")
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units

async def handle_edges_data(all_edges_data,query_vector,embedding_func):
    new_all_edges_data=[]
    for edge in all_edges_data:
        edge_description=[]
        description_list=edge['description'].split("<SEP>")
        if len(description_list)<=3:
            new_all_edges_data.append(edge)
            continue
        else:
            for des in description_list:
                embedding = await embedding_func([des])
                des_vector=  embedding[0]
                sim = 1 - cosine(des_vector,query_vector)
                edge_description.append((des,sim))
            edge_description_sort=sorted(edge_description, key=lambda x: x[1], reverse=True)
            new_description='<SEP>'.join([item[0] for item in edge_description_sort[:3]])
            edge['description']=new_description
            new_all_edges_data.append(edge)   
    return  new_all_edges_data

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    query_vector,
    embedding_func
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    new_all_edges_data=await handle_edges_data(all_edges_data,query_vector,embedding_func)
    #print("new_all_edges_data",new_all_edges_data)
    
    all_edges_data = truncate_list_by_token_size(
        new_all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data

async def build_descrition_embedding(entity_name,description,type,rank,query_vector,nodes_descriptions_chunk_data,embedding_func):
    node_description=[]
    description_list=description.split("<SEP>")
    for des in description_list:
        try:
            embedding = await embedding_func([des])
            des_vector=  embedding[0]
            sim = 1 - cosine(des_vector,query_vector)
            #处理名称和描述格式
            #print("原始名字",entity_name)
            closest_name=get_close_matches(entity_name, nodes_descriptions_chunk_data.keys(), n=1, cutoff=0.85)
            #print(entity_name,closest_name)
            #print("原始描述",des)
            closest_des= get_close_matches(des, nodes_descriptions_chunk_data[closest_name[0]].keys(), n=1, cutoff=0.85)
            #print(des,"dddddd",closest_des)
            chunk_id=nodes_descriptions_chunk_data[closest_name[0]][closest_des[0]]
            node_description.append((des,sim,chunk_id))
        except IndexError:
            continue
    node_description_sort=sorted(node_description, key=lambda x: x[1], reverse=True)
    return {'entity_name':entity_name,'description':node_description_sort,"entity_type":type,'rank':rank}

def find_useful_description(data):
    all_description={}
    for item in data:
        for des in item['description']:
            all_description[(des[0],des[1],des[2])]=des[1]            
    uesful_description=dict(sorted(all_description.items(), key=lambda x: x[1], reverse=True)[:15])
    uesful_description_dict={}
    for key in uesful_description.keys():
        uesful_description_dict[key[0]]=(key[1],key[2])
    return uesful_description_dict

def second_hanle_node(nodes_data,useful_description):
    new_nodes_data=[]
    for node in nodes_data:
        new_descriptions=[]
        use_num=0
        ues_rank=0
        for des in node['description']:
            if des[0] in useful_description:
                new_descriptions.append(des)
                use_num=use_num+1
                ues_rank=ues_rank+des[1]
        if new_descriptions:
            node['description']=new_descriptions
            node['score']=ues_rank/use_num
            new_nodes_data.append(node)
    new_nodes_data = sorted(new_nodes_data, key=lambda x: x["score"], reverse=True)
    
    return new_nodes_data
def handle_SEP(useful_node_datas):
    output=[]
    for node in useful_node_datas:
        descriptions=node['description']
        new_descriptions='<SEP>'.join([item[0] for item in descriptions])
        node['description']=new_descriptions
        node['entity_type']=node['entity_type'].split('<SEP>')[0]
        output.append(node)
    return output
               

async def _build_single_time_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    print("results",results)
    if not len(results):
        return None
    
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    print("node_datas",node_datas)
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    print("node_datas--222",node_datas)
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites,  {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list) #转换为 CSV 字符串

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "data source time","content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([f"########text unit-{i}:",f"{t["time"]}" ,f"content:{t["content"]}##########"])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

async def _build_new_time_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    node_names=[r["entity_name"] for r in results]
    print("粗糙细粒度的node_names",node_names)

    node_file = os.path.join(global_config['working_dir'], "merged_nodes_descriptions_chunks.json")
    with open(node_file,'r', encoding='utf-8') as file:
        nodes_descriptions_chunk_data = json.load(file)
    
    embedding_func=global_config["embedding_func"]
    query_vector= await embedding_func([query],query=True)
    query_vector=query_vector[0]
    description_embedding=await asyncio.gather(
        *[build_descrition_embedding(r["entity_name"],r['description'],r['entity_type'],r['rank'],query_vector,nodes_descriptions_chunk_data,embedding_func) for r in node_datas]
    )
    useful_description_dict=find_useful_description(description_embedding)
    
    useful_node_datas=second_hanle_node(description_embedding,useful_description_dict)
    print('useful_node_datas',useful_node_datas)
    use_text_units = await _find_most_related_text_unit_from_entities(
        useful_node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        useful_node_datas, query_param, knowledge_graph_inst,query_vector,embedding_func
    )
       

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    
    output_useful_node_datas=handle_SEP(useful_node_datas)
    use_node_datas = truncate_list_by_token_size(
        output_useful_node_datas,
        key=lambda x: x["description"],
        max_token_size=600,
    ) 
    logger.info(
        f"Using {len(use_node_datas)} entites,  {len(use_relations)} relations, {len(use_text_units)} text units"
    )    
    for i, n in enumerate(use_node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list) #转换为 CSV 字符串

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "data source time","content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([f"########text unit-{i}:",f"{t["time"]}" ,f"content:{t["content"]}##########"])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

async def single_time_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    context = await _build_new_time_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        text_chunks_db,
        query_param,
        global_config
    )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    prompt = sys_prompt_temp.format(
        question=query,context_data=context
    )
    print(f'检索出来的数据是{context}')
    sys_prompt='You need to answer questions based on the provided knowledge.'
    response = await use_model_func(
        prompt,
        system_prompt=sys_prompt
    )
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response
