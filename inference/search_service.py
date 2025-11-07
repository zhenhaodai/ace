# encoding=utf-8
"""
独立的搜索服务模块
用于调用华为搜索API获取相关文档
"""
import hashlib
import hmac
import json
import time
from typing import List, Dict, Optional

import requests


def get_md5(my_string):
    """计算字符串的MD5哈希值"""
    hash_object = hashlib.md5()
    hash_object.update(my_string.encode())
    md5_hash = hash_object.hexdigest()
    return md5_hash


def get_hmac_sha256_search_plan(body, url, key='799de706dba4bac8047e2c25e741ed1c5c46e3d966a426c5b0afa271b43fc490'):
    """生成HMAC-SHA256签名用于API认证"""
    timestamp = int(time.time() * 1000)
    timestamp = str(timestamp)
    body = body.replace("\n", "")
    key = key.encode('utf-8')
    value = "timestamp=" + timestamp + "&url=" + url + "&body=" + body
    HAMCObj = hmac.new(key, value.encode('utf-8'), hashlib.sha256)
    auth = HAMCObj.hexdigest()
    return timestamp, auth


def get_search_plan_result(query, sn_prefix, env="zc", debug=False, url=""):
    """
    调用搜索API获取搜索结果

    Args:
        query: 搜索查询文本
        sn_prefix: sn号前缀（每个人的工号）
        env: 环境选择，"zc"表示众测环境，"effect"表示效果环境
        debug: 是否开启debug模式（debug模式较慢）
        url: 自定义服务URL（可选）

    Returns:
        API返回的搜索结果字典
    """
    service_url = ""
    if env == "zc":
        # 广州众测环境
        service_url = "http://10.97.130.5:3399/search-gpt"
    elif env == "effect":
        # 效果环境
        service_url = "http://10.97.130.186:3399/search-gpt"

    if url:
        service_url = url

    sn = sn_prefix + get_md5(query)
    body = {
        "query": query,
        "sn": sn,  # sn号  场景名+每次请求不同的后缀号
        "from": "CC002600",
        "sregion": "cn",
        "need_box_ids": [
            "108", "1801", "1311", "109", "200", "9003", "9007", "2601", "2602",
            "11004", "1200", "1221", "1260", "1430", "1002", "1000", "1001", "1112",
            "1113", "1114", "105", "1301", "1448", "1449", "1012", "1014", "1013",
            "1011", "2100", "2102", "2103", "1211", "1230", "1310", "1250", "8620",
            "8621", "8622", "8628", "1051", "4451", "1053", "8400", "8401", "8402",
            "8410", "8411", "1900044", "1900045", "1900046", "1900015", "1900011",
            "1900071", "1900006", "1900070", "1900013", "1900040", "1900023", "9995",
            "7720", "7721", "7722", "7723", "7724", "1900043", "2602", "9010"
        ],
        "device": {
            "vendor": "HUAWEI"
        },
        "extra_info": {}
    }

    if debug:
        body["extra_info"]["debug_search"] = True

    payload = json.dumps(body)

    sing1, sing2 = get_hmac_sha256_search_plan(payload, "/search-gpt")
    headers_ = {
        'Authorization': sing2,
        'timestamp': sing1,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", service_url, headers=headers_, data=payload)
    result = response.content.decode('utf-8')
    answer = json.loads(result)
    return answer


def extract_passages_from_search_result(search_result: Dict, top_k: int = 3) -> List[str]:
    """
    从搜索API返回的结果中提取文本段落

    Args:
        search_result: 搜索API返回的完整结果
        top_k: 返回的段落数量

    Returns:
        提取的文本段落列表
    """
    passages = []

    try:
        # 根据搜索API的返回格式提取文本内容
        # 这里需要根据实际的API返回格式进行调整
        # 常见的字段可能是: results, items, documents 等

        if "results" in search_result:
            for item in search_result["results"][:top_k]:
                if "content" in item:
                    passages.append(item["content"])
                elif "text" in item:
                    passages.append(item["text"])
                elif "snippet" in item:
                    passages.append(item["snippet"])

        elif "items" in search_result:
            for item in search_result["items"][:top_k]:
                if "content" in item:
                    passages.append(item["content"])
                elif "text" in item:
                    passages.append(item["text"])

        # 如果以上方法都没有提取到内容，可以尝试直接转换整个结果
        if not passages:
            # 保留原始结果的字符串形式作为fallback
            passages.append(json.dumps(search_result, ensure_ascii=False))

    except Exception as e:
        print(f"Error extracting passages from search result: {e}")
        # 返回一个包含错误信息的空列表
        passages = []

    return passages[:top_k]


def retrieve_context_from_api(
    query: str,
    sn_prefix: str,
    top_k: int = 3,
    env: str = "zc",
    debug: bool = False,
    url: str = ""
) -> List[str]:
    """
    调用搜索API并返回检索到的上下文段落
    这是一个兼容原始 retrieve_context 函数的接口

    Args:
        query: 搜索查询
        sn_prefix: 工号前缀
        top_k: 返回的文档数量
        env: 环境（"zc" 或 "effect"）
        debug: 是否开启debug模式
        url: 自定义URL

    Returns:
        检索到的文本段落列表
    """
    search_result = get_search_plan_result(
        query=query,
        sn_prefix=sn_prefix,
        env=env,
        debug=debug,
        url=url
    )

    passages = extract_passages_from_search_result(search_result, top_k=top_k)
    return passages


if __name__ == "__main__":
    # 测试代码
    # 不能多并发跑
    # env=zc 只能用这个环境跑
    # sn_prefix传入自己的工号
    # env="effect" 效果环境谨慎使用，影响小艺评测

    query = "坚持走中国特色社会主义政治发展道路全面发展什么"
    answer = get_search_plan_result(query, sn_prefix="y84387018", env="zc", debug=False)
    print("原始搜索结果：")
    print(json.dumps(answer, ensure_ascii=False, indent=2))

    print("\n提取的段落：")
    passages = extract_passages_from_search_result(answer, top_k=3)
    for i, p in enumerate(passages):
        print(f"\n段落 {i+1}:")
        print(p)
