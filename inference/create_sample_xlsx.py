"""
生成示例 xlsx 文件用于测试中文 claim 分解脚本

文件格式：
- 第一列：编号（ID）
- 第二列：claim（要分解的中文陈述）
- 第三列：人工测评结果（T/F/uncertain）
"""

import pandas as pd
import os

# 示例中文 claims 数据
# 格式：编号, claim, 人工测评结果
sample_data = {
    'ID': [1, 2, 3, 4, 5],
    'claim': [
        '张三是一位著名的物理学家，他在2020年获得了诺贝尔物理学奖，并在北京大学任教',
        '这部电影由李四导演，于2021年上映，在中国大陆的票房超过10亿元人民币',
        '王五是中国的一位企业家，他创立了一家科技公司，该公司在2019年成功上市',
        '这本书由赵六撰写，于2018年出版，获得了茅盾文学奖，并被翻译成20多种语言',
        '刘七是一位运动员，他在2016年里约奥运会上获得了金牌，并打破了世界纪录'
    ],
    'label': [
        'T',         # True - 支持
        'F',         # False - 反驳
        'uncertain', # 不确定
        'T',         # True - 支持
        'F'          # False - 反驳
    ]
}

# 创建 DataFrame
df = pd.DataFrame(sample_data)

# 保存为 xlsx 文件
output_dir = 'sample_data'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'sample_claims.xlsx')
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"示例文件已生成: {output_file}")
print(f"\n文件内容预览:")
print(df.to_string(index=False))
print(f"\n共 {len(df)} 条 claim 数据")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n使用方法:")
print(f"python inference/decompose_vllm_chinese.py \\")
print(f"    --model_path /path/to/your/model \\")
print(f"    --input_file {output_file} \\")
print(f"    --output_dir output \\")
print(f"    --expname sample_test")
