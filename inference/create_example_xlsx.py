"""
Create an example xlsx file with Chinese claims for testing
"""
import pandas as pd

# Create sample data with Chinese claims
data = {
    'id': [1, 2, 3, 4, 5],
    'claim': [
        '中国是世界上人口最多的国家，并且拥有悠久的历史文化',
        '人工智能技术在医疗领域有广泛应用，可以辅助诊断疾病',
        '太阳系有八大行星，地球是其中唯一有生命的行星',
        '量子计算机的运算速度远超传统计算机，可以解决复杂问题',
        '全球变暖导致海平面上升，威胁沿海城市的安全'
    ],
    'label': ['SUPPORT', 'SUPPORT', 'SUPPORT', 'SUPPORT', 'SUPPORT']
}

df = pd.DataFrame(data)

# Save to xlsx
output_file = 'inference/example_chinese_claims.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Created example file: {output_file}")
print("\nSample data:")
print(df)
print("\nYou can now use this file with:")
print("python inference/decompose_vllm_chinese.py --model_path YOUR_MODEL_PATH --input_file inference/example_chinese_claims.xlsx")
