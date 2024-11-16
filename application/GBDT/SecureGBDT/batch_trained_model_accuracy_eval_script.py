import subprocess
import itertools

# 参数列表
data_names = ["breast_cancer", "phishing_websites", "credit", "skin", "covertype"]
alg_types = ["Ours", "SiGBDT", "HEP_XGB"]
# data_names = ["covertype"]
# alg_types = ["Ours", "SiGBDT"]
max_heights = [4]

# 生成所有参数组合
combinations = list(itertools.product(data_names, alg_types, max_heights))

# 批量运行所有组合
for data_name, alg_type, max_height in combinations:
    cmd = [
        "python", "trained_model_accuracy_eval.py",  # 替换为实际的 Python 文件路径
        "--data_name", data_name,
        "--alg_type", alg_type,
        "--max_height", str(max_height)
    ]

    print(f"Running: {' '.join(cmd)}")  # 打印命令以供调试

    # 执行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 打印输出结果和错误日志
    print("Output:", result.stdout)
    if result.stderr:
        print("Error Log:", result.stderr)
    print("-" * 50)  # 分隔线，用于区分不同组合的输出
