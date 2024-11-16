import subprocess
import itertools

# 参数列表
data_names = ["breast_cancer", "phishing_websites", "credit", "skin", "covertype"]
# data_names = ["covertype"]
alg_types = ["Ours", "SiGBDT", "HEP-XGB"]
max_heights = [4]
tree_numbers = [1]
reg_lambdas = [0.01]
learning_rates = [0.01]
# alg_types = ["Ours", "SiGBDT", "HEP-XGB"]

# 生成所有参数组合
combinations = list(
    itertools.product(data_names, max_heights, tree_numbers, reg_lambdas, learning_rates, alg_types)
)

# 依次运行所有组合
for data_name, max_height, t, reg_lambda, lr, alg_type in combinations:
    cmd = [
        "python", "secure_training_protocol.py",  # 替换为您的 Python 文件路径
        "--data_name", data_name,
        "--max_height", str(max_height),
        "--t", str(t),
        "--reg_lambda", str(reg_lambda),
        "--lr", str(lr),
        "--alg_type", alg_type
    ]

    print(f"Running: {' '.join(cmd)}")  # 输出当前命令以供调试

    # 执行命令并捕获输出和错误日志
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 输出命令执行结果
    print("Output:", result.stdout)
    if result.stderr:
        print("Log (Error):", result.stderr)
    print("-" * 50)  # 用于分隔不同组合的输出
