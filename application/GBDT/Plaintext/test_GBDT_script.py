import subprocess
import itertools

# 参数列表
alg_types = ["Ours", "SiGBDT", "HEP-XGB"]
n_estimators_list = [10]
n_segments_list = [20]
data_names = ["breast_cancer", "phishing_websites", "credit", "skin", "covertype"]
max_depth_list = [4, 8]

# 生成所有参数组合
combinations = list(itertools.product(alg_types, n_estimators_list, n_segments_list, data_names, max_depth_list))

# 依次运行所有组合
for alg_type, n_estimators, n_segments, data_name, max_depth in combinations:
    # 构造命令行参数列表
    cmd = [
        "python", "GBDT.py",
        "--alg_type", alg_type,
        "--n_estimators", str(n_estimators),
        "--n_segments", str(n_segments),
        "--data_name", data_name,
        "--max_depth", str(max_depth)
    ]

    print(f"Running: {' '.join(cmd)}")

    # 执行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 输出命令执行结果
    print("Output:", result.stdout)
    print("Log:", result.stderr)
    print("-" * 40)
