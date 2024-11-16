import subprocess
import itertools

# 参数列表
max_heights = [4]
tree_numbers = [1]
# alg_types = ["Ours","SiGBDT"]
alg_types = ["Ours","SiGBDT","HEP-XGB"]
num_samples_list = [10000, 50000]
num_features_list = [10, 20]
num_bins_list = [8,16]
data_names = ["synthetic"]

# 生成所有参数组合
combinations = list(
    itertools.product(max_heights, tree_numbers, alg_types, num_samples_list, num_features_list, num_bins_list,
                      data_names))

# 依次运行所有组合
for max_height, t, alg_type, num_samples, num_features, num_bins, data_name in combinations:
    cmd = [
        "python", "./training_on_synthetic_data.py",
        "--max_height", str(max_height),
        "--t", str(t),
        "--alg_type", alg_type,
        "--num_samples", str(num_samples),
        "--num_features", str(num_features),
        "--num_bins", str(num_bins),
        "--data_name", data_name
    ]

    print(f"Running: {' '.join(cmd)}")

    # 执行命令并捕获输出
    # result = subprocess.run(cmd, capture_output=True, text=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="")

    # # 输出命令执行结果
    # print("Output:", result.stdout)
    # print("Log:", result.stderr)
