import subprocess
import os

# 获取当前工作目录
current_directory = os.path.abspath(os.getcwd())

init_script_cmd = ["python", "offline_data_preprocess.py"]
init_script_dir = os.path.join(current_directory, "./")
# 定义第一个脚本的命令和目录
first_script_cmd = ["python", "test_GBDT_script.py"]
first_script_dir = os.path.join(current_directory, "Plaintext")

# 定义第二个脚本的命令和目录
second_script_cmd = ["python", "batch_training_script.py"]
second_script_dir = os.path.join(current_directory, "SecureGBDT")

# 定义第三个脚本的命令和目录
third_script_cmd = ["python", "batch_trained_model_accuracy_eval_script.py"]
third_script_dir = os.path.join(current_directory, "SecureGBDT")
# 定义第三个脚本的命令和目录
forth_script_cmd = ["python", "training_synthetic_data_main_script.py"]
forth_script_dir = os.path.join(current_directory, "SecureGBDT")

# 初始化脚本
print("Running the init script...")
init_result = subprocess.run(init_script_cmd, cwd=init_script_dir, capture_output=True, text=True)
print("First Script Output:", init_result.stdout)
if init_result.stderr:
    print("First Script log:", init_result.stderr)

# 运行第一个脚本
print("Running the first script...")
first_result = subprocess.run(first_script_cmd, cwd=first_script_dir, capture_output=True, text=True)
print("First Script Output:", first_result.stdout)
if first_result.stderr:
    print("First Script log:", first_result.stderr)

# 运行第二个脚本（在第一个脚本结束后）
print("Running the second script...")
second_result = subprocess.run(second_script_cmd, cwd=second_script_dir, capture_output=True, text=True)
print("Second Script Output:", second_result.stdout)
if second_result.stderr:
    print("Second Script log:", second_result.stderr)

# 运行第三个脚本（在第二个脚本结束后）
print("Running the third script...")
third_result = subprocess.run(third_script_cmd, cwd=third_script_dir, capture_output=True, text=True)
print("Third Script Output:", third_result.stdout)
if third_result.stderr:
    print("Third Script log:", third_result.stderr)
# 运行第四个脚本（在第三个脚本结束后）
print("Running the third script...")
forth_result = subprocess.run(forth_script_cmd, cwd=forth_script_dir, capture_output=True, text=True)
print("Third Script Output:", forth_result.stdout)
if forth_result.stderr:
    print("Third Script log:", forth_result.stderr)

