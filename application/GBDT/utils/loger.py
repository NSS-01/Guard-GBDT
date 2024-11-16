
import logging
import os
from datetime import datetime

import colorlog

# 创建一个日志处理器，用于文件输出
# 文件路径
def setup_logger(name,level=logging.INFO,timestamp=1):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if  timestamp == 1:
        timestamp = datetime.now().strftime("%Y%m%d")
    else:
        timestamp ="all"
    log_file_path = parent_dir+f"/logs/{timestamp}_{name}.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path,encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 创建一个彩色的日志处理器，用于控制台输出
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(console_formatter)

    # 获取一个日志对象，并添加处理器
    logger = logging.getLogger("PrettyLogger")
    logger.setLevel(level)  # 设置最低日志级别
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger



if __name__ == "__main__":
    # 示例日志
    logger = setup_logger(name="PrettyLogger", level=logging.INFO)
    logger.debug("这是调试信息")
    logger.info("这是一般信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
