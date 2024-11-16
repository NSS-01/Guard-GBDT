import time


import math
import numpy as np
import torch

from NssMPC.secure_model.mpc_party import SemiHonestCS

from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from application.GBDT.SecureGBDT.utils import path
from application.GBDT.utils.loger import setup_logger

import threading


def sigmoid( x):
    """计算 sigmoid 函数的值"""
    return 1 / (1 + np.exp(-x))


def HEP_XGB_sigmoid(x):
    return 0.5 + 0.5 * x / (1 + np.abs(x))


def create_sigmoid_lookup_table(n=10):
    """
    创建查找表 LUT_delta，包含分段的 sigmoid 值。
    ω_i = -5 + 10 * i / n, i 从 0 到 n
    δ(ω_i) = sigmoid(ω_i)
    """
    lookup_table = {}
    for i in range(n + 1):
        omega_i = -5 + 10 * i / n
        lookup_table[i] = sigmoid(omega_i)
    return lookup_table
def approximate_sigmoid(x):
    """
    使用查找表 LUT_delta 近似计算 sigmoid(x)。
    - x: 输入行向量
    - lookup_table: 预先计算好的查找表
    - n: 查找表的分段数量
    返回近似的 sigmoid 值的行向量。
    """
    lookup_table =create_sigmoid_lookup_table()
    n = len(lookup_table)-1
    omega_values = np.linspace(-5, 5, n + 1)

    # 使用 np.digitize 找到每个 x 值对应的区间索引
    indices = np.digitize(x, omega_values) - 1

    # 将索引值裁剪到查找表的边界范围内
    indices = np.clip(indices, 0, n)

    # indices = indices.squeeze(-1)

    # 通过索引查找每个 x 值对应的近似 sigmoid 值
    approx_sigmoid_values = np.array([lookup_table[i] for i in indices])

    return approx_sigmoid_values


# 创建一个全局锁
lock = threading.Lock()
class Tree:
    def __init__(self, m=-1, t=-1):
        self.m = m
        self.t = t
    def __str__(self):
        return f"f={self.m}, t={self.t}"
def predict(x, tree_list, tree_indices, p,c, h, party):
    if len(tree_list) == 0:
        return []
    if len(tree_list) == 1:
        if isinstance(tree_list.t,RingTensor):
            y_share = ArithmeticSecretSharing(tree_list.t, party=party)
            ground_y = y_share.restore().convert_to_real_field()
            logger.info(f"inference result:{ground_y}")
        return ground_y
    p = path(x, tree_list, tree_indices, p, h)

    p = RingTensor.convert_to_ring(torch.tensor(p))
    zore = RingTensor.zeros_like(p)
    if party.party_id == 1:
        p0 = ArithmeticSecretSharing(p, party=party)
        p1 = ArithmeticSecretSharing(zore, party=party)
    else:
        p0 = ArithmeticSecretSharing(zore, party=party)
        p1 = ArithmeticSecretSharing(p, party=party)

    label = ArithmeticSecretSharing(RingTensor.stack(c), party=party)
    res = (p0*p1*label).sum()
    return res


def batch_predict(X, tree_list, tree_indices, p, c, h, party):
    results = [None]*len(X) # 初始化结果列表，与输入样本数量一致
    # 创建并启动线程
    for i, x in enumerate(X):
        result = predict(x, tree_list, tree_indices, p,c, h, party)
        results[i]= result
    return results


def setup_server(args,server_models, X_test,y_test):
    server = SemiHonestCS(type='server')
    server.set_comparison_provider()
    server.set_multiplication_provider()
    server.online()

    n = len(server_models[-1])
    h = int(math.log2(n+1))
    tree_indices = torch.arange(0,n)
    p = np.ones(2 ** (h - 1))
    c = [node.t for node in server_models[-1][n//2:]]

    st = time.time()
    res =[0]*len(X_test)
    for server_model in server_models:
        predictions = batch_predict(X_test, server_model, tree_indices, p, c, h, server)
        for i, prediction in enumerate(predictions):
            res[i] +=0.01*prediction.restore().convert_to_real_field()
    et = time.time()
    if args.alg_type=="Ours":
        sigmoid_function=approximate_sigmoid
    elif args.alg_type=="HEP_XGB":
        sigmoid_function = HEP_XGB_sigmoid
    else:
        sigmoid_function = sigmoid

    res = (sigmoid_function(np.array(res))>=0.5).astype(int)
    server.close()


def setup_client(args,client_models,X_test,y_test):
    client = SemiHonestCS(type='client')
    client.set_comparison_provider()
    client.set_multiplication_provider()

    # p, tree_indices, c = init_inference(T, h)
    client.online()

    n = len(client_models[-1])
    h = int(math.log2(n + 1))
    tree_indices = torch.arange(0, n)
    p = np.ones(2**(h-1))
    c = [node.t for node in client_models[-1][n // 2:]]
    res = [0]*len(X_test)
    for client_model in client_models:
        predictions = batch_predict(X_test, client_model, tree_indices, p, c, h, client)
        for i, prediction in enumerate(predictions):
            res[i]+= 0.1*prediction.restore().convert_to_real_field()
    client.close()


if __name__ == '__main__':
    import threading
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="prediction")

    # 配置 `data_name` 参数，限定可选值
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["breast_cancer", "phishing_websites", "credit", "skin", "covertype"],
        default="breast_cancer",
        help="Dataset name (choices: breast_cancer, phishing_websites, credit, skin, covertype)"
    )

    # 配置 `alg_type` 参数，限定可选值
    parser.add_argument(
        "--alg_type",
        type=str,
        choices=["Ours", "SiGBDT", "HEP_XGB"],
        default="SiGBDT",
        help="Algorithm type (choices: Ours, SiGBDT, HEP_XGB)"
    )

    # 解析参数
    args = parser.parse_args()
    # data_name = 'bank_marketing'
    data_name = args.data_name
    logger = setup_logger(data_name)
    server_models = torch.load(f"../model/{data_name}_server_model.pth")
    # print(f"server--- tree{[(node.m, node.t) for node in server_model]}")
    client_models = torch.load(f"../model/{data_name}_client_model.pth")
    # print(f"client----tree{[(node.m, node.t) for node in client_model]}")
    _, X_test, _, y_test = torch.load(f'../data/{data_name}.pth')


    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client,args=(args,client_models,torch.tensor(X_test),y_test), name="client")
    server_thread = threading.Thread(target=setup_server,args=(args,server_models,torch.tensor(X_test),y_test), name="server")
    # Starting threads
    client_thread.start()
    server_thread.start()

    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()
