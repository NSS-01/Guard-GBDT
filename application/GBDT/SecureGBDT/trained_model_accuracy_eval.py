import logging
import os.path
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
from application.GBDT.utils.loger import setup_logger

class Tree:
    def __init__(self, f=-1, t=-1):
        self.f = f
        self.t = t

    def __str__(self):
        return f"f={self.f}, t={self.t}"


class DecisionTreeNode:
    """A decision tree node class for binary tree"""

    def __init__(self, feature_index=None, threshold=None):
        """
        - feature_index: Index of the feature used for splitting.
        - threshold: Threshold value for splitting.
        - left: Left subtree.
        - right: Right subtree.
        - value: Value of the node if it's a leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = None
        self.right = None
        self.value = None

    def is_leaf_node(self):
        """Check if the node is a leaf node"""
        return self.value is not None

    def __str__(self):
        return f"feature_index, threshold:{(self.feature_index, self.threshold)}"


def predict(tree, sample):
    """Predict the label for a given sample using the decision tree"""
    if tree.is_leaf_node():
        return tree.value
    if sample[tree.feature_index] <= tree.threshold:
        return predict(tree.left, sample)
    return predict(tree.right, sample)


def Merge_trees(server_tree_model, cleint_tree_model):
    if len(server_tree_model) != len(cleint_tree_model):
        return None
    def build_node(index):
        if cleint_tree_model[index].m == -2 and server_tree_model[index].m == -2:
            node = DecisionTreeNode()
            node.value = (cleint_tree_model[index].t + server_tree_model[index].t).tensor
            return node
        node = DecisionTreeNode()
        node.feature_index = cleint_tree_model[index].m + server_tree_model[index].m + 1
        node.threshold = cleint_tree_model[index].t + server_tree_model[index].t + 1
        node.left = build_node(2 * index + 1)
        node.right = build_node(2 * index + 2)
        return node

    return build_node(0)


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



if __name__ == "__main__":

    # data_name = 'skin'
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="prediction")

    # 配置 `data_name` 参数，限定可选值
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["breast_cancer", "phishing_websites", "credit", "skin", "covertype"],
        default="skin",
        help="Dataset name (choices: breast_cancer, phishing_websites, credit, skin, covertype)"
    )

    # 配置 `alg_type` 参数，限定可选值
    parser.add_argument(
        "--alg_type",
        type=str,
        default="SiGBDT",
        choices=["Ours", "SiGBDT", "HEP-XGB"],

        help="Algorithm type (choices: Ours, SiGBDT, HEP-XGB)"
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=4,
        help="Max tree height: int"
    )
    # 解析参数
    args = parser.parse_args()
    logger = setup_logger("accuacy_eval",timestamp=0)
    server_model_file = f"../model/{args.alg_type}_{args.data_name}_max_height_{args.max_height}_server_model.pth.pth"
    client_model_file = f"../model/{args.alg_type}_{args.data_name}_max_height_{args.max_height}_client_model.pth.pth"

    if not os.path.exists(server_model_file) or not os.path.exists(client_model_file):
        logging.info(f"{server_model_file} or {client_model_file} not exist")
        print(f"{server_model_file} or {client_model_file} not exist")
        exit(-1)
    # 配置函数:
    if args.alg_type == "Ours":
        sigmoid_function = approximate_sigmoid
    elif args.alg_type == "HEP-XGB":
        sigmoid_function = HEP_XGB_sigmoid
    else:
        sigmoid_function = sigmoid

    data_name = args.data_name

    server_information = torch.load(server_model_file)
    client_information = torch.load(client_model_file)
    lr = server_information["lr"]
    server_models = server_information["model"]
    client_models = client_information["model"]
    result = 0
    for server_model,client_model in zip(server_models, client_models):
        tree = Merge_trees(server_model, client_model)
        X_train, X_test, y_train, y_test = torch.load(f'../data/{data_name}.pth')
        # predictions = [sigmoid(predict(tree, sample))>0.5 for sample in X_test]
        predictions= [predict(tree, sample) for sample in X_test]
        prediction_np = np.stack(predictions)
        result += lr*prediction_np
    hat_y= sigmoid_function(result)>0.5
    accuracy = accuracy_score(y_test, hat_y)
    logger.info(f"algorithm {args.alg_type} test {len(client_models)} trees with lightweight {args.max_height}: accuracy: {accuracy*100:.2f} on {data_name} dataset")
