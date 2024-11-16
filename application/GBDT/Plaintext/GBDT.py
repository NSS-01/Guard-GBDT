import time

import numpy as np
import torch
from tqdm import tqdm
from application.GBDT.utils.loger import setup_logger

class GBDT:
    def __init__(self, alg="Ours", n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=2, reg_lambda=0.1, n_segments=10,logger=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.trees = []
        self.n_segments = n_segments
        self.sigmoid_table = self.create_sigmoid_lookup_table(self.n_segments)
        self.alg = alg
        self.logger = logger
        self.sigmoid_function = None


    def sigmoid(self, x):
        """计算 sigmoid 函数的值"""
        return 1 / (1 + np.exp(-x))
    def HEP_XGB_sigmoid(self,x):
        return 0.5+0.7*x/(1+np.abs(x))
    def create_sigmoid_lookup_table(self, n):
        """
        创建查找表 LUT_delta，包含分段的 sigmoid 值。
        ω_i = -5 + 10 * i / n, i 从 0 到 n
        δ(ω_i) = sigmoid(ω_i)
        """
        lookup_table = {}
        for i in range(n + 1):
            omega_i = -5 + 10 * i / n
            lookup_table[i] = self.sigmoid(omega_i)
        return lookup_table

    def approximate_sigmoid(self, x):
        """
        使用查找表 LUT_delta 近似计算 sigmoid(x)。
        - x: 输入行向量
        - lookup_table: 预先计算好的查找表
        - n: 查找表的分段数量
        返回近似的 sigmoid 值的行向量。
        """
        lookup_table = self.sigmoid_table
        n = self.n_segments
        omega_values = np.linspace(-5, 5, n + 1)

        # 使用 np.digitize 找到每个 x 值对应的区间索引
        indices = np.digitize(x, omega_values) - 1

        # 将索引值裁剪到查找表的边界范围内
        indices = np.clip(indices, 0, n)

        # indices = indices.squeeze(-1)

        # 通过索引查找每个 x 值对应的近似 sigmoid 值
        approx_sigmoid_values = np.array([lookup_table[i] for i in indices])

        return approx_sigmoid_values

    def fit(self, X, y):
        y_pred = np.zeros_like(y)*1.0  # 初始预测值为0
        if self.alg == "Ours":
            sigmoid_function = self.approximate_sigmoid
        elif self.alg == "HEP_XGB":
            sigmoid_function = self.HEP_XGB_sigmoid
        else:
            sigmoid_function = self.sigmoid
        self.sigmoid_function = sigmoid_function
        for _ in range(self.n_estimators):
            p = sigmoid_function(y_pred.squeeze(-1))
            p =p.reshape(-1,1)
            gradient = p - y  # 对数损失的梯度
            hessian = p * (1 - p)  # 对数损失的Hessian
            tree = DecisionTree(alg=self.alg, max_depth=self.max_depth, min_samples_split=self.min_samples_split, reg_lambda=self.reg_lambda, n_segments=self.n_segments,logger=self.logger)
            tree.fit(X, gradient, hessian)
            prediction = tree.predict(X)
            hat_y = sigmoid_function(prediction)
            y_pred += self.learning_rate* hat_y.reshape(-1,1)
            self.trees.append(tree)


    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate*tree.predict(X)
        # print(y_pred)
        # print(y_pred)
        # print(self.sigmoid_function(y_pred))
        return self.sigmoid_function(y_pred)  # 使用sigmoid函数将输出转换为概率

def unique_values_by_column(matrix):
    unique_vals = [torch.unique(matrix[:, i]).unsqueeze(-1) for i in range(matrix.size(1))]
    return unique_vals

class DecisionTree:
    def __init__(self, alg="Ours", max_depth=3, min_samples_split=2, reg_lambda=1,n_segments=10,logger=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.tree = None
        self.n_segments = n_segments
        self.leaf_weight_table = self.__create_lookup_table(self.n_segments)
        self.alg = alg
        self.logger = logger

    def __create_lookup_table(self, n):

        """
        创建查找表LUT_w，包含ω'的近似值。
        ω_i' = -5 + 10 * i / n, 其中 i 从 0 到 n。
        """
        lookup_table = [-5 + 10 * i / n for i in range(n + 1)]
        return lookup_table
    def approximate_weight(self, GI_prime, HI_prime):
        """
        根据输入的 -GI'/HI' 值和查找表LUT_w的值，返回近似的叶子权重w'。
       """
        lookup_table = self.leaf_weight_table
        value = - GI_prime / HI_prime
        n = len(lookup_table) - 1

        # 根据分段函数规则查找对应的ω'
        if value < lookup_table[0]:
            return lookup_table[0]
        elif value >= lookup_table[n]:
            return lookup_table[n]
        else:
            # 找到满足条件的ω'区间
            for i in range(n):
                if lookup_table[i] <= value < lookup_table[i + 1]:
                    return lookup_table[i]
    def fit(self, X, gradient,hessian):
        self.tree = self._build_tree(X, gradient, hessian)

    def _build_tree(self, X, gradient, hessian,depth=0):

        if len(gradient) < self.min_samples_split or depth==self.max_depth:
            if self.alg == "Ours":
                w = self.approximate_weight(gradient.sum(), hessian.sum() + self.reg_lambda)
                return w
            else:
                return -gradient.sum()/(hessian.sum() + self.reg_lambda)


        best_split = self._find_best_split(X, gradient, hessian)
        if best_split['gain'] == -float('inf'):
            if self.alg == "Ours":
                w = self.approximate_weight(gradient.sum(), hessian.sum() + self.reg_lambda)
                return w
            else:
                return -gradient.sum()/(hessian.sum() + self.reg_lambda)
        left_idx = best_split['left_idx']
        right_idx = best_split['right_idx']

        left_tree = self._build_tree(X[left_idx], gradient[left_idx], hessian[left_idx],depth + 1)
        right_tree = self._build_tree(X[right_idx], gradient[right_idx], hessian[right_idx],depth + 1)

        return {'feature': best_split['feature'], 'threshold': best_split['threshold'], 'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, gradient, hessian):
        best_bin_gin =[]
        best_bin_index = []
        _, m = X.shape
        for feature in range(m):
            threshold = np.unique(X[:, feature])
            b = X[:,feature] <= threshold.reshape(-1,1)
            GL = b@gradient
            HL = b@hessian
            G = gradient.sum()
            H = hessian.sum()
            GR = G-GL
            HR = H-HL
            beta = 0
            if self.alg == "Ours":
                gain = (H - HL) * (GL ** 2) + (H - HR) * (
                        GR ** 2)
            else:
                gain = 0.5*((GL**2)/(HL+self.reg_lambda) + (GR**2)/(HR+self.reg_lambda + beta)-(G**2)/(self.reg_lambda+H))
            gain = gain.squeeze(-1)
            best_index = np.argmax(gain)
            # print(gain[best_index])
            best_bin_gin.append(gain[best_index])
            best_bin_index.append(best_index)

        best_bin_gain = np.array(best_bin_gin)
        feature = np.argmax(best_bin_gain)
        # print("best_gin: ", best_bin_gin, f"max_f:{feature}")
        gain =best_bin_gain[feature]
        threshold =best_bin_index[feature]
        left_idx = np.where(X[:, feature] <= threshold)[0]
        right_idx = np.where(X[:, feature] > threshold)[0]
        best_split = {'feature': feature, 'threshold': threshold, 'gain': gain, 'left_idx': left_idx,
                      'right_idx': right_idx}
        return best_split

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

# 示例用法
if __name__ == "__main__":
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="GBDT plaintext training")

    # 配置参数
    parser.add_argument(
        "--alg_type",
        type=str,
        default="Ours",
        choices=["Ours","SiGBDT","HEP-XGB"],
        help="Algorithm type (default: Ours), Ours, SiGBDT, HEP-XGB"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=5,
        help="Number of estimators (default: 10)"
    )

    parser.add_argument(
        "--n_segments",
        type=int,
        default=20,
        help="Number of segments (default: 20)"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["breast_cancer", "phishing_websites", "credit", "skin", "covertype"],
        default="breast_cancer",
        help="Dataset name (choices: breast_cancer, phishing_websites, credit, skin, covertype)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="The maximum depth of the tree (default: 4)}"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1,
        help="learning learning (default:0.1)}"
    )

    # 解析参数
    args = parser.parse_args()
    from sklearn.metrics import accuracy_score
    data_name = args.data_name
    X_train, X_test, y_train, y_test = torch.load(f'../data/{data_name}.pth')
    alg = args.alg_type
    n_estimators = args.n_estimators
    n_segments = args.n_segments
    max_depth = args.max_depth
    logger = setup_logger(name=f"{data_name}_plaintext")

    if alg == "Ours":
        accs = []
        for n_segment in range(1,n_segments):
            model = GBDT(alg=alg, n_estimators=n_estimators, learning_rate= args.lr, max_depth=max_depth, min_samples_split=1, reg_lambda=1, n_segments=n_segment,logger=logger)
            st = time.time()
            model.fit(X_train, y_train)
            et = time.time()
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)  # 将概率转换为0或1
            accuracy = accuracy_score(y_test > 0, y_pred_binary)
            accs.append(round(accuracy * 100, 2))
            logger.info(f"algorithm:{alg},tree number: {args.n_estimators} ,n_segments: {n_segment}, depth:{max_depth}, accuracy:{accuracy*100:.2f}")
        logger.info(f"all segments acc: {accs}")
    else:
        model = GBDT(alg=alg, n_estimators=n_estimators, learning_rate=0.01, max_depth=max_depth, min_samples_split=1,
                     reg_lambda=1, logger=logger)
        st = time.time()
        model.fit(X_train, y_train)
        et = time.time()
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred >= 0.5).astype(int)  # 将概率转换为0或1
        accuracy = accuracy_score(y_test, y_pred_binary)
        logger.info(
        f"algorithm:{alg}, depth:{max_depth}, accuracy:{accuracy * 100:.2f}")