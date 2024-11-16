"random synthetic datasets"
import sys
import  os
path_dir = os.path.dirname(__file__)
sys.path.append(path_dir)

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import os.path
from NssMPC.secure_model.mpc_party import SemiHonestCS
from application.GBDT.Protocols.online import *
from NssMPC.common.ring.ring_tensor import RingTensor
from application.GBDT.Protocols.offline import  SelectKeyOffline
from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider

def unique_values_by_column(matrix):
    # print(matrix.shape)
    unique_vals = [torch.unique(matrix[:, i]).unsqueeze(-1) for i in range(matrix.size(1))]
    # torch.cat(unique_vals, dim=1)
    return unique_vals



import warnings
warnings.filterwarnings("ignore")
def _max_with_index(x,indices,dim=0):
    def min_(inputs, indices):
        # 初始化索引
        if inputs.shape[0] == 1:
            return inputs, indices
        if inputs.shape[0] % 2 == 1:
            inputs_ = inputs[-1:]
            inputs = ArithmeticSecretSharing.cat([inputs, inputs_], 0)
            indices_ = indices[-1:]
            indices = ArithmeticSecretSharing.cat([indices, indices_], 0)
        inputs_0 = inputs[0::2]
        inputs_1 = inputs[1::2]
        indices_0 = indices[0::2]
        indices_1 = indices[1::2]

        b = inputs_0>inputs_1 # x<y=b
        min_values = b * inputs_0 + (1-b)* inputs_1
        min_indices = b * indices_0 + (1-b) * indices_1
        return min_values, min_indices  # 仅保留胜出的索引

    if dim is None:
        x = x.flatten()
        indices = indices.flatten()
    else:
        x = x.transpose(dim, 0)
        indices = indices.transpose(dim, 0)
    if x.shape[0] == 1:
        return x.transpose(dim, 0).squeeze(-1), indices.transpose(dim, 0).squeeze(-1)
    else:
        x, indices = min_(x, indices)
    # 需要适当调整最后的转置和索引选择以匹配原始维度
    return _max_with_index(x.transpose(0, dim), indices.transpose(0, dim),dim)

def _max_with_index_threshold(x,indices,thresholds,dim=0):
    def max_(inputs,indices,thresholds):
        # 初始化索引
        if inputs.shape[0] == 1:
            return inputs, indices,thresholds
        if inputs.shape[0] % 2 == 1:
            inputs_ = inputs[-1:]
            inputs = ArithmeticSecretSharing.cat([inputs, inputs_], 0)
            indices_ = indices[-1:]
            indices = ArithmeticSecretSharing.cat([indices, indices_], 0)
            thresholds_ = thresholds[-1:]
            thresholds = ArithmeticSecretSharing.cat([thresholds,thresholds_],0)


        inputs_0 = inputs[0::2]
        inputs_1 = inputs[1::2]
        indices_0 = indices[0::2]
        indices_1 = indices[1::2]
        thresholds_0 = thresholds[0::2]
        thresholds_1 = thresholds[1::2]
        b = inputs_0< inputs_1 # x<y=b
        # bs =b.restore().convert_to_real_field()
        min_values = b * inputs_0 + (1-b) * inputs_1
        min_indices = b * indices_0 + (1-b) * indices_1
        min_thresholds = b*thresholds_0+(1-b)*thresholds_1
        return min_values, min_indices,min_thresholds  # 仅保留胜出的索引

    if dim is None:
        x = x.flatten()
        indices = indices.flatten()
        thresholds = thresholds.flatten()
    else:
        x = x.transpose(dim, 0)
        indices = indices.transpose(dim, 0)
        thresholds = thresholds.transpose(dim, 0)
    if x.shape[0] == 1:
        return x.transpose(dim, 0).squeeze(-1), indices.transpose(dim, 0).squeeze(-1), thresholds.transpose(dim, 0).squeeze(-1)
    else:
        x, indices,thresholds = max_(x, indices,thresholds)
    # 需要适当调整最后的转置和索引选择以匹配原始维度
    return _max_with_index_threshold(x.transpose(0, dim),indices.transpose(0, dim),thresholds.transpose(0,dim),dim)

def HEP_XGB_Sigmoid_protocol(x:ArithmeticSecretSharing):
    '''
    Args:
        x:

    Returns:
        0.5+0.5*x/(1+|x|)
    '''
    b= (x>=0)
    abs_x =2*b*x-x
    return RingTensor.convert_to_ring(0.5)+(x/abs_x)
def SiGBDT_Sigmoid(x:ArithmeticSecretSharing):
    e = ArithmeticSecretSharing.exp(x)
    return e/(e+RingTensor.convert_to_ring(1.0))

class Tree:
    def __init__(self, t=-1, m=-1):
        self.m = m
        self.t = t

    def __str__(self):
        return f"TreeList Object: m={self.m}, t={self.t}"

def SecureBuildTree(args, m0,data, thresholds,gradient:ArithmeticSecretSharing, hessian:ArithmeticSecretSharing,share_w: ArithmeticSecretSharing, T: list, h: int, maxinum_depth: int):
    if h in range(0, 2 ** (maxinum_depth - 1) - 1):
        share_m, share_t = SeureFindBestSplit(args,m0,data,gradient, hessian,share_w, thresholds)
        c = (share_m-m0)>=0
        c= c.restore().convert_to_real_field()
        m = share_m.restore().convert_to_real_field()
        t = share_t.restore().convert_to_real_field()
        z = (share_w.party.party_id == c)
        T[h].m = z * m.int() - (1 - z.int())
        T[h].t = z * t - (1 - z.int())
        logger.info(f"{threading.currentThread().name}--- the {h}-node has been completed!")
        if c:
            b_test = (data[:,m.int()]<=t)*share_w.party.party_id
        else:
            b_test = (data[:,m.int()]<=t)*(1-share_w.party.party_id)

        b = ArithmeticSecretSharing(RingTensor.convert_to_ring(b_test), party=share_w.party)
        wl = share_w * b
        wr = share_w*(1-b)

        SecureBuildTree(args,m0=m0,data=data,thresholds=thresholds,gradient=wl*gradient,hessian=wl*hessian, share_w=wl,T=T, h=2*h+1, maxinum_depth=maxinum_depth)
        SecureBuildTree(args,m0=m0, data=data, thresholds=thresholds,gradient=wr*gradient,hessian=wr*hessian,share_w= wr, T=T,h=2*h+2, maxinum_depth=maxinum_depth)
        # left = threading.Thread(target=SecureBuildTree,args=(args,m0,data,thresholds,wl*gradient, wl*hessian, wl,T, 2*h+1, maxinum_depth,), name="left")
        # right = threading.Thread(target=SecureBuildTree, args=(args, m0, data, thresholds, wl * gradient, wl * hessian, wl, T, 2 * h + 1, maxinum_depth,), name="right")
        # left.start()
        # right.start()
        # left.join()
        # right.join()

    elif h in range(2 ** (maxinum_depth - 1) - 1, 2 ** maxinum_depth - 1):
        G = gradient.sum(dim=-1)
        H = hessian.sum(dim=-1)
        if args.alg_type =="Ours":
            w = SecureLeafProtocol(G, H)
        else:
            w = -G/(H+0.01)
        logger.info(f"{threading.currentThread().name}---the {h}-leaf has been completed!")
        T[h].t = w
        T[h].m = -2
def SeureFindBestSplit(args,m0,data,gradient:ArithmeticSecretSharing, hessian:ArithmeticSecretSharing,share_w, thresholds):
    n, m = data.shape
    each_best_ts = []
    each_best_gs = []
    for j in range(0, m-1):
        threshold = thresholds[j].view(1, -1)
        feature_vector = data[:, j].unsqueeze(1)
        local_test = feature_vector<=threshold
        # server_id: 1 and client_id: 0
        if j < m0:
            share_local_test = ArithmeticSecretSharing(RingTensor(local_test * (1-share_w.party.party_id)),
                                                       party=share_w.party)
        else:
            share_local_test = ArithmeticSecretSharing(RingTensor((local_test * share_w.party.party_id)),party=share_w.party)
        wl_share = share_w.view(n,-1) * share_local_test
        # wr_share = share_w.view(n,-1)*(1-share_local_test)
        GL = gradient@wl_share
        HL = hessian@wl_share
        G = gradient.sum()
        H = hessian.sum()
        GR = G-GL
        HR= H-HL
        t = threshold.to(torch.int).view(-1)
        ts = ArithmeticSecretSharing(RingTensor.convert_to_ring(t*share_w.party.party_id),party=share_w.party)
        if args.alg_type=="Ours":
            gain = (H - HL) * (GL*GL) + (H - HR) * (GR*GR)
        else:
            gain = ((GL*GL)/(HL+0.01)+(GR*GR)/(HR+0.01)+(G*G)/(H+0.01))*RingTensor.convert_to_ring(0.5)
        gain = gain.squeeze(0)
        g, t = _max_with_index(gain, ts)
        each_best_gs.append(g.view(-1))
        each_best_ts.append(t.view(-1))

    ms = ArithmeticSecretSharing(RingTensor(torch.arange(m - 1) * share_w.party.party_id),
                                party=share_w.party)
    best_gs = ArithmeticSecretSharing.cat(each_best_gs)
    best_ts = ArithmeticSecretSharing.cat(each_best_ts)
    _, best_m, best_t = _max_with_index_threshold(best_gs, ms, best_ts)

    return best_m,best_t


def setup_client(arg):
    data_name = args.data_name
    max_height = args.max_height
    t = args.t
    path = f"{data_name}_num_samples_{args.num_samples}_num_features_{args.num_features}_num_bins_{args.num_bins}"

    file_name = f'../synthetic_data/{path}_client_data.pth'
    m0, thresholds, client_data = torch.load(file_name)
    n, m = client_data.shape
    client_tree_list = [Tree() for _ in range(2 ** max_height-1)]

    client = SemiHonestCS(type='client')
    client.set_comparison_provider()
    client.set_multiplication_provider()
    client.set_nonlinear_operation_provider()

    client.append_provider(ParamProvider(param_type=SelectKeyOffline))
    y = ArithmeticSecretSharing(RingTensor.zeros(size=(n, 1), dtype='float'), party=client)
    hat_y = ArithmeticSecretSharing(RingTensor.zeros(size=(n,1),dtype='float'), party=client)
    client.online()
    '''
    client training code
    '''

    M = []
    if args.alg_type == "Ours":
        sigmoid_function = SecureSigmoidProtocol
    elif args.alg_type == "SiGBDT":
        sigmoid_function = SiGBDT_Sigmoid
    elif args.alg_type == "HEP-XGB":
        sigmoid_function = HEP_XGB_Sigmoid_protocol

    for _ in range(t):
        p = sigmoid_function(hat_y)
        gradient = p - y
        hessian = p * (1 - p)
        gradient = gradient.squeeze(-1)
        hessian = hessian.squeeze(-1)
        root_share_w = ArithmeticSecretSharing(RingTensor.ones(size=(n,)), party=client)
        SecureBuildTree(args, m0, client_data, thresholds, gradient, hessian, root_share_w, client_tree_list, h=0,
                        maxinum_depth=max_height)
        M.append(client_tree_list)
    if os.path.exists("../model"):
        torch.save(M, f"../model/{data_name}_server_model.pth")
    else:
        os.mkdir("../model")
        torch.save(M, f"../model/{data_name}_server_model.pth")
    client.close()

def setup_server(args):
    data_name=args.data_name
    max_height=args.max_height
    t = args.t
    path = f"{data_name}_num_samples_{args.num_samples}_num_features_{args.num_features}_num_bins_{args.num_bins}"

    file_name = f'../synthetic_data/{path}_server_data.pth'
    logger.info(f"{data_name} is loading  data set")
    m0, thresholds, server_data = torch.load(file_name)
    logger.info(f"{data_name} loading is done")
    server_tree_list = [Tree() for _ in range(2 ** max_height-1)]
    n, m = server_data.shape
    server = SemiHonestCS(type='server')
    server.set_comparison_provider()
    server.set_nonlinear_operation_provider()
    server.set_multiplication_provider()
    y = server_data[:,-1].view(n,-1)
    y = ArithmeticSecretSharing(RingTensor.convert_to_ring(y), party=server)
    hat_y = ArithmeticSecretSharing(RingTensor.zeros(size=(n, 1), dtype='float'), party=server)
    server.online()
    st = time.time()
    '''
    server training code
    '''
    M =[]
    if args.alg_type=="Ours":
        sigmoid_function = SecureSigmoidProtocol
    elif args.alg_type=="SiGBDT":
        sigmoid_function = SiGBDT_Sigmoid
    elif args.alg_type=="HEP-XGB":
        sigmoid_function = HEP_XGB_Sigmoid_protocol
    for _ in range(t):
        p = sigmoid_function(hat_y)
        gradient = p-y
        hessian = p*(1-p)
        gradient = gradient.squeeze(-1)
        hessian = hessian.squeeze(-1)
        root_share_w = ArithmeticSecretSharing(RingTensor.ones(size=(n, )), party=server)
        SecureBuildTree(args,m0,server_data, thresholds,gradient,hessian,root_share_w, server_tree_list, h=0, maxinum_depth=max_height)
        M.append(server_tree_list)
    if os.path.exists("../model"):
        torch.save(M,f"../model/{data_name}_server_model.pth")
    else:
        os.mkdir("../model")
        torch.save(M, f"../model/{data_name}_server_model.pth")
    et = time.time()
    logger.info(f"alg:{args.alg_type}: training time  {et - st:.2f} second")
    server.close()



if __name__ == '__main__':
    import threading
    import NssMPC.config.configs as cfg
    import argparse
    from application.GBDT.utils.loger import setup_logger
    parser = argparse.ArgumentParser(description="Secure GBDT Training Protocol")
    parser.add_argument(
        "--max_height",
        type=int,
        default=5,
        help="Max tree height: int"
    )
    parser.add_argument(
        "--t",
        type=int,
        default=1,
        help="Tree numbers"
    )
    parser.add_argument(
        "--alg_type",
        type=str,
        default="HEP-XGB",
        choices=["Ours", "SiGBDT", "HEP-XGB"],
        help="Secure training algorithm: {Ours, SiGBDT, HEP-XGB}"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples"
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=10,
        help="Number of features"
    )

    # 新增 bin 数量参数
    parser.add_argument(
        "--num_bins",
        type=int,
        default=8,
        help="Number of bins"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="synthetic"
    )
    # 解析参数
    args = parser.parse_args()

    "random synthetic datasets"
    X = np.random.randint(0, 100, size=(args.num_samples, args.num_features))
    y = np.random.randint(0, 2, size=(args.num_samples, 1))
    num_bins = args.num_bins
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2031)
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    threshold = unique_values_by_column(torch.tensor(data))
    m = len(data[0])
    m0 = m // 2
    client_data = np.zeros(data.shape)
    client_data[:, 0:m0] = data[:, 0:m0]
    server_data = np.zeros(data.shape)
    server_data[:, m0:] = data[:, m0:]
    server_data = torch.tensor(server_data, dtype=torch.int64)
    client_data = torch.tensor(client_data, dtype=torch.int64)

    args = parser.parse_args()
    data_name = args.data_name
    file = f"{data_name}_num_samples_{args.num_samples}_num_features_{args.num_features}_num_bins_{args.num_bins}"
    os.makedirs(os.path.dirname(f"../synthetic_data/"), exist_ok=True)

    client_file_path = f'../synthetic_data/{file}_client_data.pth'
    server_file_path = f'../synthetic_data/{file}_server_data.pth'

    torch.save((m0, threshold, client_data), client_file_path)
    torch.save((m0, threshold, server_data), server_file_path)

    if args.alg_type == "HEP-XGB":
        cfg.GE_TYPE = "MSB"
    else:
        cfg.GE_TYPE = "SIGMA"
    logger = setup_logger(f"{args.alg_type}_WAN_{file}")
    logger.info(f"training GBDT on the data set--{data_name}:{file}")

    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client, args=(args,), name="client")
    server_thread = threading.Thread(target=setup_server, args=(args,), name="server")
    # Starting threads
    client_thread.start()
    server_thread.start()
    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()


