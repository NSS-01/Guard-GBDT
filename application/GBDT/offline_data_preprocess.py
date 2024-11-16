# key gen:
import os.path
import warnings

from sklearn.feature_selection import VarianceThreshold
from application.GBDT.Protocols.offline import SelectKeyOffline
warnings.filterwarnings("ignore")

from NssMPC.crypto.aux_parameter import *
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.crypto.aux_parameter.truncation_keys.ass_trunc_aux_param import AssTruncAuxParams
def gen_key():
    gen_num = 1000
    l_ring = 64
    lf_ring =16
    AssMulTriples.gen_and_save(gen_num, saved_name='2PCBeaver', num_of_party=2)
    BooleanTriples.gen_and_save(gen_num, num_of_party=2)

    Wrap.gen_and_save(gen_num)
    GrottoDICFKey.gen_and_save(gen_num)
    RssMulTriples.gen_and_save(gen_num)
    DICFKey.gen_and_save(gen_num)
    SigmaDICFKey.gen_and_save(gen_num)
    ReciprocalSqrtKey.gen_and_save(gen_num)
    DivKey.gen_and_save(gen_num)
    GeLUKey.gen_and_save(gen_num)
    RssTruncAuxParams.gen_and_save(gen_num)
    B2AKey.gen_and_save(gen_num)
    AssTruncAuxParams.gen_and_save(gen_num)
    SelectKeyOffline.gen_and_save(100000, None, l_ring, lf_ring)



# data processing
def config_dataset():
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, KBinsDiscretizer
    def unique_values_by_column(matrix):
        # print(matrix.shape)
        unique_vals = [torch.unique(matrix[:, i]).unsqueeze(-1) for i in range(matrix.size(1))]
        # torch.cat(unique_vals, dim=1)
        return unique_vals
    def save(threshold,data,data_name):
        m = len(data[0])
        m0 = m // 2
        # print(data[:, 0:m0])
        # print(data[:, m0:])
        client_data = np.zeros(data.shape)
        client_data[:, 0:m0] = data[:, 0:m0]
        server_data = np.zeros(data.shape)
        server_data[:, m0:] = data[:, m0:]
        server_data = torch.tensor(server_data,dtype=torch.int64)
        client_data = torch.tensor(client_data,dtype=torch.int64)

        client_file_path= f'./data/{data_name}_client_data.pth'
        server_file_path= f'./data/{data_name}_server_data.pth'
        torch.save((m0,threshold,client_data),client_file_path )
        torch.save((m0,threshold,server_data), server_file_path )

    # '''breast_cancer'''
    # breast_cancer = load_breast_cancer()
    # X, y = breast_cancer.data, breast_cancer.target
    # y = y.reshape(-1, 1)
    # num_bins = 8
    # binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    # X = binner.fit_transform(X)
    # # 使用 VarianceThreshold 移除常量特征
    # selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    # X_reduced = selector.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2023)
    # torch.save((X_train, X_test, y_train, y_test), './data/breast_cancer.pth')
    # data = np.hstack((X_train, y_train.reshape(-1, 1)))
    # thresholds = unique_values_by_column(torch.tensor(data))
    #
    # m = len(data[0])
    #
    # m0 = m // 2
    # # print(data[:, 0:m0])
    # # print(data[:, m0:])
    # client_data = np.zeros(data.shape)
    # client_data[:, 0:m0] = data[:, 0:m0]
    # server_data = np.zeros(data.shape)
    # server_data[:, m0:] = data[:, m0:]
    # client_data, server_data = torch.tensor(client_data, dtype=torch.int64), torch.tensor(server_data,
    #                                                                                       dtype=torch.int64)
    # torch.save((m0,thresholds,client_data),'./data/breast_cancer_client_data.pth')
    # torch.save((m0,thresholds,server_data),'./data/breast_cancer_server_data.pth')
    # print(f"breast_cancer is done")
    # "phishing_websites"
    # # fetch dataset
    # phishing_websites = fetch_ucirepo(id=327)
    #
    # data = phishing_websites.data
    #
    # X = phishing_websites.data.features
    # y = phishing_websites.data.targets
    # y = np.where(y == -1, 0, y)
    #
    # # y = (np.array(phishing_websites.data.targets).reshape(-1)>0)*1
    # num_bins = 8
    # binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    # X = binner.fit_transform(X)
    # # 使用 VarianceThreshold 移除常量特征
    # selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    # X_reduced = selector.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2028)
    # torch.save((X_train, X_test, y_train, y_test), './data/phishing_websites.pth')
    # data = np.hstack((X_train, y_train.reshape(-1, 1)))
    # thresholds = unique_values_by_column(torch.tensor(data))
    # save(thresholds, data, 'phishing_websites')
    # print(f"phishing_websites is done")
    #
    #
    # "credit"
    # # fetch dataset
    # default_of_credit_card_clients = fetch_ucirepo(id=350)
    #
    # # data (as pandas dataframes)
    # X = default_of_credit_card_clients.data.features
    # y = default_of_credit_card_clients.data.targets
    # X = X.to_numpy()
    # y = y.to_numpy()
    # num_bins = 8
    # binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    # X = binner.fit_transform(X)
    # # 使用 VarianceThreshold 移除常量特征
    # selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    # X_reduced = selector.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2028)
    # torch.save((X_train, X_test, y_train, y_test), './data/credit.pth')
    # data = np.hstack((X_train, y_train.reshape(-1, 1)))
    # thresholds = unique_values_by_column(torch.tensor(data))
    # save(thresholds, data, 'credit')
    # print(f"credit is done")
    # "skin"
    # # fetch dataset
    # skin_segmentation = fetch_ucirepo(id=229)
    #
    # # data (as pandas dataframes)
    # X = skin_segmentation.data.features
    # y = skin_segmentation.data.targets
    # X = X.to_numpy()
    # y = y.to_numpy()
    # num_bins = 8
    # binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    # X = binner.fit_transform(X)
    # # 使用 VarianceThreshold 移除常量特征
    # selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    # X_reduced = selector.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X_reduced, y == 0, test_size=0.2, random_state=2031)
    # torch.save((X_train, X_test, y_train, y_test), './data/skin.pth')
    # data = np.hstack((X_train, y_train.reshape(-1, 1)))
    # thresholds = unique_values_by_column(torch.tensor(data))
    # save(thresholds, data, 'skin')
    # print(f"skin is done")


    '''
    covertype
    '''

    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features
    y = covertype.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    y = np.where(y > 1, 1, y)
    num_bins = 16
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=1021)
    torch.save((X_train, X_test, y_train, y_test), './data/covertype.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds,data,'covertype')
    print(f"covertype is done")



if __name__ == '__main__':
    from pathlib import Path
    data_path = "./data"
    model_path = "./model"
    if not os.path.exists(data_path):
        Path(data_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_path):
        Path(model_path).mkdir(parents=True,exist_ok=True)
    print("正在离线加载....")
    config_dataset()
    # gen_key()
