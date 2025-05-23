from application.neural_network.party import HonestMajorityNeuralNetWork3PC

# 测试恶意乘法


import NssMPC.application.neural_network as nn
from data.AlexNet.Alexnet import AlexNet

# 测试恶意乘法
if __name__ == '__main__':
    Party = HonestMajorityNeuralNetWork3PC(1)
    Party.online()
    net = AlexNet()

    print("接收权重")
    local_param = Party.receive(0)

    print("预处理一些东西")
    num = Party.dummy_model()
    net = nn.utils.load_model(net, local_param)
    print("接收输入")

    pass
