import pickle
import time

from NssMPC.crypto.primitives.boolean_secret_sharing import BooleanSecretSharing
from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
import torch
from NssMPC.common.ring.ring_tensor import RingTensor
from application.GBDT.Protocols.offline import SelectKeyOffline


# def HEP_Agg(s: BooleanSecretSharing, x: ArithmeticSecretSharing):
#     from heu import phe
#     from heu import numpy as hnp
#     phe_kit = phe.setup(phe.SchemaType.OU, 2048)
#     kit = hnp.HeKit(phe_kit)
#     encryptor = kit.encryptor()
#     decryptor = kit.decryptor()
#     scale=1
#     harr = kit.array(x.item.tensor.numpy(), phe.FloatEncoderParams(scale))
#
#     ct = encryptor.encrypt(harr)
#     party = x.party
#     party.send(ct)
#     ct_arr = party.receive()
#     ct_arr = pickle.loads(ct_arr)
#     pt_arr = decryptor.decrypt(ct_arr)
#
#
#     return pt_arr




def AggSiGBDT(s: BooleanSecretSharing, x: ArithmeticSecretSharing):
    st = time.time()
    party = s.party
    n = s.numel()
    key = party.get_param('SelectKeyOffline', n)
    rs_b = key.rs_Bshare
    rs_l = key.rs_l_Ashare
    rg_l = key.rg_l_Ashare
    x0 = x.ring_tensor.tensor+ rg_l
    s0 = s.ring_tensor.tensor^rs_b

    # st =  time.time()
    party.send((x0,s0))
    x1, s1 = party.receive()
    # end = time.time()
    #
    # communication_time = end - st
    hat_x = x0+x1
    hat_s = s0^s1
    res0 = hat_s *hat_s+hat_s*rg_l
    res1 = (1-2*hat_s)*hat_x*rs_l+(1-2*hat_s)*rs_l*rg_l
    res = res0+res1
    end = time.time()
    return end-st, ArithmeticSecretSharing(RingTensor(res),party=s.party).sum(dim=-1)



def AggOnline(s: BooleanSecretSharing, x: ArithmeticSecretSharing):
    st = time.time()
    party = s.party
    n = s.numel()
    key = party.get_param('SelectKeyOffline', n)

    rs_b = key.rs_Bshare
    rs_l = key.rs_l_Ashare
    rg_f = key.rg_f_Ashare
    rg_l = key.rg_l_Ashare
    rg_msb = key.Msb_of_rg_Ashare
    rs_mul_msb_rg= key.rs_mul_msb_Ashare
    rs_mul_rg = key.rs_mul_rg_Ashare
    x0 = (x.ring_tensor.tensor+ rg_f).to(torch.int16)
    s0 = (s.ring_tensor.tensor^rs_b).to(torch.bool)
    et = time.time()
    time0 =  et-st

    party.send((x0,s0))
    x1,s1 = party.receive()

    st =time.time()
    hat_x = (x0+x1+2**(16-2)).to(torch.int16)
    hat_s = s0^s1
    hat_x_msb =((hat_x >> (16 - 1)) & 1)
    res0 = hat_s *hat_s+hat_s*rg_msb*hat_x_msb*(2**16)+hat_s*(rg_l-(2**16-2))
    res1 = (1-2*hat_s)*hat_x*rs_l+(1-2*hat_s)*rs_mul_msb_rg*hat_x_msb*(2**16)+(1-2*hat_s)*(rs_mul_rg- rs_l*(2**16-2))
    res = res0+res1
    et = time.time()
    time1 = et-st
    return time1+time0, ArithmeticSecretSharing(RingTensor(res),party=s.party).sum(dim=-1)


class SigmoidTable:
    def __init__(self, n):
        self.n = n
        self.LookUpTabel = self.__getLookUpTable__(self.n)

    def __getLookUpTable__(self, n):
        min_value = -5.0  # 最小值
        max_value = 5.0  # 最大值
        num_points = n  # 查找表的精度，即采样点的数量
        x_values = torch.linspace(min_value, max_value, num_points+1)
        sigmoid_lookup_table = torch.sigmoid(x_values)
        return x_values, sigmoid_lookup_table


class LeafTabel:
    def __init__(self, n):
        self.n = n
        self.LookUpTabel = self.__getLookUpTable__(self.n)

    def __getLookUpTable__(self, n):
        min_value = -5.0  # 最小值
        max_value = 5.0  # 最大值
        num_points = n  # 查找表的精度，即采样点的数量
        x_values = torch.linspace(min_value, max_value, num_points+1)
        return x_values



def SecureLeafProtocol(G, H, n=10):
    delta = LeafTabel(n)
    values = delta.LookUpTabel
    key = torch.tensor(values)
    x = -G - RingTensor.convert_to_ring(key) * H
    b = x >=0
    vb =b.restore().convert_to_real_field()

    res = (b[1:] * RingTensor.convert_to_ring(values[1:] - values[:-1])).sum(dim=0) + RingTensor.convert_to_ring(
        values[0])
    return res


def SecureSigmoidProtocol(x, n=10):
    delta = SigmoidTable(n)
    keys, values = delta.LookUpTabel
    n, _ = x.shape
    key = keys.repeat(n, 1)
    b = (x - RingTensor.convert_to_ring(key)) >= 0

    res = (b[:, 1:] * RingTensor.convert_to_ring(values[1:] - values[:-1])).sum(dim=1) + RingTensor.convert_to_ring(
        values[0])
    return res.reshape(x.shape)


def setup_client(N):
    from NssMPC.secure_model.mpc_party import SemiHonestCS
    from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
    from NssMPC.crypto.primitives import ArithmeticSecretSharing
    from NssMPC.crypto.primitives.boolean_secret_sharing import BooleanSecretSharing
    client = SemiHonestCS(type='client')
    client.append_provider(ParamProvider(param_type=SelectKeyOffline))
    client.online()
    x = RingTensor.ones(size=(10 ** N, ),dtype='int')

    x = ArithmeticSecretSharing(x,party=client)
    b = RingTensor(torch.randint(0,2,size=(10**N,)))
    s = BooleanSecretSharing(b,party=client)
    st = time.time()
    # AggOnline(s, x)
    AggSiGBDT(s, x)
    # HEP_Agg(s,x)
    et = time.time()
    print(f"runtime:{et - st}")
    client.close()
def setup_server(N):
    from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
    from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
    server = SemiHonestCS(type='server')
    server.append_provider(ParamProvider(param_type=SelectKeyOffline))
    server.online()
    x = RingTensor.ones(size=(10 ** N, ), dtype='int')
    x = ArithmeticSecretSharing(x, party=server)
    b = RingTensor(torch.randint(0,2,size=(10**N,)))
    s = BooleanSecretSharing(b, party=server)
    # AggOnline(s, x)
    AggSiGBDT(s, x)
    # HEP_Agg(s,x)
    server.close()




if __name__ == "__main__":
    import threading
    N =2
    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client,args=(N,))
    server_thread = threading.Thread(target=setup_server,args=(N,))

    # Starting threads
    client_thread.start()
    server_thread.start()

    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()
    ''' 
  LAN (delay 0.1ms, 1Gbit/s)    WAN (delay 100ms, 400MB/S)
            Ours   SiGBDT          Ours   SiGBDT
    N=10^3  1.8 ms  0.9 ms         210 ms  210 ms
    N=10^4  1.9 ms  1.2 ms         211 ms  271 ms
    N=10^5  8.7 ms  29.8 ms        271 ms  341 ms
    N=10^6  69.5 ms 312 ms         457 ms  1861 ms
    N=10^7  1.6 s   3.9 s          3.73 s  17.24 s
    
    
    WAN (delay 100ms, 200MB/S)  WAN (delay 400ms, 400MB/S)
              Ours   SiGBDT          Ours   SiGBDT
    N=10^3  210 ms  211 ms          419 ms  421 ms
    N=10^4  211 ms  272 ms          421 ms  542 ms
    N=10^5  272 ms  345 ms          544 ms  632 ms
    N=10^6  501 ms  1946 ms         755 ms  2721 ms
    N=10^7  4.16 s  17.0 s          5.48 s  23.08 s
    
  
  
  SiGBDT:
                N=10^3    N=10^4   N=10^5    N=10^6   N= 10^7   
  LAN:
       runtime: 1.56      3.55    30.77     276.00    3707.35 
     comm_time: 1.07      2.98    28.23     258.88    3185.03 
     comp_time: 0.49      0.56    2.54      17.11     522.32 
  WAN:
     runtime:   85.90     113.04  298.25    1275.79    27606.01 
     comm_time: 85.44     112.49  295.48    1258.88    27055.94
     comp_time: 0.46      0.55    2.77      16.91      523.07
  
  Our:
             N=10^3    N=10^4   N=10^5    N=10^6  N= 10^7
  LAN:
    runtime:   1.67    2.85     10.08     70.17    1515.39
    com_time:  0.97    2.05     4.99      47.75    766.57
    comp_time: 0.70    0.80     5.09      22.42    648.82
  WAN:
    runtime:  84.89    87.20    117.89   568.02    5556.84
    com_time: 84.21    86.53    114.59   544.71    4905.36
    comp_time: 0.68    0.79     3.30     23.31     651.48
    '''


# 91.10 88.93 2.17