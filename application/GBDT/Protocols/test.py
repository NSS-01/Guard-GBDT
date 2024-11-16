import time

from NssMPC import ArithmeticSecretSharing, RingTensor
from application.GBDT.Protocols.online import SecureLeafProtocol,SecureSigmoidProtocol

def HEP_XGB_Sigmoid_protocol(x:ArithmeticSecretSharing):
    '''
    Args:
        x:

    Returns:
        0.5+0.5*x/(1+|x|)
    '''
    b= (x>=0)
    abs_x =2*b*x-x
    return (RingTensor.convert_to_ring(0.5)*x/abs_x)+RingTensor.convert_to_ring(0.5)

def SiGBDT_Sigmoid(x:ArithmeticSecretSharing):
    e = ArithmeticSecretSharing.exp(x)
    return e/(e+RingTensor.convert_to_ring(1.0))


def setup_client(N):
    from NssMPC.secure_model.mpc_party import SemiHonestCS
    from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
    from NssMPC.crypto.primitives import ArithmeticSecretSharing
    from NssMPC.crypto.primitives.boolean_secret_sharing import BooleanSecretSharing
    client = SemiHonestCS(type='client')
    client.set_multiplication_provider()
    client.set_comparison_provider()
    client.set_nonlinear_operation_provider()
    client.online()
    # x = torch.randint(0,2**32,size=(10**N,1))
    x = RingTensor.ones(size=(10**N,1), dtype='float')
    x = ArithmeticSecretSharing(x,party=client)
    # st = time.time()
    # SiGBDT_Sigmoid(x)
    # et = time.time()
    # print(f"SiGBDT runtime:{et - st:.2f}")
    # st = time.time()
    # SecureSigmoidProtocol(x, n=10)
    # et = time.time()
    # print(f"Our runtime:{et - st:.2f}")
    # st = time.time()
    # HEP_XGB_Sigmoid_protocol(x)
    # et = time.time()
    # print(f"HEP-XGB runtime:{et - st:.2f}")
    # x * x * x + x * x * x
    # st = time.time()
    # x/x+x/x+x/x
    # et = time.time()
    # SecureLeafProtocol(x,x)
    client.close()

def setup_server(N):
    from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
    server = SemiHonestCS(type='server')
    server.set_multiplication_provider()
    server.set_comparison_provider()
    server.set_nonlinear_operation_provider()
    server.online()
    x = RingTensor.ones(size=(10**N,1), dtype='float')
    x = ArithmeticSecretSharing(x, party=server)
    # st = time.time()
    # SiGBDT_Sigmoid(x)
    # et = time.time()
    # print(f"SiGBDT runtime:{et - st:.2f}")
    # st=time.time()
    # SecureSigmoidProtocol(x, n=10)
    # et = time.time()
    # print(f"Our runtime:{et - st:.2f}")
    # st = time.time()
    # HEP_XGB_Sigmoid_protocol(x)
    # et = time.time()
    # print(f"HEP-XGB runtime:{et - st:.2f}")
    # SecureLeafProtocol(x,x)
    # st = time.time()
    # x/x+x/x+x/x
    # et = time.time()
    # print(f"leaf weight:{et-st:.2f}")
    # x *x*x+x*x*x
    # x*x

    # print(f"runtime:{et-st:.2f}")
    server.close()




if __name__ == "__main__":
    import threading
    import NssMPC.config.configs as cfg
    # cfg.GE_TYPE = "MSB"
    N =5
    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client,args=(N,))
    server_thread = threading.Thread(target=setup_server,args=(N,))

    # Starting threads
    client_thread.start()
    server_thread.start()

    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()
