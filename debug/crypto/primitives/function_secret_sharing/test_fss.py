"""
function_secret_sharing 测试
"""
import unittest

from NssMPC.crypto.primitives.function_secret_sharing.dpf import prefix_parity_query

from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.function_secret_sharing import *

num_of_keys = 10**5  # We need a few keys for a few function values, but of course we can generate many keys in advance.

# generate keys in offline phase
# set alpha and beta
alpha = RingTensor(5)
beta = RingTensor(1)
down_bound = RingTensor(3)
upper_bound = RingTensor(7)
# online phase
# generate some values what we need to evaluate
x = RingTensor([1]*10**5)


class Test(unittest.TestCase):
    # distributed comparison function
    def test_dcf(self):
        key0, key1 = DCF.gen(num_of_keys=num_of_keys, alpha=alpha, beta=beta)

        # Party 0:
        res_0 = DCF.eval(x=x, keys=key0, party_id=0)
        # Party 1:
        res_1 = DCF.eval(x=x, keys=key1, party_id=1)

        # restore result
        res = res_0 + res_1
        print(res)
        # assert (res == RingTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], device=res.device)).all()

    # distributed point function
    def test_dpf(self):
        key0, key1 = DPF.gen(num_of_keys=num_of_keys, alpha=alpha, beta=beta)

        # Party 0:
        res_0 = DPF.eval(x=x, keys=key0, party_id=0)
        # Party 1:
        res_1 = DPF.eval(x=x, keys=key1, party_id=1)

        # restore result
        res = res_0 + res_1
        print(res)
        # assert (res == RingTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], device=res.device)).all()

    # prefix parity query
    def test_ppq(self):
        key0, key1 = DPF.gen(num_of_keys=1, alpha=alpha, beta=beta)

        # Party 0:
        res_0 = prefix_parity_query(RingTensor(5), key0, party_id=0)
        # Party 1:
        res_1 = prefix_parity_query(RingTensor(5), key1, party_id=1)

        # restore result
        res = res_0 ^ res_1
        res = res
        print(res)
        assert (res == RingTensor([0], device=res.device)).all()