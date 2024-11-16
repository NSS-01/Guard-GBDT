from heu import phe
from heu import numpy as hnp
import torch
import time
phe_kit = phe.setup(phe.SchemaType.OU, 2048)
kit = hnp.HeKit(phe_kit)
encryptor = kit.encryptor()
decryptor = kit.decryptor()
evaluator = kit.evaluator()
encoder = phe.IntegerEncoder(phe.SchemaType.OU)
a = torch.randn(size=(10**4, ))
print("plaintext:",a*a)
# Init from nested lists
scale=100000
harr = kit.array(a.numpy(), phe.FloatEncoderParams(scale))
# print(harr)
# print("plaintext", harr*harr)
# encrypt
st = time.time()
ct_arr = encryptor.encrypt(harr)
# decrypt
pt_arr = decryptor.decrypt(ct_arr)

c2 = evaluator.mul(ct_arr, pt_arr)
et= time.time()

print("dec--result:")
result = decryptor.decrypt(c2)
print(result.to_numpy()/(scale**2)) # [[30]]
print(et-et)
