import os
import random
import re

from NssMPC.crypto.aux_parameter._parameter_base import Parameter
from NssMPC.common.ring import RingTensor
from NssMPC.config import param_path
from NssMPC.crypto.primitives.homomorphic_encryption.paillier import Paillier


class BooleanTriples(Parameter):
    def __init__(self, a=None, b=None, c=None):
        self.a = a
        self.b = b
        self.c = c
        self.size = 0

    def __iter__(self):
        return iter((self.a, self.b, self.c))

    @staticmethod
    def gen(num_of_triples, num_of_party=2, type_of_generation='TTP', party=None):
        """
        Args:
            num_of_triples: number of parameters
            num_of_party: number of parties
            type_of_generation: generation type,
                                TTP: generated by trusted third party, HE: generated by homomorphic encryption
            party: party if HE
        """
        if type_of_generation == 'HE':
            return gen_msb_triples_by_homomorphic_encryption(num_of_triples, party)
        elif type_of_generation == 'TTP':
            return gen_msb_triples_by_ttp(num_of_triples, num_of_party)

    @classmethod
    def gen_and_save(cls, num, saved_name = None, num_of_party = 2,  saved_path = None, type_of_generation='TTP', party=None):
        """
        Args:
            num_of_triples: number of parameters
            num_of_party: number of parties
            type_of_generation: generation type,
                                TTP: generated by trusted third party, HE: generated by homomorphic encryption
            party: party if HE
        """
        assert type_of_generation in ['HE', 'TTP']
        triples = cls.gen(num, num_of_party, type_of_generation, party)
        if saved_path is None:
            file_path = f"{param_path}BooleanTriples/"
        else:
            file_path = saved_path
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if saved_name is None:
            saved_name = 'BooleanTriples'


        if type_of_generation == 'TTP':
            for i in range(num_of_party):
                file_name = f"{saved_name}_{i}.pth"
                triples[i].save(file_path=file_path, name=file_name)

        # todo :未作HE适配
        else:
            file_path = f"{param_path}BooleanTriples/{num_of_party}party/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            file_names = os.listdir(file_path)
            max_ptr = 0
            for fname in file_names:
                match = re.search(rf"BooleanTriples_{party.party_id}+_(\d+)\.pth", fname)
                if match:
                    max_ptr = max(max_ptr, int(match.group(1)))

            file_name = f"BooleanTriples_{party.party_id}_{max_ptr + 1}.pth"
            triples.save(file_path + file_name)



def gen_msb_triples_by_ttp(bit_len, num_of_party=2):
    """
    Generate the multiplication Beaver triple by trusted third party

    Args:
       bit_len (int): the length of the binary string
       num_of_party (int): the number of parties
    Returns:
        the list of triples
    """
    a = RingTensor.random([bit_len], down_bound=0, upper_bound=2, device='cpu')
    b = RingTensor.random([bit_len], down_bound=0, upper_bound=2, device='cpu')
    c = a & b
    from NssMPC.crypto.primitives.boolean_secret_sharing.boolean_secret_sharing import BooleanSecretSharing
    a_list = BooleanSecretSharing.share(a, num_of_party)
    b_list = BooleanSecretSharing.share(b,  num_of_party)
    c_list = BooleanSecretSharing.share(c, num_of_party)

    triples = []
    for i in range(num_of_party):
        triples.append(BooleanTriples())
        triples[i].a = a_list[i].to('cpu')
        triples[i].b = b_list[i].to('cpu')
        triples[i].c = c_list[i].to('cpu')
        triples[i].size = bit_len

    return triples


def gen_msb_triples_by_homomorphic_encryption(bit_len, party):
    """
    Generate the multiplication Beaver triple by homomorphic encryption

    Args:
        bit_len (int): the length of the binary string
        party
    """

    a = [random.randint(0, 2) for _ in range(bit_len)]
    b = [random.randint(0, 2) for _ in range(bit_len)]
    c = []

    if party.party_id == 0:
        paillier = Paillier()
        paillier.gen_keys()

        encrypted_a = paillier.encrypt(a)
        encrypted_b = paillier.encrypt(b)
        party.send([encrypted_a, encrypted_b, paillier.public_key])
        d = party.receive()
        decrypted_d = paillier.decrypt(d)
        c = [decrypted_d[i] + a[i] * b[i] for i in range(bit_len)]

    elif party.party_id == 1:
        r = [random.randint(0, 2) for _ in range(bit_len)]
        c = [a[i] * b[i] - r[i] for i in range(bit_len)]

        messages = party.receive()

        encrypted_r = Paillier.encrypt_with_key(r, messages[2])
        d = [messages[0][i] ** b[i] * messages[1][i] ** a[i] * encrypted_r[i] for i in range(bit_len)]
        party.send(d)

    msb_triples = BooleanTriples(RingTensor(a).to('cpu'), RingTensor(b).to('cpu'),
                                 RingTensor(c).to('cpu'))
    return msb_triples
