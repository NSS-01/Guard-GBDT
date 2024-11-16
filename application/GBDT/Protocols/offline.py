import math
import torch
from NssMPC.crypto.aux_parameter import Parameter
class SelectKeyOffline(Parameter):
    def __init__(self,rs_Bshare=None,rs_Ashare=None, rg_Ashare=None,rg_Alshare=None,rg_msb_share=None,rsg_share=None,rs_msb_share=None):
        super(SelectKeyOffline, self).__init__()
        self.rs_Bshare = rs_Bshare
        self.rg_l_Ashare = rg_Ashare
        self.rs_l_Ashare = rs_Ashare
        self.rg_f_Ashare = rg_Alshare
        self.Msb_of_rg_Ashare= rg_msb_share
        self.rs_mul_msb_Ashare = rs_msb_share
        self.rs_mul_rg_Ashare = rsg_share
    @staticmethod
    def gen(nums, l_ring,ll_ring):
        rs = torch.randint(0,2,size=(nums,))
        rs_Bshare0 = torch.randint(0,2,size=(nums,))
        rs_Bshare1 = rs_Bshare0^rs
        rs_Ashare0 = torch.randint(0,ll_ring,size=(nums,))
        rs_Ashare1 = (rs-rs_Ashare0)%ll_ring
        rg = torch.randint(0,l_ring,size=(nums,))
        rg_Alshare0 = torch.randint(0,l_ring,size=(nums,))
        rg_Alshare1 = (rg-rg_Alshare0)%l_ring
        rg_Ashare0 =  torch.randint(0,ll_ring,size=(nums,))
        rg_Ashare1 = rg-rg_Ashare0

        msb_rg= (rg>>int(math.log(l_ring,2)-1))&1
        msb_share0 = torch.randint(0,ll_ring,size=(nums,))
        msb_share1 = (msb_rg-msb_share0)%ll_ring

        rgs_share0 = torch.randint(0,ll_ring,size=(nums,))
        rgs_share1 = (rs*rg-rgs_share0)%ll_ring

        rs_msb_share0 = torch.randint(0,ll_ring,size=(nums,))
        rs_msb_share1 = rs*msb_rg-rs_msb_share0
        return SelectKeyOffline(rs_Bshare0,rs_Ashare0,rg_Ashare0,rg_Alshare0, msb_share0,rgs_share0,rs_msb_share0), SelectKeyOffline(rs_Bshare1,rs_Ashare1,rg_Ashare1,rg_Alshare1, msb_share1,rgs_share1,rs_msb_share1)

if __name__ == '__main__':
    l_ring = 2**16
    ll_ring= 2**32
    SelectKeyOffline.gen_and_save(100000,None,l_ring,ll_ring)
    from NssMPC.secure_model.mpc_party.semi_honest import SemiHonestCS
    from NssMPC.secure_model.utils.param_provider.param_provider import ParamProvider
    server = SemiHonestCS(type='server')
    server.append_provider(ParamProvider(param_type=SelectKeyOffline))
    # print(server.party_id)
    # print(server.providers)
    # server.online()
    server.load_aux_params()
    key1 = server.get_param('SelectKeyOffline',100)

    print(key1.rs_Bshare)
    print(key1.rg_f_Ashare)
    # server.online()

