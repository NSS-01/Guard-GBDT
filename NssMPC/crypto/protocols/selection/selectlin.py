class SelectLin(object):
    @staticmethod
    def eval(x_shift, w_shift, d_shift, key):
        shape = x_shift.shape
        i_shift = (w_shift * 2 + d_shift) % 4

        key.p.dtype = x_shift.dtype
        key.q.dtype = x_shift.dtype

        p = key.p.__class__.gather(key.p, -1, i_shift.tensor.unsqueeze(1)).squeeze() if len(key.p.shape) == 2 else \
            key.p[i_shift.tensor]
        q = key.p.__class__.gather(key.q, -1, i_shift.tensor.unsqueeze(1)).squeeze() if len(key.p.shape) == 2 else \
            key.q[i_shift.tensor]
        return (p * x_shift.flatten() + q).reshape(shape)

    @staticmethod
    def eval_with_comm(x_shift, w, d, key):
        from NssMPC.crypto.primitives import ArithmeticSecretSharing
        w_shift = ArithmeticSecretSharing(key.w, w.party) + w.flatten()
        d_shift = ArithmeticSecretSharing(key.d, d.party) + d.flatten()
        length = w_shift.numel()
        w_and_d = ArithmeticSecretSharing.cat([w_shift, d_shift], dim=0).restore()
        w_shift = w_and_d[:length]
        d_shift = w_and_d[length:]
        return SelectLin.eval(x_shift, w_shift, d_shift, key)
