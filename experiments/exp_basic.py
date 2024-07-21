import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, S_Mamba, \
    Flashformer_M, Flowformer_M, Autoformer, Autoformer_M, Transformer_M, \
    Informer_M, Reformer_M, \
    S_Mamba_G_add_G,\
    S_Mamba_G_add_L,\
    S_Mamba_G_add_GL,\
    S_Mamba_G_add_L_shared,\
    S_Mamba_G_add_GL_shared,\
    S_Mamba_ens_GL_add_GL,\
    S_Mamba_with_mamba_embedder,\
    S_Mamba_G_add_L_v2,\
    S_Mamba_G_add_L_v3,\
    S_Mamba_G_add_L_v4,\
    S_Mamba_gate,\
    S_Mamba_gate2,\
    S_Mamba_gate3,\
    S_Mamba_gate4,\
    S_Mamba_gate5,\
    S_Mamba_gate6,\
    S_Mamba_gate_HH,\
    S_Mamba_gate_Hn,\
    S_Mamba_gate_nH,\
    S_Mamba_reg,\
    S_Mamba_reg2,\
    S_Mamba_reg3,\
    S_Mamba_reg3_wo_conv,\
    S_Mamba_reg3_wo_conv_wo_TD1,\
    S_Mamba_reg3_wo_conv_wo_TD2,\
    S_Mamba_reg4_wo_conv,\
    S_Mamba_reg4_wo_conv_wo_TD1,\
    S_Mamba_reg4_wo_conv_wo_TD2,\
    S_Mamba_reg4,\
    S_Mamba_reg_k1,\
    S_Mamba_reg2_k1,\
    S_Mamba_reg_mask,\
    S_Mamba_reg2_mask

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S_Mamba_G_add_G': S_Mamba_G_add_G,
            'S_Mamba_G_add_L': S_Mamba_G_add_L,
            'S_Mamba_G_add_GL': S_Mamba_G_add_GL,
            'S_Mamba_G_add_L_shared': S_Mamba_G_add_L_shared,
            'S_Mamba_G_add_GL_shared': S_Mamba_G_add_GL_shared,
            'S_Mamba_ens_GL_add_GL':S_Mamba_ens_GL_add_GL,
            'S_Mamba_with_mamba_embedder':S_Mamba_with_mamba_embedder,
            'S_Mamba_G_add_L_v2':S_Mamba_G_add_L_v2,
            'S_Mamba_G_add_L_v3':S_Mamba_G_add_L_v3,
            'S_Mamba_G_add_L_v4':S_Mamba_G_add_L_v4,
            'S_Mamba_gate':S_Mamba_gate,
            'S_Mamba_gate2':S_Mamba_gate2,
            'S_Mamba_gate3':S_Mamba_gate3,
            'S_Mamba_gate4':S_Mamba_gate4,
            'S_Mamba_gate5':S_Mamba_gate5,
            'S_Mamba_gate6':S_Mamba_gate6,
            'S_Mamba_gate_HH':S_Mamba_gate_HH,
            'S_Mamba_gate_Hn':S_Mamba_gate_Hn,
            'S_Mamba_gate_nH':S_Mamba_gate_nH,
            'S_Mamba_reg':S_Mamba_reg,
            'S_Mamba_reg2':S_Mamba_reg2,
            'S_Mamba_reg3':S_Mamba_reg3,
            'S_Mamba_reg3_wo_conv':S_Mamba_reg3_wo_conv,
            'S_Mamba_reg3_wo_conv_wo_TD1':S_Mamba_reg3_wo_conv_wo_TD1,
            'S_Mamba_reg3_wo_conv_wo_TD2':S_Mamba_reg3_wo_conv_wo_TD2,
            'S_Mamba_reg4':S_Mamba_reg4,
            'S_Mamba_reg4_wo_conv':S_Mamba_reg4_wo_conv,
            'S_Mamba_reg4_wo_conv_wo_TD1':S_Mamba_reg4_wo_conv_wo_TD1,
            'S_Mamba_reg4_wo_conv_wo_TD2':S_Mamba_reg4_wo_conv_wo_TD2,
            'S_Mamba_reg_mask':S_Mamba_reg_mask,
            'S_Mamba_reg2_mask':S_Mamba_reg2_mask,
            'S_Mamba_reg_k1':S_Mamba_reg_k1,
            'S_Mamba_reg2_k1':S_Mamba_reg2_k1,
            #########################################
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,

            'Transformer': Transformer,
            'Transformer_M': Transformer_M,

            'Informer': Informer,
            'Informer_M': Informer_M,

            'Reformer': Reformer,
            'Reformer_M': Reformer_M,

            'Flowformer': Flowformer,
            'Flashformer_M': Flashformer_M,

            'Flashformer': Flashformer,
            'Flowformer_M': Flowformer_M,

            'Autoformer': Autoformer,
            'Autoformer_M': Autoformer_M,

            'S_Mamba': S_Mamba,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
