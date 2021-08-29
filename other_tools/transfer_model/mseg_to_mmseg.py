import torch

def mseg2mmseg(mseg_model):
    ckpt = mseg_model['state_dict']
    transfered_ckpt = {}
    for k, v in ckpt.items():
        if 'module.segmodel.encoder.' in k:
            new_k = k.replace('module.segmodel.encoder.', 'backbone.')
            transfered_ckpt[new_k] = v
        elif 'module.segmodel.head.' in k:
            new_k = k.replace('module.segmodel.head.', 'decode_head.')
            transfered_ckpt[new_k] = v
        else:
            continue
    return transfered_ckpt

if __name__ == '__main__':
    dir = '/home/ubuntu/Projects/SIW/SegFormer/pretrained/'
    mseg_model = torch.load(dir + 'train_epoch_final.pth', map_location='cpu')
    siw_pretrain = mseg2mmseg(mseg_model)

    mmseg_model = torch.load(dir + 'segformer.b5.640x640.ade.160k.pth', map_location='cpu')
    torch.save(siw_pretrain, dir + 'mitb5_siw_pretrain.pth')
    print(list(siw_pretrain.keys())[-20:])
    print(list(mmseg_model['state_dict'].keys())[-20:])