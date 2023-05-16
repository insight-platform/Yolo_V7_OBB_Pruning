import argparse
import logging
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from yolov7obb.models.yolo import Model
from yolov7obb.utils.torch_utils import select_device, is_parallel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg-deploy', type=str, default='', help='model.yaml path')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    opt = parser.parse_args()
    full_model_weights_path = Path(opt.weights)
    cfg_deploy = opt.cfg_deploy
    # device = select_device('')
    # state = torch.load(full_model_weights_path, map_location=device)
    # model = state["model"]
    # model.info()

    device = select_device('0', batch_size=1)
    # model trained by cfg/training/*.yaml
    ckpt = torch.load(full_model_weights_path, map_location=device)
    # reparameterized model in cfg/deploy/*.yaml
    # model = Model('/opt/app/tram/configs/ver2.0.0/yolov7_dep.yaml', ch=3, nc=1).to(device)
    model = Model(cfg_deploy, ch=3, nc=1).to(device)

    # with open('/opt/app/tram/configs/ver2.0.0/yolov7_dep.yaml') as f:
    with open(cfg_deploy) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if
                            k in model.state_dict() and not any(x in k for x in exclude) and v.shape ==
                            model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    # reparametrized YOLOR
    for i in range((model.nc + 185) * anchors):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i,
                                                                       ::].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i,
                                                                       ::].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i,
                                                                       ::].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(
        state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(
        state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(
        state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    torch.save(ckpt, full_model_weights_path.parent / f"{full_model_weights_path.stem}_reparam.pt")
