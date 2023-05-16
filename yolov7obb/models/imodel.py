import logging
from copy import deepcopy
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch import device

from yolov7obb.utils.torch_utils import model_info


class IModel(nn.Module):
    def __init__(self, channel):
        super(IModel, self).__init__()
        self.device = torch.device('cpu')
        self.channel = channel
        self.logger = logging.getLogger(self.__class__.__name__)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, device]] = None):
        if device is None:
            self.device = torch.device('cuda:0')
        else:
            self.device = device
        return super().cuda(device)

    def cpu(self):
        self.device = torch.device('cpu')
        return super().cpu()

    def info(self, img_size: Tuple[int, int] = (640, 640)) -> Tuple[int, int, int, Optional[float]]:
        """Returns model information for a given input image sizes.
        :param img_size: image size (width, height)
        :return: tuple or list of tuple (Layers number, parameters number, gradient number, computational complexity)
        """
        # Model information.

        parameters_number = sum(x.numel() for x in self.parameters())  # number parameters
        gradient_numbers = sum(x.numel() for x in self.parameters() if x.requires_grad)  # number gradients
        try:  # FLOPS
            from thop import profile
            # stride = max(int(self.stride.max()), 32) if hasattr(self, 'stride') else 32
            img = torch.zeros((1, self.channel, img_size[0], img_size[1]),
                              device=next(self.parameters()).device)  # input
            tmp_model = deepcopy(self)
            flops = profile(tmp_model, inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        except (ImportError, Exception):
            flops = None
            self.logger.warning("An error occurred during the calculation of computational complexity:", exc_info=True)

        return len(list(self.modules())), parameters_number, gradient_numbers, flops

