import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # Check for AMD ROCm/HIP support
            if torch.version.hip is not None:
                # AMD GPU detected via ROCm
                print('AMD ROCm GPU detected')
                if self.args.use_multi_gpu:
                    # For multi-GPU, set HIP_VISIBLE_DEVICES
                    os.environ["HIP_VISIBLE_DEVICES"] = self.args.devices
                    print(f'Use AMD Multi-GPU: {self.args.devices}')
                    device = torch.device('cuda:0')  # AMD uses cuda:0 interface through ROCm
                else:
                    # Single AMD GPU
                    device = torch.device('cuda:{}'.format(self.args.gpu))
                    print('Use AMD GPU: cuda:{}'.format(self.args.gpu))
            elif torch.cuda.is_available():
                # NVIDIA GPU detected
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use NVIDIA GPU: cuda:{}'.format(self.args.gpu))
            else:
                # No GPU detected, fallback to CPU
                device = torch.device('cpu')
                print('No GPU detected, using CPU')
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
