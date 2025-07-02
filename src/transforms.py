import torch
import numpy as np
from torchvision import transforms
from helpers import convert_force_to_rotating_tool_frame, compute_smoothed_diff, compute_cutting_foce

class TransformCuttingForceToToolRotatingCoordinate(object):
    '''
    This class will help to transform the cutting force to the tool rotating coordinate system
    '''
    def __init__(self):
        self.rpm = 2000

    def __call__(self, sample):
        '''
        sample should be a list with the following elements: 
        [time_step, channel_1, channel_2, channel_3, channel_4, channel_5, channel_6, channel_7, channel_8, ss, ap, ft, ad]
        '''
        time_step, channels = sample["input"][0], sample["input"][1:8]
        vc, ap, ft, ad = sample["input"][8], sample["input"][9], sample["input"][10], sample["input"][11]

        Fx, Fy, Fz = compute_cutting_foce(channels)
        Ft, Fn, Fa = convert_force_to_rotating_tool_frame(Fx, Fy, Fz, time_step, self.rpm)

        return {
            "input": np.array([Ft, Fn, Fa, vc, ap, ft, ad], dtype=np.float32),
            "output": None
        }

class CreateAugmentedLib(object):
    def __call__(self, sample):
        vc, ap, ft, ad = sample["input"][3], sample["input"][4], sample["input"][5], sample["input"][6]
        ln_vc, ln_ap, ln_ft, ln_ad = np.log(vc), np.log(ap), np.log(ft), np.log(ad)

        return {
            "input": np.array([1, ln_vc, ln_ap, ln_ft, ln_ad], dtype=np.float32),
            "output": sample["output"]
        }
    
class TransformTarget(object):
    '''
    This class will help to get the output by computing differential from the Ft, Fr, Fa
    '''
    def __call__(self, sample):
        Ft, Fn, Fa = sample["input"][:3]

        dFt, dFn, dFa = compute_smoothed_diff(Ft, Fn, Fa)

        return {
            "input": sample["input"],
            "output": np.array([dFt, dFn, dFa], dtype=np.float32)
        }
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        input, output = sample["input"], sample["output"]

        return {
            "input": torch.from_numpy(input),
            "output": torch.from_numpy(output)
        }
    
def get_transforms():
    return transforms.Compose([
        TransformCuttingForceToToolRotatingCoordinate(),
        TransformTarget(),
        CreateAugmentedLib(),
        ToTensor()
    ])