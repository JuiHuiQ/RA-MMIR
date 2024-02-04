import torch
from .RAMM_Point_bn import RAMM_Point
from .RA_MMIR import RA_MMIR

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}, frame_1 = None):
        super().__init__()
        self.RAMM_Point_bn = RAMM_Point(config.get('RAMM_Point', {}))
        self.RA_MMIR_bn = RA_MMIR(config.get('RA_MMIR', {}), frame_1)

    def forward(self, data):
        pred = {}

        # Extract RAMM_Point
        print(data)
        if 'keypoints0' not in data:
            pred0 = self.RAMM_Point_bn({'image': data['image0']}, 1)
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}

        if 'keypoints1' not in data:
            pred1 = self.RAMM_Point_bn({'image': data['image1']}, 0)
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching;
        pred = {**pred, **self.superglue(data)}

        return pred
