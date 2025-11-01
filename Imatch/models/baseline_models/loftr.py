import torch
import cv2

import numpy as np
import kornia as K
from kornia_moons.viz import draw_LAF_matches


class LoFTR:
    def __init__(self, device=None):
        self.device = device if device else K.utils.get_cuda_device_if_available()
        self.matcher = K.feature.LoFTR(pretrained='outdoor').eval().to(self.device)

    def __call__(self, img0, img1, confidence_threshold: float = 0.9):
        img0 = img0.cpu().float()
        img1 = img1.cpu().float()

        input_dict = {
            "image0": K.color.rgb_to_grayscale(img0),
            "image1": K.color.rgb_to_grayscale(img1)
        }

        with torch.no_grad():
            correspondences = self.matcher(input_dict)

        mask = correspondences['confidence'] > confidence_threshold
        keypoints0 = correspondences['keypoints0'][mask].numpy()
        keypoints1 = correspondences['keypoints1'][mask].numpy()
        confidence = correspondences['confidence'][mask].numpy()

        if len(keypoints0) > 8:
            Fm, inliers = cv2.findFundamentalMat(
                keypoints0, keypoints1,
                cv2.FM_RANSAC, 0.5, 0.99, maxIters=1000  # Reduced iterations
            )
            inliers = inliers > 0 if inliers is not None else np.zeros(len(keypoints0), dtype=bool)
        else:
            inliers = np.zeros(len(keypoints0), dtype=bool)

        return {
            'image0': img0,
            'image1': img1,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'confidence': confidence,
            'inliers': inliers,
        }

    @staticmethod
    def draw(matched_dict):
        return draw_LAF_matches(
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(matched_dict['keypoints0']).view(1, -1, 2),
                torch.ones(matched_dict['keypoints0'].shape[0]).view(1, -1, 1, 1),
                torch.ones(matched_dict['keypoints0'].shape[0]).view(1, -1, 1),
            ),
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(matched_dict['keypoints1']).view(1, -1, 2),
                torch.ones(matched_dict['keypoints1'].shape[0]).view(1, -1, 1, 1),
                torch.ones(matched_dict['keypoints1'].shape[0]).view(1, -1, 1),
            ),
            torch.arange(matched_dict['keypoints0'].shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(matched_dict['image0']),
            K.tensor_to_image(matched_dict['image1']),
            matched_dict['inliers'],
            draw_dict={"inlier_color": (0.2, 1, 0.2),
                       "tentative_color": None,
                       "feature_color": (0.2, 0.5, 1),
                       "vertical": False},
        )
