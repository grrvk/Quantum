import torch
import cv2
import numpy as np
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches


class LightGlue:
    def __init__(self, device=None, num_features: int = 2048):
        self.device = device if device else K.utils.get_cuda_device_if_available()
        self.num_features = num_features

        self.detector = KF.DISK.from_pretrained("depth").eval().to(self.device)
        self.matcher = KF.LightGlue("disk").eval().to(self.device)

    def __call__(self, img0, img1, confidence_threshold: float = 0.8):
        if len(img0.shape) == 3:
            img0 = img0.unsqueeze(0)
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)

        img0 = img0.cpu().float()
        img1 = img1.cpu().float()

        with torch.no_grad():
            inp = torch.cat([img0, img1], dim=0).to(self.device)

            features1, features2 = self.detector(inp, self.num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors

            image0 = {
                "keypoints": kps1[None],
                "descriptors": descs1[None],
                "image_size": torch.tensor(img0.shape[-2:][::-1]).view(1, 2).to(self.device),
            }
            image1 = {
                "keypoints": kps2[None],
                "descriptors": descs2[None],
                "image_size": torch.tensor(img1.shape[-2:][::-1]).view(1, 2).to(self.device),
            }

            out = self.matcher({"image0": image0, "image1": image1})
            idxs = out["matches"][0]
            scores = out["scores"][0] if "scores" in out else torch.ones(idxs.shape[0], device=self.device)

        if idxs.shape[0] > 0:
            mask = scores > confidence_threshold
            filtered_idxs = idxs[mask]
            filtered_confidence = scores[mask].cpu().numpy()

            keypoints0 = kps1[filtered_idxs[:, 0]].cpu().numpy()
            keypoints1 = kps2[filtered_idxs[:, 1]].cpu().numpy()
        else:
            keypoints0 = np.array([]).reshape(0, 2)
            keypoints1 = np.array([]).reshape(0, 2)
            filtered_confidence = np.array([])

        if len(keypoints0) > 8:
            Fm, inliers = cv2.findFundamentalMat(
                keypoints0, keypoints1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
            )
            inliers = inliers > 0 if inliers is not None else np.zeros(len(keypoints0), dtype=bool)
        else:
            inliers = np.zeros(len(keypoints0), dtype=bool)

        return {
            'image0': img0.squeeze(0),
            'image1': img1.squeeze(0),
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'confidence': filtered_confidence,
            'inliers': inliers,
            'num_tentative_matches': idxs.shape[0],
        }

    @staticmethod
    def draw(matched_dict):
        return draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(matched_dict['keypoints0']).view(1, -1, 2),
                torch.ones(matched_dict['keypoints0'].shape[0]).view(1, -1, 1, 1),
                torch.ones(matched_dict['keypoints0'].shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
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