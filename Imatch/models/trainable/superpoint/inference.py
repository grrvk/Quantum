import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import kornia as K
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import rasterio
from datetime import datetime

from Imatch.dataset.dataset import ImageDataset


class SuperPointLightGlueInference:
    """
    SuperPoint + LightGlue inference
    """

    def __init__(self,
                 superpoint_weights: str,
                 max_keypoints: int = 1024):
        """
        Initialize inference class
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_keypoints = max_keypoints

        self.superpoint = SuperPoint(max_num_keypoints=max_keypoints).to(self.device)
        self.lightglue = LightGlue(features='superpoint').to(self.device)

        self.load_superpoint_weights(superpoint_weights)

        self.superpoint.eval()
        self.lightglue.eval()

    def load_superpoint_weights(self, weights_path: str):
        """
        Load trained SuperPoint weights
        """

        checkpoint = torch.load(weights_path, map_location=self.device)

        if 'superpoint_state_dict' in checkpoint:
            self.superpoint.load_state_dict(checkpoint['superpoint_state_dict'])
        else:
            self.superpoint.load_state_dict(checkpoint)

        print(f"Successfully loaded SuperPoint weights from {weights_path}")

    def preprocess_image_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image tensor for SuperPoint
        """

        image_gray = K.color.rgb_to_grayscale(image_tensor)
        image_gray = image_gray.to(self.device)
        return image_gray

    def load_image_from_path(self, image_path: str, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Load and preprocess image from file path (supports JP2 or standard image formats)
        """

        image_path = Path(image_path)

        if image_path.suffix.lower() == '.jp2':
            with rasterio.open(image_path) as src:
                image = src.read()
            
            if image.shape[0] > 3:
                image = image[:3, :, :]
            elif image.shape[0] == 2:
                image = np.concatenate([image, image[-1:, :, :]], axis=0)
            elif image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            
            image = image.transpose(1, 2, 0)
            
            if image.dtype != np.uint8:
                image_min = image.min()
                image_max = image.max()
                if image_max > image_min:
                    image = (image - image_min) / (image_max - image_min) * 255
                image = image.astype(np.uint8)
            
            if target_size is not None:
                image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if target_size is not None:
                image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

        image_tensor = K.utils.image_to_tensor(image)
        image_tensor = image_tensor.float().unsqueeze(dim=0).to(self.device) / 255.0
        return image_tensor

    def extract_features(self, image: torch.Tensor) -> Dict:
        """
        Extract SuperPoint features from image tensor
        """

        with torch.no_grad():
            feats = self.superpoint({"image": image})
            return rbd(feats)

    def match_features(self, feats0: Dict, feats1: Dict) -> Dict:
        """
        Match features using LightGlue
        """

        with torch.no_grad():
            matches = self.lightglue({
                "image0": {"image": torch.rand(1, 1, 512, 512).to(self.device)},
                "image1": {"image": torch.rand(1, 1, 512, 512).to(self.device)},
                "keypoints0": feats0["keypoints"].unsqueeze(0),
                "keypoints1": feats1["keypoints"].unsqueeze(0),
                "descriptors0": feats0["descriptors"].unsqueeze(0),
                "descriptors1": feats1["descriptors"].unsqueeze(0),
            })
            return rbd(matches)

    def process_image_pair_from_dataset(self,
                                        dataset: ImageDataset,
                                        idx0: int,
                                        idx1: int,
                                        confidence_threshold: float = 0.5) -> Dict:
        """
        Process a pair of images from the dataset using LightGlue
        """

        sample0 = dataset[idx0]
        sample1 = dataset[idx1]

        image0_tensor = sample0['image']
        image1_tensor = sample1['image']

        image0_gray = self.preprocess_image_tensor(image0_tensor)
        image1_gray = self.preprocess_image_tensor(image1_tensor)

        feats0 = self.extract_features(image0_gray)
        feats1 = self.extract_features(image1_gray)

        kps0 = feats0['keypoints']
        descs0 = feats0['descriptors']
        kps1 = feats1['keypoints']
        descs1 = feats1['descriptors']

        descs0 = torch.nn.functional.normalize(descs0, p=2, dim=1)
        descs1 = torch.nn.functional.normalize(descs1, p=2, dim=1)

        image0_data = {
            "keypoints": kps0.unsqueeze(0),
            "descriptors": descs0.unsqueeze(0),
            "image_size": torch.tensor([image0_gray.shape[-1], image0_gray.shape[-2]]).unsqueeze(0).to(self.device),
        }
        image1_data = {
            "keypoints": kps1.unsqueeze(0),
            "descriptors": descs1.unsqueeze(0),
            "image_size": torch.tensor([image1_gray.shape[-1], image1_gray.shape[-2]]).unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            out = self.lightglue({"image0": image0_data, "image1": image1_data})

        matches = out["matches"][0]
        scores = out["scores"][0] if "scores" in out else torch.ones(matches.shape[0], device=self.device)

        if len(matches) > 0:
            mask = scores > confidence_threshold
            filtered_matches = matches[mask]
            filtered_scores = scores[mask]

            matched_kps0 = kps0[filtered_matches[:, 0]].cpu().numpy()
            matched_kps1 = kps1[filtered_matches[:, 1]].cpu().numpy()
            confidence_scores = filtered_scores.cpu().numpy()
        else:
            matched_kps0 = np.array([]).reshape(0, 2)
            matched_kps1 = np.array([]).reshape(0, 2)
            confidence_scores = np.array([])
            print("No matches found after filtering")

        inliers = np.zeros(len(matched_kps0), dtype=bool)
        if len(matched_kps0) > 7:
            try:
                F, inliers_mask = cv2.findFundamentalMat(
                    matched_kps0, matched_kps1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
                )
                if inliers_mask is not None:
                    inliers = inliers_mask.ravel() > 0
            except Exception as e:
                print(f"Fundamental matrix estimation failed: {e}")
        else:
            print(f"Not enough matches ({len(matched_kps0)}) for fundamental matrix estimation")

        return {
            'feats0': feats0,
            'feats1': feats1,
            'matches': out,
            'image0_np': sample0['image_np'],
            'image1_np': sample1['image_np'],
            'filename0': sample0['filename'],
            'filename1': sample1['filename'],
            'image0_tensor': image0_tensor,
            'image1_tensor': image1_tensor,
            'filtered_keypoints0': matched_kps0,
            'filtered_keypoints1': matched_kps1,
            'confidence': confidence_scores,
            'inliers': inliers,
            'num_tentative_matches': len(matches),
            'num_filtered_matches': len(matched_kps0),
            'num_inliers': inliers.sum()
        }

    def process_single_image_from_dataset(self,
                                          dataset: ImageDataset,
                                          idx: int) -> Dict:
        """
        Process single image from dataset
        """

        sample = dataset[idx]
        image_tensor = sample['image']
        image_gray = self.preprocess_image_tensor(image_tensor)

        feats = self.extract_features(image_gray)

        return {
            'image_np': sample['image_np'],
            'feats': feats,
            'keypoints': feats['keypoints'].cpu().numpy(),
            'scores': feats['scores'].cpu().numpy() if 'scores' in feats else None,
            'num_keypoints': len(feats['keypoints']),
            'filename': sample['filename']
        }

    def process_image_pair(self,
                          image0_path: str,
                          image1_path: str,
                          target_size: Tuple[int, int] = (512, 512),
                          confidence_threshold: float = 0.5) -> Dict:
        """
        Process a pair of images from file paths
        """

        image0_tensor = self.load_image_from_path(image0_path, target_size)
        image1_tensor = self.load_image_from_path(image1_path, target_size)

        image0_np = K.tensor_to_image(image0_tensor)
        image1_np = K.tensor_to_image(image1_tensor)

        image0_gray = self.preprocess_image_tensor(image0_tensor)
        image1_gray = self.preprocess_image_tensor(image1_tensor)

        feats0 = self.extract_features(image0_gray)
        feats1 = self.extract_features(image1_gray)
        
        kps0 = feats0['keypoints']
        descs0 = feats0['descriptors']
        kps1 = feats1['keypoints']
        descs1 = feats1['descriptors']

        descs0 = torch.nn.functional.normalize(descs0, p=2, dim=1)
        descs1 = torch.nn.functional.normalize(descs1, p=2, dim=1)

        image0_data = {
            "keypoints": kps0.unsqueeze(0),
            "descriptors": descs0.unsqueeze(0),
            "image_size": torch.tensor([image0_gray.shape[-1], image0_gray.shape[-2]]).unsqueeze(0).to(self.device),
        }
        image1_data = {
            "keypoints": kps1.unsqueeze(0),
            "descriptors": descs1.unsqueeze(0),
            "image_size": torch.tensor([image1_gray.shape[-1], image1_gray.shape[-2]]).unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            out = self.lightglue({"image0": image0_data, "image1": image1_data})
        
        matches = out["matches"][0]
        scores = out["scores"][0] if "scores" in out else torch.ones(matches.shape[0], device=self.device)

        if len(matches) > 0:
            mask = scores > confidence_threshold
            filtered_matches = matches[mask]
            filtered_scores = scores[mask]
            
            matched_kps0 = kps0[filtered_matches[:, 0]].cpu().numpy()
            matched_kps1 = kps1[filtered_matches[:, 1]].cpu().numpy()
            confidence_scores = filtered_scores.cpu().numpy()
        else:
            matched_kps0 = np.array([]).reshape(0, 2)
            matched_kps1 = np.array([]).reshape(0, 2)
            confidence_scores = np.array([])
            print("No matches found after filtering")

        inliers = np.zeros(len(matched_kps0), dtype=bool)
        if len(matched_kps0) > 7:
            try:
                F, inliers_mask = cv2.findFundamentalMat(
                    matched_kps0, matched_kps1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
                )
                if inliers_mask is not None:
                    inliers = inliers_mask.ravel() > 0
            except Exception as e:
                print(f"Fundamental matrix estimation failed: {e}")
        else:
            print(f"Not enough matches ({len(matched_kps0)}) for fundamental matrix estimation")
        
        return {
            'feats0': feats0,
            'feats1': feats1,
            'matches': out,
            'image0_np': image0_np,
            'image1_np': image1_np,
            'filename0': Path(image0_path).name,
            'filename1': Path(image1_path).name,
            'image0_tensor': image0_tensor,
            'image1_tensor': image1_tensor,
            'filtered_keypoints0': matched_kps0,
            'filtered_keypoints1': matched_kps1,
            'confidence': confidence_scores,
            'inliers': inliers,
            'num_tentative_matches': len(matches),
            'num_filtered_matches': len(matched_kps0),
            'num_inliers': inliers.sum()
        }

    def draw_matches(self,
                     result: Dict,
                     output_path: Optional[str] = None,
                     show_keypoints: bool = False,
                     show_inliers: bool = True,
                     max_matches: int = 256,
                     inference_folder: str = 'models/inference_res') -> np.ndarray:
        """
        Draw matches between two images
        """

        image0 = result['image0_np']
        image1 = result['image1_np']
        keypoints0 = result['filtered_keypoints0']
        keypoints1 = result['filtered_keypoints1']
        inliers = result['inliers']
        confidence = result['confidence']

        if len(image0.shape) == 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
        elif image0.shape[2] == 1:
            image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)

        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        elif image1.shape[2] == 1:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

        if len(keypoints0) > max_matches:
            indices = np.random.choice(len(keypoints0), max_matches, replace=False)
            keypoints0 = keypoints0[indices]
            keypoints1 = keypoints1[indices]
            inliers = inliers[indices]
            confidence = confidence[indices] if len(confidence) > 0 else confidence

        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]
        
        # Add titles with filenames
        filename0 = result.get('filename0', 'Image 0')
        filename1 = result.get('filename1', 'Image 1')
        
        # Add space for title bar at the top
        title_height = 30
        vis_height = max(h0, h1) + title_height
        vis_image = np.zeros((vis_height, w0 + w1, 3), dtype=np.uint8)
        
        # Place images below the title
        vis_image[title_height:title_height+h0, :w0] = image0
        vis_image[title_height:title_height+h1, w0:w0+w1] = image1
        
        # Draw black rectangles for title backgrounds
        cv2.rectangle(vis_image, (0, 0), (w0, title_height), (0, 0, 0), -1)
        cv2.rectangle(vis_image, (w0, 0), (w0 + w1, title_height), (0, 0, 0), -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Calculate text size for centering
        (text_width0, text_height0), _ = cv2.getTextSize(filename0, font, font_scale, thickness)
        (text_width1, text_height1), _ = cv2.getTextSize(filename1, font, font_scale, thickness)
        
        # Center text in each half
        x0 = (w0 - text_width0) // 2
        y0 = title_height // 2 + text_height0 // 2
        
        x1 = w0 + (w1 - text_width1) // 2
        y1 = title_height // 2 + text_height1 // 2
        
        cv2.putText(vis_image, filename0, (x0, y0), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis_image, filename1, (x1, y1), font, font_scale, (255, 255, 255), thickness)

        if show_keypoints:
            feats0 = result['feats0']
            feats1 = result['feats1']
            all_kpts0 = feats0['keypoints'].cpu().numpy()
            all_kpts1 = feats1['keypoints'].cpu().numpy()

            for kp in all_kpts0:
                cv2.circle(vis_image, (int(kp[0]), int(kp[1]) + title_height), 2, (0, 255, 255), -1)
            for kp in all_kpts1:
                cv2.circle(vis_image, (int(kp[0]) + w0, int(kp[1]) + title_height), 2, (0, 255, 255), -1)

        for i, (kp0, kp1) in enumerate(zip(keypoints0, keypoints1)):
            pt0 = (int(kp0[0]), int(kp0[1]) + title_height)
            pt1 = (int(kp1[0]) + w0, int(kp1[1]) + title_height)

            if show_inliers and i < len(inliers):
                if inliers[i]:
                    color = (0, 255, 0)
                    line_width = 2
                else:
                    color = (0, 0, 255)
                    line_width = 1
            else:
                if len(confidence) > 0 and i < len(confidence):
                    conf = confidence[i]
                    color = (int(255 * (1 - conf)), 0, int(255 * conf))
                else:
                    color = (255, 0, 0)
                line_width = 1

            cv2.line(vis_image, pt0, pt1, color, line_width)
            cv2.circle(vis_image, pt0, 3, color, -1)
            cv2.circle(vis_image, pt1, 3, color, -1)

        if output_path:
            os.makedirs(inference_folder, exist_ok=True)
            image_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{output_path}'
            savepath = os.path.join(inference_folder, image_name)
            cv2.imwrite(savepath, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"Match visualization saved to: {savepath}")

        return vis_image


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Run SuperPoint + LightGlue inference on image pairs')

    current_dir = os.path.basename(os.getcwd())
    if not current_dir == "Imatch":
        raise Exception("Must be called from Imatch directory")

    parser.add_argument('--superpoint_weights', type=str, required=True,
                        help='Path to SuperPoint checkpoint file')

    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Directory containing images for inference (dataset mode)')
    parser.add_argument('--image_idx1', type=int, default=3,
                        help='First image index in dataset (default: 3)')
    parser.add_argument('--image_idx2', type=int, default=5,
                        help='Second image index in dataset (default: 5)')
    
    parser.add_argument('--image1_path', type=str, default=None,
                        help='Path to first image file (direct mode)')
    parser.add_argument('--image2_path', type=str, default=None,
                        help='Path to second image file (direct mode)')
    
    parser.add_argument('--max_keypoints', type=int, default=1024,
                        help='Maximum number of keypoints to extract (default: 1024)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[512, 512],
                        help='Target image size as height width (default: 512 512)')
    parser.add_argument('--output_path', type=str, default='image.png',
                        help='Path to save match visualization (default: image.png)')
    parser.add_argument('--show_keypoints', action='store_true', default=True,
                        help='Show keypoints in visualization (default: True)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on (default: cpu)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for filtering matches (default: 0.5)')

    args = parser.parse_args()

    pipeline = SuperPointLightGlueInference(
        superpoint_weights=args.superpoint_weights,
        max_keypoints=args.max_keypoints
    )

    if args.image1_path and args.image2_path:
        print(f"Processing images:")
        print(f"  Image 1: {args.image1_path}")
        print(f"  Image 2: {args.image2_path}")
        
        result = pipeline.process_image_pair(
            args.image1_path, 
            args.image2_path,
            target_size=tuple(args.target_size),
            confidence_threshold=args.confidence_threshold
        )
    elif args.dataset_dir:
        dataset = ImageDataset(
            root_dir=args.dataset_dir,
            target_size=tuple(args.target_size),
            device=args.device
        )
        
        print(f"Dataset loaded with {len(dataset)} images")
        print(f"Processing images at indices {args.image_idx1} and {args.image_idx2}")
        
        result = pipeline.process_image_pair_from_dataset(dataset, args.image_idx1, args.image_idx2)
    else:
        raise ValueError("Either provide --dataset_dir OR --image1_path and --image2_path")

    feats0, feats1 = result['feats0'], result['feats1']
    matches = result['matches']

    print(f"\nResults:")
    print(f"Image 0: {result['filename0']}")
    print(f"Image 1: {result['filename1']}")
    print(f"Keypoints in image 0: {len(feats0['keypoints'])}")
    print(f"Keypoints in image 1: {len(feats1['keypoints'])}")
    print(f"Number of matches: {len(matches['matches'])}")

    vis_image = pipeline.draw_matches(
        result,
        output_path=args.output_path,
        show_keypoints=args.show_keypoints
    )