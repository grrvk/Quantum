import torch
import cv2
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import rasterio

from Imatch.models.baseline_models.loftr import LoFTR
from Imatch.models.baseline_models.lightglue import LightGlue
from Imatch.dataset.dataset import ImageDataset
import kornia as K


def load_image_from_path(image_path: str, target_size=(512, 512)):
    """
    Load image from path
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
        
        if image.dtype != 'uint8':
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype('uint8')
            else:
                image = image.astype('uint8')
        
        if target_size:
            image = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_AREA)
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if target_size:
            image = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_AREA)
    
    return K.utils.image_to_tensor(image).float().unsqueeze(0) / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline model inference')
    
    parser.add_argument('--model', type=str, choices=['loftr', 'lightglue'], required=True,
                        help='Model to use: loftr or lightglue')
    
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Directory containing images (dataset mode)')
    parser.add_argument('--image_idx0', type=int, default=3,
                        help='First image index')
    parser.add_argument('--image_idx1', type=int, default=5,
                        help='Second image index')
    
    parser.add_argument('--image0_path', type=str, default=None,
                        help='Path to first image')
    parser.add_argument('--image1_path', type=str, default=None,
                        help='Path to second image')
    
    parser.add_argument('--output_path', type=str, default='models/inference_res/baseline_matches.png',
                        help='Output path for visualization')
    parser.add_argument('--confidence_threshold', type=float, default=None,
                        help='Confidence threshold (default: 0.9 for LoFTR, 0.8 for LightGlue)')

    args = parser.parse_args()

    if args.model == 'loftr':
        matcher = LoFTR()
        confidence_threshold = args.confidence_threshold if args.confidence_threshold else 0.9
    else:
        matcher = LightGlue()
        confidence_threshold = args.confidence_threshold if args.confidence_threshold else 0.8

    if args.image0_path and args.image1_path:
        print(f"Loading images: {args.image0_path} and {args.image1_path}")
        img0 = load_image_from_path(args.image0_path)
        img1 = load_image_from_path(args.image1_path)
        filename0 = Path(args.image0_path).name
        filename1 = Path(args.image1_path).name
    elif args.dataset_dir:
        dataset = ImageDataset(root_dir=args.dataset_dir, target_size=(512, 512))
        print(f"Dataset loaded with {len(dataset)} images")
        print(f"Processing images at indices {args.image_idx0} and {args.image_idx1}")
        
        sample0 = dataset[args.image_idx0]
        sample1 = dataset[args.image_idx1]
        
        img0 = sample0['image']
        img1 = sample1['image']
        filename0 = sample0['filename']
        filename1 = sample1['filename']
    else:
        raise ValueError("Provide either --dataset_dir OR --image0_path and --image1_path")

    print(f"Matching with confidence threshold: {confidence_threshold}")
    matches = matcher(img0, img1, confidence_threshold=confidence_threshold)

    print(f"\nResults:")
    print(f"Image 0: {filename0}")
    print(f"Image 1: {filename1}")
    print(f"Number of matches: {len(matches['keypoints0'])}")
    print(f"Inliers: {matches['inliers'].sum()}")

    print(f"\nVisualizing matches...")
    plt.figure(figsize=(16, 8))
    matcher.draw(matches)
    plt.title(f"{args.model.upper()} Matches: {matches['inliers'].sum()} inliers")
    plt.tight_layout()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {args.output_path}")
    print("Done!")

