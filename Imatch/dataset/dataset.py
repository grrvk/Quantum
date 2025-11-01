from torch.utils.data import Dataset
import rasterio
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import kornia as K
import torch


class ImageDataset(Dataset):
    """
    Dataset that loads images
    """

    def __init__(self,
                 root_dir: str,
                 target_size: Tuple[int, int] = (1280, 1280),
                 transform: Optional[A.Compose] = None,
                 device: str = 'cpu'):
        """
        Initialize dataset
        """

        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.image_paths = list(self.root_dir.rglob("*.jp2"))
        self.transform = transform
        self.device = device

    def load_image(self, img_path: Path) -> np.ndarray:
        """
        Load and preprocess image
        """

        with rasterio.open(img_path) as src:
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

        if self.target_size is not None:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        return image

    def _convert_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to normalized torch tensor with resizing
        """

        image_tensor = K.utils.image_to_tensor(image)
        image_tensor = image_tensor.float().unsqueeze(dim=0).to(self.device) / 255.0

        image_tensor = K.geometry.resize(image_tensor, self.target_size, interpolation='area')
        return image_tensor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        image = self.load_image(img_path)

        if self.transform:
            image = self.transform(image=image)['image']

        image_tensor = self._convert_image(image)

        return {
            'image': image_tensor,
            'image_np': image,
            'image_path': str(img_path),
            'filename': img_path.name,
        }

    def draw(self, idx: int, figsize: Tuple[int, int] = (15, 10)):
        sample = self[idx]
        plt.figure(figsize=figsize)

        plt.subplot(1, 3, 1)
        plt.imshow(sample['image_np'])
        plt.title(f"Original shape: {sample['image_np'].shape}")

        plt.subplot(1, 3, 2)
        image_tensor_np = K.tensor_to_image(sample['image'])
        plt.imshow(image_tensor_np)
        plt.title(f"Tensor shape: {sample['image'].shape}")

        plt.subplot(1, 3, 3)
        plt.hist(sample['image_np'].flatten(), bins=50, alpha=0.7, label='Original')
        plt.hist(sample['image'].cpu().numpy().flatten(), bins=50, alpha=0.7, label='Tensor')

        plt.title("Pixel value distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dataset = ImageDataset(
        root_dir="dataset/data"
    )

    i = 0
    sample = dataset[i]
    print(f"Original image shape: {sample['image_np'].shape}")
    print(f"Tensor image shape: {sample['image'].shape}")
    print(f"Tensor image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Device: {sample['image'].device}")
    dataset.draw(i)