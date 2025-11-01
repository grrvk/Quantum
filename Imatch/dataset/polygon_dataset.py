import json
import rasterio
from pyproj import Transformer
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import Polygon, shape
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
import albumentations as A
from Imatch.dataset.utils import get_train_transforms_with_resize, get_val_transforms_with_resize
import matplotlib.pyplot as plt


class PolygonImageDataset(Dataset):
    """
    Dataset that loads images with GeoJSON polygon annotations.
    """

    def __init__(
            self,
            root_dir: str,
            geojson_file: str,
            target_size: Tuple[int, int] = (512, 512),
            mode: str = 'train',
            max_keypoints: int = 512,
            transform: Optional[A.Compose] = None,
    ):
        """
        Initialize dataset
        """

        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.mode = mode
        self.max_keypoints = max_keypoints

        self.image_paths = sorted(list(self.root_dir.rglob("*.jp2")))

        self.geojson_file = Path(geojson_file)
        self.load_polygons()
        self.transform = transform

    @staticmethod
    def poly_to_utm(polygon, transform, image_crs: str, src_crs: str = 'EPSG:4326'):
        """
        Convert polygon coordinates from geographic to UTM pixel coordinates
        """

        polygons_pixel = []
        transformer = Transformer.from_crs(src_crs, image_crs, always_xy=True)

        if polygon.geom_type == 'MultiPolygon':
            geometries = polygon.geoms
        else:
            geometries = [polygon]

        for geom in geometries:
            poly_pts = []
            for coord in geom.exterior.coords:
                lon, lat = coord[0], coord[1]
                x_utm, y_utm = transformer.transform(lon, lat)

                col, row = ~transform * (x_utm, y_utm)
                poly_pts.append((col, row))

            if len(poly_pts) > 2:
                polygons_pixel.append(Polygon(poly_pts))

        return polygons_pixel

    def load_polygons(self):
        """
        Load polygons from GeoJSON file and convert to pixel coordinates
        """

        print(f"Loading polygons from {self.geojson_file}")

        with open(self.geojson_file, 'r') as f:
            geojson_data = json.load(f)

        self.polygon_map = {}

        images_with_polygons = set()
        for feature in geojson_data.get('features', []):
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})

            tile = properties.get('tile', '')
            img_date = properties.get('img_date', '')

            matching_image = self.find_matching_image(tile, img_date)
            if matching_image is None:
                continue

            img_path = None
            for path in self.image_paths:
                if path.name == matching_image:
                    img_path = path
                    break

            if img_path is None:
                continue

            shapely_geom = shape(geometry)

            with rasterio.open(img_path) as src:
                transform = src.transform
                image_crs = src.crs
                image_width = src.width
                image_height = src.height

            pixel_polygons = self.poly_to_utm(shapely_geom, transform, image_crs)

            if matching_image not in self.polygon_map:
                self.polygon_map[matching_image] = []

            for pixel_polygon in pixel_polygons:
                vertices = list(pixel_polygon.exterior.coords)

                vertices_array = np.array(vertices)
                if len(vertices_array) > 0:
                    if np.allclose(vertices_array[0], vertices_array[-1]):
                        vertices_array = vertices_array[:-1]

                    valid_mask = (vertices_array[:, 0] >= 0) & (vertices_array[:, 0] < image_width) & \
                                 (vertices_array[:, 1] >= 0) & (vertices_array[:, 1] < image_height)
                    vertices_array = vertices_array[valid_mask]

                    if len(vertices_array) > 0:
                        self.polygon_map[matching_image].append(vertices_array)
                        images_with_polygons.add(matching_image)

        original_count = len(self.image_paths)
        self.image_paths = [path for path in self.image_paths if path.name in images_with_polygons]

        all_image_names = {path.name for path in self.image_paths}
        images_without_polygons = all_image_names - images_with_polygons

        print(f"Total images in directory: {original_count}")
        print(f"Images with polygons: {len(images_with_polygons)}")
        print(f"Images without polygons: {len(images_without_polygons)}")
        print(f"Final dataset size (images with polygons): {len(self.image_paths)}")

    def find_matching_image(self, tile: str, img_date: str) -> Optional[str]:
        """
        Find image file matching tile and date
        """

        date_formatted = img_date.replace('-', '')

        for img_path in self.image_paths:
            filename = img_path.name
            if tile in filename and date_formatted in filename:
                return filename
        return None

    @staticmethod
    def load_image(img_path: Path) -> np.ndarray:
        """
        Load image
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

        return image

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item with image and ground polygon keypoints
        """

        img_path = self.image_paths[idx]
        filename = img_path.name

        image = self.load_image(img_path)
        original_height, original_width = image.shape[:2]

        gt_keypoints = None
        has_polygons = len(self.polygon_map[filename]) > 0

        if has_polygons:
            all_vertices = []
            for vertices in self.polygon_map[filename]:
                all_vertices.append(vertices)

            gt_keypoints = np.concatenate(all_vertices, axis=0)
            valid_mask = (gt_keypoints[:, 0] >= 0) & (gt_keypoints[:, 0] < original_width) & \
                         (gt_keypoints[:, 1] >= 0) & (gt_keypoints[:, 1] < original_height)
            gt_keypoints = gt_keypoints[valid_mask]

            if len(gt_keypoints) > self.max_keypoints:
                indices = np.random.choice(
                    len(gt_keypoints),
                    self.max_keypoints,
                    replace=False
                )
                gt_keypoints = gt_keypoints[indices]

        if self.transform is not None:
            if has_polygons and gt_keypoints is not None and len(gt_keypoints) > 0:
                keypoints_list = [(x, y) for x, y in gt_keypoints]
                transformed = self.transform(image=image, keypoints=keypoints_list)
                image = transformed['image']
                gt_keypoints = np.array(transformed['keypoints']) if transformed['keypoints'] else np.array([])
                transformed_height, transformed_width = image.shape[1], image.shape[2]

                if len(gt_keypoints) > 0:
                    valid_mask = (gt_keypoints[:, 0] >= 0) & (gt_keypoints[:, 0] < transformed_width) & \
                                 (gt_keypoints[:, 1] >= 0) & (gt_keypoints[:, 1] < transformed_height)
                    gt_keypoints = gt_keypoints[valid_mask]

                gt_keypoints = torch.from_numpy(gt_keypoints).float() if len(gt_keypoints) > 0 else None
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            if gt_keypoints is not None and len(gt_keypoints) > 0:
                gt_keypoints = torch.from_numpy(gt_keypoints).float()

        return {
            'image': image,
            'keypoints': gt_keypoints,
            'has_polygons': has_polygons,
            'filename': filename,
            'image_path': str(img_path)
        }

    def draw(self, idx: int, figsize: Tuple[int, int] = (15, 10),
             point_size: int = 10, alpha: float = 0.8) -> None:
        """
        Plot image with polygon keypoints overlay
        """

        sample = self[idx]
        image = sample['image']
        keypoints = sample['keypoints']
        filename = sample['filename']
        has_polygons = sample['has_polygons']

        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()
            if image.ndim == 3 and image.shape[0] == 1:
                image = image.squeeze()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        ax1.imshow(image, cmap='gray')
        ax1.set_title(f'Image: {filename}\nShape: {image.shape}')
        ax1.axis('off')

        ax2.imshow(image, cmap='gray')
        if has_polygons and keypoints is not None:
            if isinstance(keypoints, torch.Tensor):
                keypoints_np = keypoints.numpy()
            else:
                keypoints_np = keypoints

            h, w = image.shape[:2]
            valid_mask = (keypoints_np[:, 0] > 2) & (keypoints_np[:, 0] < w - 2) & \
                             (keypoints_np[:, 1] > 2) & (keypoints_np[:, 1] < h - 2)
            keypoints_np = keypoints_np[valid_mask]

            if len(keypoints_np) > 0:
                ax2.scatter(keypoints_np[:, 0], keypoints_np[:, 1],
                            c='red', s=point_size, alpha=alpha, marker='o')
                ax2.set_title(f'With Polygon Keypoints')
            else:
                ax2.set_title('No Valid Keypoints')
        else:
            ax2.set_title('No Polygons Available')

        ax2.axis('off')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.show()


if __name__ == "__main__":
    dataset = PolygonImageDataset(
        root_dir="data/base_dataset",
        geojson_file="data/base_dataset/deforestation_labels.geojson",
        target_size=(512, 512),
        mode='train',
        transform=get_val_transforms_with_resize((512, 512))
    )

    print(f"Dataset size: {len(dataset)}")

    i = 2
    sample = dataset[i]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Has polygons: {sample['has_polygons']}")
    if sample['has_polygons']:
        print(f"GT keypoints shape: {sample['keypoints'].shape}")
        print(f"GT keypoints: {sample['keypoints']}")
        dataset.draw(i)