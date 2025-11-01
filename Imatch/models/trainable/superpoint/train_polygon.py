import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from pathlib import Path

from Imatch.dataset.polygon_dataset import PolygonImageDataset
from Imatch.dataset.utils import get_train_transforms_with_resize, get_val_transforms_with_resize

from lightglue import SuperPoint


def collate_fn_polygon(batch):
    """
    Custom collate function for PolygonImageDataset to handle variable-length keypoints
    """

    if not batch:
        return {}

    images = []
    keypoints = []
    has_polygons = []
    filenames = []
    image_paths = []

    for item in batch:
        images.append(item['image'])
        keypoints.append(item['keypoints'])
        has_polygons.append(item['has_polygons'])
        filenames.append(item['filename'])
        image_paths.append(item['image_path'])

    try:
        images_batch = torch.stack(images, dim=0)
    except RuntimeError as e:
        print(f"Error stacking images: {e}")
        print(f"Image shapes: {[img.shape for img in images]}")
        raise e

    return {
        'image': images_batch,
        'keypoints': keypoints,
        'has_polygons': has_polygons,
        'filename': filenames,
        'image_path': image_paths
    }


class PolygonSuperPointTrainer:
    """
    SuperPoint Trainer to detect polygon keypoints extracted from geojson
    """

    def __init__(self,
                 train_dir: str,
                 val_dir: Optional[str] = None,
                 geojson_path: str = None,
                 target_size: Tuple[int, int] = (512, 512),
                 max_keypoints: int = 512,
                 learning_rate: float = 1e-4,
                 proximity_threshold: float = 5.0):
        """
        Initialize Polygon SuperPoint Trainer
        """

        self.target_size = target_size
        self.max_keypoints = max_keypoints
        self.learning_rate = learning_rate
        self.proximity_threshold = proximity_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.geojson_path = geojson_path

        self._dataset_setup()
        self._model_setup()

        self.history = {
            'train_loss': [],
            'train_recall': [],
            'train_precision': [],
            'train_mean_distance': [],
            'val_loss': [],
            'val_recall': [],
            'val_precision': [],
            'val_mean_distance': []
        }

    def _model_setup(self):
        """
        Setup SuperPoint model with frozen backbone
        """

        self.superpoint = SuperPoint(max_num_keypoints=self.max_keypoints).to(self.device)

        trainable_params = self.superpoint.parameters()
        self.optimizer = optim.Adam(trainable_params, lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def _dataset_setup(self):
        """
        Initialize datasets with keypoints
        """

        if not self.geojson_path or not Path(self.geojson_path).exists():
            raise ValueError(f"GeoJSON path not found: {self.geojson_path}")

        self.train_dataset = PolygonImageDataset(
            root_dir=self.train_dir,
            geojson_file=self.geojson_path,
            target_size=self.target_size,
            mode='train',
            max_keypoints=self.max_keypoints,
            transform=get_train_transforms_with_resize(self.target_size)
        )

        if self.val_dir:
            self.val_dataset = PolygonImageDataset(
                root_dir=self.val_dir,
                geojson_file=self.geojson_path,
                target_size=self.target_size,
                mode='val',
                max_keypoints=self.max_keypoints,
                transform=get_val_transforms_with_resize(self.target_size)
            )
        else:
            self.val_dataset = None

    def superpoint_loss(self, pred_keypoints: torch.Tensor, gt_keypoints: torch.Tensor,
                                pred_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Loss function
        """

        metrics = {}

        pred_keypoints = pred_keypoints.to(self.device).detach().requires_grad_(False)
        gt_keypoints = gt_keypoints.to(self.device).detach().requires_grad_(False)
        pred_scores = pred_scores.to(self.device)

        if len(gt_keypoints) == 0:
            if len(pred_keypoints) > 0:
                confidence_loss = pred_scores.mean()
                metrics.update({
                    'recall': 0.0,
                    'precision': 0.0,
                    'mean_distance': float('inf'),
                    'confidence_loss': confidence_loss.item()
                })
                return confidence_loss * 0.1, metrics
            else:
                metrics.update({
                    'recall': 0.0,
                    'precision': 0.0,
                    'mean_distance': float('inf'),
                    'confidence_loss': 0.0
                })
                return torch.tensor(0.0, device=self.device, requires_grad=True), metrics

        if len(pred_keypoints) == 0:
            metrics.update({
                'recall': 0.0,
                'precision': 0.0,
                'mean_distance': float('inf'),
                'confidence_loss': 1.0
            })
            return torch.tensor(0.1, device=self.device, requires_grad=True), metrics

        try:
            distances = torch.cdist(gt_keypoints.unsqueeze(0), pred_keypoints.unsqueeze(0))[0]

            min_distances, closest_pred_indices = torch.min(distances, dim=1)

            recall_mask = min_distances <= self.proximity_threshold
            recall = recall_mask.float().mean()

            pred_to_gt_distances, _ = torch.min(distances, dim=0)
            precision_mask = pred_to_gt_distances <= self.proximity_threshold
            precision = precision_mask.float().mean()

            if recall_mask.any():
                mean_distance_tensor = min_distances[recall_mask].mean()
                mean_distance_value = mean_distance_tensor.item()
            else:
                mean_distance_tensor = torch.tensor(float('inf'), device=self.device)
                mean_distance_value = float('inf')

            confidence_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if recall_mask.any():
                matched_pred_indices = closest_pred_indices[recall_mask]
                matched_confidences = pred_scores[matched_pred_indices]
                confidence_loss = (1.0 - matched_confidences).mean()

            if torch.isfinite(mean_distance_tensor):
                distance_loss = mean_distance_tensor.detach().requires_grad_(True)
            else:
                distance_loss = torch.tensor(10.0, device=self.device, requires_grad=True)

            coverage_ratio = torch.tensor(
                len(pred_keypoints) / max(len(gt_keypoints), 1),
                device=self.device,
                requires_grad=True
            )
            target_ratio = torch.tensor(1.5, device=self.device, requires_grad=True)
            coverage_loss = F.mse_loss(coverage_ratio, target_ratio)

            recall_loss = (1.0 - recall).requires_grad_(True)
            total_loss = recall_loss + 0.1 * distance_loss + 0.1 * confidence_loss + 0.05 * coverage_loss

            metrics.update({
                'recall': recall.item(),
                'precision': precision.item(),
                'mean_distance': mean_distance_value,
                'confidence_loss': confidence_loss.item(),
                'distance_loss': distance_loss.item(),
                'coverage_loss': coverage_loss.item(),
                'num_gt': len(gt_keypoints),
                'num_pred': len(pred_keypoints),
                'coverage_ratio': coverage_ratio.item()
            })

            return total_loss, metrics

        except Exception as e:
            print(f"Error in loss computation: {e}")
            print(f"pred_keypoints: {pred_keypoints.shape}, gt_keypoints: {gt_keypoints.shape}")
            import traceback
            traceback.print_exc()
            metrics.update({
                'recall': 0.0,
                'precision': 0.0,
                'mean_distance': float('inf'),
                'error': str(e)
            })
            return torch.tensor(0.1, device=self.device, requires_grad=True), metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """
        Train SuperPoint for one epoch
        """

        self.superpoint.train()

        epoch_loss = 0.0
        epoch_recall = 0.0
        epoch_precision = 0.0
        epoch_distance = 0.0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            gt_keypoints_list = batch['keypoints']
            has_polygons_list = batch['has_polygons']

            batch_size = images.shape[0]
            batch_loss = 0.0
            batch_recall = 0.0
            batch_precision = 0.0
            batch_distance = 0.0
            processed_samples = 0

            for i in range(batch_size):
                if not has_polygons_list[i]:
                    continue

                self.optimizer.zero_grad()

                try:
                    image = images[i].unsqueeze(0)
                    feats = self.superpoint.forward({"image": image})

                    pred_keypoints = feats['keypoints'][0]
                    pred_scores = feats['keypoint_scores'][0]
                    gt_keypoints = gt_keypoints_list[i].to(self.device)

                    if len(pred_keypoints) == 0 and len(gt_keypoints) == 0:
                        continue

                    pred_keypoints = pred_keypoints.to(self.device)
                    pred_scores = pred_scores.to(self.device)

                    loss, metrics = self.superpoint_loss(pred_keypoints, gt_keypoints, pred_scores)
                    loss.backward()
                    self.optimizer.step()

                    batch_loss += loss.item()
                    batch_recall += metrics['recall']
                    batch_precision += metrics['precision']
                    if 'mean_distance' in metrics and np.isfinite(metrics['mean_distance']):
                        batch_distance += metrics['mean_distance']
                    processed_samples += 1

                except Exception as e:
                    print(f"Error in batch {batch_idx}, sample {i}: {e}")
                    continue

            if processed_samples > 0:
                epoch_loss += batch_loss
                epoch_recall += batch_recall
                epoch_precision += batch_precision
                epoch_distance += batch_distance
                total_samples += processed_samples

                progress_bar.set_postfix({
                    'Loss': f'{batch_loss / processed_samples:.4f}',
                    'Recall': f'{batch_recall / processed_samples:.3f}',
                    'Precision': f'{batch_precision / processed_samples:.3f}',
                    'Samples': f'{processed_samples}/{batch_size}'
                })

        if total_samples > 0:
            return {
                'loss': epoch_loss / total_samples,
                'recall': epoch_recall / total_samples,
                'precision': epoch_precision / total_samples,
                'mean_distance': epoch_distance / total_samples
            }
        else:
            return {'loss': 0.0, 'recall': 0.0, 'precision': 0.0, 'mean_distance': float('inf')}

    def validate(self, dataloader: DataLoader) -> Dict:
        """
        Validate SuperPoint
        """

        self.superpoint.eval()

        val_loss = 0.0
        val_recall = 0.0
        val_precision = 0.0
        val_distance = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                gt_keypoints_list = batch['keypoints']
                has_polygons_list = batch['has_polygons']

                batch_size = images.shape[0]

                for i in range(batch_size):
                    if not has_polygons_list[i]:
                        continue

                    try:
                        image = images[i].unsqueeze(0)
                        feats = self.superpoint.forward({"image": image})

                        pred_keypoints = feats['keypoints'][0]
                        pred_scores = feats['keypoint_scores'][0]
                        gt_keypoints = gt_keypoints_list[i].to(self.device)

                        loss, metrics = self.superpoint_loss(pred_keypoints, gt_keypoints, pred_scores)

                        val_loss += loss.item()
                        val_recall += metrics['recall']
                        val_precision += metrics['precision']
                        if np.isfinite(metrics['mean_distance']):
                            val_distance += metrics['mean_distance']
                        total_samples += 1

                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue

        if total_samples > 0:
            return {
                'loss': val_loss / total_samples,
                'recall': val_recall / total_samples,
                'precision': val_precision / total_samples,
                'mean_distance': val_distance / total_samples
            }
        else:
            return {'loss': 0.0, 'recall': 0.0, 'precision': 0.0, 'mean_distance': float('inf')}

    def train(self, num_epochs: int = 50, batch_size: int = 4, save_dir: str = 'polygon_superpoint_checkpoints'):
        """
        Main training loop for SuperPoint
        """

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_polygon,
            pin_memory=False
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_polygon,
            pin_memory=False
        ) if self.val_dataset else None

        print(f"Starting PolygonSuperPoint training for {num_epochs} epochs")

        best_recall = 0.0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['train_mean_distance'].append(train_metrics['mean_distance'])

            if val_loader:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recall'].append(val_metrics['recall'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_mean_distance'].append(val_metrics['mean_distance'])

                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, Recall: {train_metrics['recall']:.3f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, Recall: {val_metrics['recall']:.3f}")

                if val_metrics['recall'] > best_recall:
                    best_recall = val_metrics['recall']
                    self.save_checkpoint(save_dir, 'best_superpoint_gt.pth', epoch)
            else:
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Recall: {train_metrics['recall']:.3f}, "
                      f"Precision: {train_metrics['precision']:.3f}")

            self.scheduler.step()

        self.save_checkpoint(save_dir, 'polygon_superpoint_final.pth', num_epochs)
        self.plot_training_history(save_dir)
        print("PolygonSuperPoint training completed!")

    def save_checkpoint(self, save_dir: str, filename: str, epoch: int):
        """
        Save SuperPoint checkpoint
        """

        checkpoint = {
            'epoch': epoch,
            'superpoint_state_dict': self.superpoint.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': {
                'target_size': self.target_size,
                'max_keypoints': self.max_keypoints,
                'learning_rate': self.learning_rate,
                'proximity_threshold': self.proximity_threshold
            }
        }
        torch.save(checkpoint, Path(save_dir) / filename)
        print(f"Checkpoint saved: {filename}")

    def plot_training_history(self, save_dir: str):
        """
        Plot training history
        """

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training Loss')

        ax2.plot(self.history['train_recall'], label='Train Recall')
        if self.history['val_recall']:
            ax2.plot(self.history['val_recall'], label='Val Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Recall')
        ax2.legend()
        ax2.set_title('Keypoint Recall')

        ax3.plot(self.history['train_precision'], label='Train Precision')
        if self.history['val_precision']:
            ax3.plot(self.history['val_precision'], label='Val Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.set_title('Keypoint Precision')

        ax4.plot(self.history['train_mean_distance'], label='Train Mean Distance')
        if self.history['val_mean_distance']:
            ax4.plot(self.history['val_mean_distance'], label='Val Mean Distance')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Mean Distance (px)')
        ax4.legend()
        ax4.set_title('Mean Distance to GT')

        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'polygon_superpoint_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()

    current_dir = os.path.basename(os.getcwd())
    if not current_dir == "Imatch":
        raise Exception("Must be called from Imatch directory")

    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory containing validation images (optional)')
    parser.add_argument('--geojson_path', type=str, required=True,
                        help='Path to GeoJSON file with polygon annotations')
    parser.add_argument('--max_keypoints', type=int, default=512,
                        help='Maximum number of keypoints to extract (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--proximity_threshold', type=float, default=5.0,
                        help='Distance threshold for keypoint matching in pixels (default: 5.0)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--save_dir', type=str,
                        default='models/trainable/superpoint/polygon_superpoint_checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    trainer = PolygonSuperPointTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        geojson_path=args.geojson_path,
        max_keypoints=args.max_keypoints,
        learning_rate=args.learning_rate,
        proximity_threshold=args.proximity_threshold
    )

    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )