import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from Imatch.dataset.utils import get_train_transforms, get_val_transforms

from pathlib import Path

from Imatch.dataset.dataset import ImageDataset

from lightglue import SuperPoint
from lightglue.utils import rbd


class SuperPointTrainer:
    """
    SuperPoint Trainer by self-matching descriptors of keypoints on one augmented different ways image
    """

    def __init__(self,
                 train_dir: str,
                 val_dir: Optional[str] = None,
                 target_size: Tuple[int, int] = (512, 512),
                 max_keypoints: int = 1024,
                 learning_rate: float = 1e-4
                 ):
        """
        Initialize SuperPoint Trainer
        """

        self.target_size = target_size
        self.max_keypoints = max_keypoints
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dir = train_dir
        self.val_dir = val_dir
        self._dataset_setup()

        self._model_setup()

        self.history = {
            'train_loss': [],
            'train_keypoints': [],
            'train_matches': [],
            'val_loss': [],
            'val_keypoints': [],
            'val_matches': []
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
        Initialize simple datasets with only images returned (no keypoints)
        """

        self.train_dataset = ImageDataset(self.train_dir, self.target_size)

        if self.val_dir:
            self.val_dataset = ImageDataset(self.val_dir, self.target_size)
        else:
            self.val_dataset = None

    def create_pair(self, image: torch.Tensor, mode: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create pair from single image
        """

        image_np = image.squeeze(0).cpu().numpy().transpose(2, 1, 0)

        if mode == 'train':
            aug1 = get_train_transforms()(image=image_np)
            aug2 = get_train_transforms()(image=image_np)
        elif mode == 'val':
            aug1 = get_val_transforms()(image=image_np)
            aug2 = get_val_transforms()(image=image_np)
        else:
            raise ValueError(f'Unknown mode {mode}')

        img1 = aug1['image']
        img2 = aug2['image']
        return img1.to(self.device), img2.to(self.device)

    def superpoint_loss(self, feats0: Dict, feats1: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Loss function
        """

        loss_components = {}

        # 1. Keypoint Confidence Loss
        confidence_loss = (1.0 - feats0['keypoint_scores'].mean()) + (1.0 - feats1['keypoint_scores'].mean())
        confidence_loss = confidence_loss * 0.1
        loss_components['confidence_loss'] = confidence_loss.item()

        # 2. Descriptor Consistency Loss with Similarity Matrix
        if feats0['descriptors'].shape[1] > 0 and feats1['descriptors'].shape[1] > 0:
            desc0 = feats0['descriptors']
            desc1 = feats1['descriptors']

            batch_size = desc0.shape[0]
            n0, n1 = desc0.shape[1], desc1.shape[1]

            desc0 = F.normalize(desc0, p=2, dim=2)
            desc1 = F.normalize(desc1, p=2, dim=2)
            max_keypoints = max(n0, n1)

            if n0 < max_keypoints:
                pad_size = max_keypoints - n0
                desc0 = F.pad(desc0, (0, 0, 0, pad_size), mode='constant', value=0)

            if n1 < max_keypoints:
                pad_size = max_keypoints - n1
                desc1 = F.pad(desc1, (0, 0, 0, pad_size), mode='constant', value=0)

            similarity_matrix = torch.bmm(desc0, desc1.transpose(1, 2))

            ideal_similarity = torch.eye(max_keypoints, device=desc0.device).unsqueeze(0)
            ideal_similarity = ideal_similarity.repeat(batch_size, 1, 1)

            descriptor_loss = F.mse_loss(similarity_matrix, ideal_similarity) * 0.1

        else:
            descriptor_loss = torch.tensor(0.0, device=self.device)

        loss_components['descriptor_loss'] = descriptor_loss.item()

        total_loss = confidence_loss + descriptor_loss
        loss_components['total_loss'] = total_loss.item()
        loss_components['num_matches'] = min(feats0['keypoints'].shape[1], feats1['keypoints'].shape[1])
        loss_components['keypoints0'] = feats0['keypoints'].shape[1]
        loss_components['keypoints1'] = feats1['keypoints'].shape[1]
        return total_loss, loss_components

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """
        Train SuperPoint for one epoch
        """

        self.superpoint.train()

        epoch_loss = 0.0
        epoch_matches = 0
        total_pairs = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            batch_size = images.shape[0]

            batch_loss = 0.0
            batch_matches = 0
            processed_pairs = 0

            for i in range(batch_size):
                image0, image1 = self.create_pair(images[i])

                self.optimizer.zero_grad()
                try:
                    image0_input = image0.unsqueeze(0)
                    image1_input = image1.unsqueeze(0)

                    feats0 = self.superpoint.forward({"image": image0_input})
                    feats1 = self.superpoint.forward({"image": image1_input})

                    if feats0['keypoints'].shape[1] == 0 or feats1['keypoints'].shape[1] == 0:
                        print(
                            f"Skipping pair - no keypoints: {feats0['keypoints'].shape[1]} vs {feats1['keypoints'].shape[1]}")
                        continue

                    loss, loss_components = self.superpoint_loss(feats0, feats1)

                    self.optimizer.zero_grad()
                    loss.backward()

                    total_grad_norm = 0.0
                    for name, param in self.superpoint.named_parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.norm().item()

                    self.optimizer.step()

                    batch_loss += loss_components['total_loss']
                    batch_matches += loss_components['num_matches']
                    processed_pairs += 1

                except Exception as e:
                    print(f"Error processing batch {batch_idx}, sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if processed_pairs > 0:
                avg_batch_loss = batch_loss / processed_pairs
                avg_batch_matches = batch_matches / processed_pairs

                epoch_loss += batch_loss
                epoch_matches += batch_matches
                total_pairs += processed_pairs

                progress_bar.set_postfix({
                    'Loss': f'{avg_batch_loss:.4f}',
                    'Matches': f'{avg_batch_matches:.1f}'
                })

        if total_pairs > 0:
            return {
                'loss': epoch_loss / total_pairs,
                'matches': epoch_matches / total_pairs
            }
        else:
            return {'loss': 0.0, 'matches': 0.0}

    def validate(self, dataloader: DataLoader) -> Dict:
        """
        Validate SuperPoint
        """

        self.superpoint.eval()

        val_loss = 0.0
        val_matches = 0
        total_pairs = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]

                for i in range(batch_size):
                    image0, image1 = self.create_pair(images[i], mode='val')

                    try:
                        image0_input = image0.unsqueeze(0)
                        image1_input = image1.unsqueeze(0)

                        feats0 = self.superpoint.forward({"image": image0_input})
                        feats1 = self.superpoint.forward({"image": image1_input})

                        loss, loss_components = self.superpoint_loss(feats0, feats1)

                        val_loss += loss_components['total_loss']
                        val_matches += loss_components['num_matches']
                        total_pairs += 1

                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue

        if total_pairs > 0:
            return {
                'loss': val_loss / total_pairs,
                'matches': val_matches / total_pairs,
            }
        else:
            return {'loss': 0.0, 'matches': 0.0}

    def train(self,
              num_epochs: int = 10,
              batch_size: int = 4,
              save_dir: str = 'superpoint_checkpoints'):
        """
        Main training loop for SuperPoint
        """

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )

        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
            )

        print(f"Starting SuperPoint training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_matches'].append(train_metrics['matches'])

            if self.val_dataset:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_matches'].append(val_metrics['matches'])

                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Matches: {train_metrics['matches']:.1f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Matches: {val_metrics['matches']:.1f}")

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_dir, 'best_superpoint.pth', epoch)
            else:
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Matches: {train_metrics['matches']:.1f}")

            self.scheduler.step()

        self.save_checkpoint(save_dir, 'superpoint_final.pth', num_epochs)
        self.plot_training_history(save_dir)
        print("SuperPoint training completed!")

    def save_checkpoint(self, save_dir: str, filename: str, epoch: int):
        """
        Save SuperPoint checkpoint
        """

        checkpoint = {
            'epoch': epoch,
            'superpoint_state_dict': self.superpoint.state_dict(),  # Only save SuperPoint!
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': {
                'target_size': self.target_size,
                'max_keypoints': self.max_keypoints,
                'learning_rate': self.learning_rate,
            }
        }

        torch.save(checkpoint, Path(save_dir) / filename)
        print(f"SuperPoint checkpoint saved: {filename}")

    def plot_training_history(self, save_dir: str):
        """
        Plot training history
        """

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('SuperPoint Training Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_matches'], label='Train Matches')
        if self.history['val_matches']:
            plt.plot(self.history['val_matches'], label='Val Matches')
        plt.xlabel('Epoch')
        plt.ylabel('Average Matches')
        plt.legend()
        plt.title('Matches per Pair')

        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'superpoint_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    current_dir = os.path.basename(os.getcwd())
    if not current_dir == "Imatch":
        raise Exception("Must be called from Imatch directory")

    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory containing validation images (optional)')
    parser.add_argument('--max_keypoints', type=int, default=512,
                        help='Maximum number of keypoints to extract (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--save_dir', type=str,
                        default='models/trainable/superpoint/superpoint_sm_checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    trainer = SuperPointTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        max_keypoints=args.max_keypoints,
        learning_rate=args.learning_rate,
    )

    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
