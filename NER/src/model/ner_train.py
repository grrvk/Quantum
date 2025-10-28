import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
import pandas as pd
import os
from typing import Optional, Dict, List, Any, Tuple


class NERTrainer:
    """
    NER Trainer
    """

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        """
        Initialize the NER Trainer params
        """

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id: Optional[Dict[str, int]] = None
        self.id2label: Optional[Dict[int, str]] = None

    def load_data_from_csv(self, csv_path: str) -> Dict[str, List[List[str]]]:
        """
        Load data from CSV with text and tags columns
        """

        df = pd.read_csv(csv_path)

        tokens_list: List[List[str]] = []
        ner_tags_list: List[List[str]] = []

        for _, row in df.iterrows():
            tokens = str(row['text']).split()
            tags = str(row['tags']).split()

            if len(tokens) == len(tags):
                tokens_list.append(tokens)
                ner_tags_list.append(tags)

        return {"tokens": tokens_list, "ner_tags": ner_tags_list}

    def _get_label_mappings(self, ner_tags_list: List[List[str]]) -> List[str]:
        """
        Create label mappings from all tags in the dataset
        """

        all_tags = set()
        for tags in ner_tags_list:
            all_tags.update(tags)

        label_list = sorted(list(all_tags))
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
        return label_list

    def _tokenize_and_align_labels(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize text and align labels with tokenized output
        """

        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=512,
        )

        labels: List[List[int]] = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx: Optional[int] = None
            label_ids: List[int] = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _compute_metrics(self, p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute metrics for NER
        """

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    def _prepare_train_data(self, data: Dict[str, List[List[str]]]) -> Tuple[Dataset, Dataset]:
        """
        Prepare training data with train/validation split
        """

        self._get_label_mappings(data["ner_tags"])
        dataset = Dataset.from_dict(data)

        train_eval_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_eval_split["train"]
        eval_dataset = train_eval_split["test"]

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(eval_dataset)}")

        tokenized_train = train_dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_eval = eval_dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        return tokenized_train, tokenized_eval

    def _prepare_test_data(self, test_data: Dict[str, List[List[str]]]) -> Dataset:
        """
        Prepare test data for evaluation
        """

        if self.label2id is None:
            raise ValueError("Label mappings not found. Train the model first or load a trained model.")

        test_dataset = Dataset.from_dict(test_data)
        tokenized_test = test_dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=test_dataset.column_names
        )
        return tokenized_test

    def train(self,
              training_data: Dict[str, List[List[str]]],
              output_dir: str = "./ner-model",
              **training_args: Any) -> Trainer:
        """
        Train the NER model
        """

        tokenized_train, tokenized_eval = self._prepare_train_data(training_data)

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )

        base_args = {
            "output_dir": output_dir,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 64,
            "per_device_eval_batch_size": 64,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "logging_dir": f"{output_dir}/logs",
        }
        merged_args = {**base_args, **training_args}
        args = TrainingArguments(**merged_args)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        print("Starting training...")
        trainer.train()

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        return trainer

    def test(self,
             trainer: Trainer,
             test_data: Dict[str, List[List[str]]]
             ) -> Dict[str, Any]:
        """
        Get test metrics
        """

        tokenized_test = self._prepare_test_data(test_data)

        predictions = trainer.predict(tokenized_test)
        return predictions[2]

    @staticmethod
    def plot_training_metrics(trainer: Trainer,
                              show_plot: bool = True) -> Optional[go.Figure]:
        """
        Create Plotly charts with metrics
        """

        history = trainer.state.log_history

        if not history:
            print("No training history found!")
            return None

        df = pd.DataFrame(history)

        train_metrics = df[df['epoch'].notna()].copy()
        eval_metrics = df[df['eval_loss'].notna()].copy()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Precision', 'Recall', 'F1 Score'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        if 'loss' in train_metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=train_metrics['epoch'],
                    y=train_metrics['loss'],
                    mode='lines+markers',
                    name='Train Loss',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8),
                    hovertemplate='Epoch: %{x}<br>Train Loss: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )

        if 'eval_loss' in eval_metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=eval_metrics['epoch'],
                    y=eval_metrics['eval_loss'],
                    mode='lines+markers',
                    name='Eval Loss',
                    line=dict(color='red', width=3),
                    marker=dict(size=8),
                    hovertemplate='Epoch: %{x}<br>Eval Loss: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )

        if 'eval_precision' in eval_metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=eval_metrics['epoch'],
                    y=eval_metrics['eval_precision'],
                    mode='lines+markers',
                    name='Eval Precision',
                    line=dict(color='green', width=3),
                    marker=dict(size=8),
                    hovertemplate='Epoch: %{x}<br>Precision: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )

        if 'eval_recall' in eval_metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=eval_metrics['epoch'],
                    y=eval_metrics['eval_recall'],
                    mode='lines+markers',
                    name='Eval Recall',
                    line=dict(color='orange', width=3),
                    marker=dict(size=8),
                    hovertemplate='Epoch: %{x}<br>Recall: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )

        if 'eval_f1' in eval_metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=eval_metrics['epoch'],
                    y=eval_metrics['eval_f1'],
                    mode='lines+markers',
                    name='Eval F1',
                    line=dict(color='purple', width=3),
                    marker=dict(size=8),
                    hovertemplate='Epoch: %{x}<br>F1: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title={
                'text': 'NER Training Metrics',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=700,
            width=900,
            showlegend=True,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_xaxes(title_text='Epoch', row=2, col=1)
        fig.update_xaxes(title_text='Epoch', row=2, col=2)

        fig.update_yaxes(title_text='Loss', row=1, col=1)
        fig.update_yaxes(title_text='Precision', row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text='Recall', row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text='F1 Score', row=2, col=2, range=[0, 1])

        if show_plot:
            fig.show()

        return fig


if __name__ == "__main__":
    import argparse

    current_dir = os.path.basename(os.getcwd())
    if not current_dir == "NER":
        raise Exception("Must be called from NER directory")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, help="Path to test CSV")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--output_dir", type=str, default="results/ner-model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")

    args = parser.parse_args()

    ner_trainer = NERTrainer(model_name=args.model_name)

    print("Loading training data...")
    train_data = ner_trainer.load_data_from_csv(args.train_csv)

    print("Starting training...")
    trained_trainer = ner_trainer.train(
        train_data,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs
    )

    print("Generating training plots...")
    NERTrainer.plot_training_metrics(trained_trainer)

    if args.test_csv:
        if os.path.exists(args.test_csv):
            print(f"Loading test data...")
            test_data = ner_trainer.load_data_from_csv(args.test_csv)

            print("Running evaluation on test set...")
            test_results = ner_trainer.test(
                trainer=trained_trainer,
                test_data=test_data
            )

            print("\nTEST RESULTS SUMMARY:")
            print(f"F1 Score:     {test_results.get('test_f1', 0):.4f}")
            print(f"Precision:    {test_results.get('test_precision', 0):.4f}")
            print(f"Recall:       {test_results.get('test_recall', 0):.4f}")

    print("\nPipeline completed successfully.")
