import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd


class NERInference:
    """
    NER Inference pipeline
    """

    def __init__(self, model_path: str = "results/ner-model"):
        """
        Initialize the NER inference pipeline
        """

        self.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="first"
        )

    def predict_single(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict NER tags for a single text string
        """

        return self.pipe(text)

    def predict_to_dataframe(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict NER tags and return as DataFrame with text, entity, label, confidence
        """

        results = []

        for text in texts:
            entities = self.predict_single(text)

            if entities:
                for entity in entities:
                    results.append({
                        'text': text,
                        'entity': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score']
                    })
            else:
                results.append({
                    'text': text,
                    'entity': None,
                    'label': None,
                    'confidence': None
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    import os

    current_dir = os.path.basename(os.getcwd())
    if not current_dir == "NER":
        raise Exception("Must be called from NER directory")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/ner-model",
                        help="Path to trained model")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--csv_file", type=str, help="CSV file with texts to analyze")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing texts in CSV")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--filename", type=str, help="Inference results save filename")

    args = parser.parse_args()

    print("Loading model...")
    ner_inference = NERInference(model_path=args.model_path)
    print("Model loaded successfully!")

    if args.text:
        entities = ner_inference.predict_single(args.text)
        print(f"\nInput: {args.text}")
        print("Entities found:")
        for entity in entities:
            print(f"  {entity['word']} -> {entity['entity_group']} (confidence: {entity['score']:.3f})")
    elif args.csv_file:
        df = pd.read_csv(args.csv_file)
        texts = df[args.text_column].tolist()

        results_df = ner_inference.predict_to_dataframe(texts)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"inference_res_{timestamp}.csv"
            else:
                filename = args.filename

            filepath = os.path.join(args.output_dir, filename)
            results_df.to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")
        else:
            print(results_df)
    else:
        raise ValueError("Must specify either --text or --csv_file")