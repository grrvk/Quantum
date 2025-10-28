import pandas as pd
from typing import List, Optional
import os
from datetime import datetime
from tqdm import tqdm

from NER.src.dataset.generator.base_ollama_generator import IOllamaGenerator
from NER.src.dataset.tagger.base_tagger import ITagger


class NERDataset:
    """
    Pipeline that generates samples, tags them, and saves to CSV
    """

    def __init__(
            self,
            generator: Optional[IOllamaGenerator] = None,
            tagger: Optional[ITagger] = None,
            output_dir: str = "data"
    ):
        """
        Initialize the dataset pipeline
        """

        self.generator = generator or IOllamaGenerator()
        self.tagger = tagger or ITagger()
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def run(
            self,
            num_samples: int,
            batch_size: int = 10,
            save_to_csv: bool = True,
            filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run the complete pipeline: generate -> tag -> save to CSV
        """

        print(f"Starting pipeline for {num_samples} samples...")

        all_samples = []

        batch_range = range(0, num_samples, batch_size)
        for batch_start in tqdm(batch_range, desc="Processing batches", unit="batch"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_size_current = batch_end - batch_start

            raw_samples = self.generator.generate(batch_size_current)
            tagged_samples = self.tagger.tag(raw_samples)
            batch_df = self._samples_to_dataframe(tagged_samples)
            all_samples.append(batch_df)

        final_df = pd.concat(all_samples, ignore_index=True)
        if save_to_csv:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ner_dataset_{timestamp}.csv"

            filepath = os.path.join(self.output_dir, filename)
            final_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"Dataset saved to: {filepath}")

        print(f"Pipeline completed. Generated {len(final_df)} samples.")
        return final_df

    @staticmethod
    def _samples_to_dataframe(samples: List) -> pd.DataFrame:
        """
        Convert tagged samples to DataFrame format
        """

        data = []

        for sample in samples:
            if hasattr(sample, 'text') and hasattr(sample, 'tags') and hasattr(sample, 'entities'):
                text = sample.text
                tags = ' '.join(sample.tags)
                entities = ', '.join([e.text for e in sample.entities])

                row = {
                    'text': text,
                    'tags': tags,
                    'entities': entities,
                }
                data.append(row)

        return pd.DataFrame(data)


if __name__ == '__main__':
    if __name__ == '__main__':
        import argparse

        from NER.src.dataset.generator.mistral.mistral_generator import MistralGenerator
        from NER.src.dataset.tagger.bio.bio_tagger import BioTagger

        current_dir = os.path.basename(os.getcwd())
        if not current_dir == "NER":
            raise Exception("Must be called from NER directory")

        parser = argparse.ArgumentParser()
        parser.add_argument("--samples", type=int, required=True,
                            help="Number of samples to generate")
        parser.add_argument("--output", type=str, default="results/data",
                            help="Output CSV file path")

        args = parser.parse_args()

        pipeline = NERDataset(
            generator=MistralGenerator(),
            tagger=BioTagger(),
            output_dir=args.output,
        )
        df = pipeline.run(args.samples)
        print(f"Generated dataset shape: {df.shape}")
        print(df.head())

