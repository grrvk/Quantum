from typing import List
import re

from NER.src.dataset.generator.base_ollama_generator import Sample, Entity
from NER.src.dataset.tagger.base_tagger import ITagger


class BioTagger(ITagger):
    """
    Implementation of BIO tagger that converts text with entities to BIO format
    """

    def _tag_sample(self, sample: Sample) -> Sample:
        """
        Convert a single sample to BIO tagging format
        """

        if not hasattr(sample, 'text') or not hasattr(sample, 'entities'):
            return sample

        tokens = []
        token_positions = []

        for match in re.finditer(r'\S+', sample.text):
            tokens.append(match.group())
            token_positions.append((match.start(), match.end()))

        bio_tags = ['O'] * len(tokens)

        for entity in sample.entities:
            entity_tokens = self._find_entity_tokens(entity, token_positions)

            if entity_tokens:
                self._assign_bio_tags(bio_tags, entity_tokens, entity.label)

        sample.tags = bio_tags
        return sample

    def _find_entity_tokens(self, entity: Entity, token_positions: List[tuple]) -> List[int]:
        """
        Find which tokens correspond to an entity
        """

        entity_tokens = []

        for i, (token_start, token_end) in enumerate(token_positions):
            if not (token_end <= entity.start or token_start >= entity.end):
                entity_tokens.append(i)

        return entity_tokens

    @staticmethod
    def _assign_bio_tags(bio_tags: List[str], entity_tokens: List[int], label: str) -> None:
        """
        Assign BIO tags to entity tokens
        """

        for j, token_idx in enumerate(entity_tokens):
            if j == 0:
                bio_tags[token_idx] = f'B-{label}'
            else:
                bio_tags[token_idx] = f'I-{label}'

    def tag(self, samples: List[Sample]) -> List[Sample]:
        """
        Convert multiple samples to BIO tagging format
        """

        tagged_samples = []

        for sample in samples:
            try:
                tagged_sample = self._tag_sample(sample)
                tagged_samples.append(tagged_sample)
            except Exception as e:
                print(f"Error tagging sample: {e}")
                tagged_samples.append(sample)

        return tagged_samples
