from NER.src.dataset.generator.base_ollama_generator import (
    IOllamaGenerator,
    GenerationConfig,
    Sample,
    Entity
)
from NER.src.dataset.generator.mistral.utils import build_prompt

import os
import re
import random
from typing import List, Optional, Tuple
from datetime import datetime


class MistralGenerator(IOllamaGenerator):
    """
    Mistral generator implementation.
    """

    def __init__(self,
                 model_name: str = "mistral:7b",
                 known_mountains_file: Optional[str] = None,
        ):
        super().__init__(model_name)
        known_mountains_file = known_mountains_file or "src/dataset/static/mountains.txt"
        self._known_mountains = self._load_known_mountains(known_mountains_file)
        self._clean_text_words = ["mountain", "mount", "mt", "peak", "volcano", "range"]
        self._validate_entity_words = ["river", "lake", "city", "town", "valley",
                                       "island", "ocean", "sea", "forest", "desert"]

    def generate(self, num_examples: int,
                 config: Optional[GenerationConfig] = None
                 ) -> List[Sample]:
        """
        Generate sample sentences.
        Ensures exactly num_examples are returned by retrying if needed.
        """

        if config is None:
            config = GenerationConfig(temperature=0.7, max_tokens=2000)

        all_samples = []
        attempts = 0
        max_attempts = 10

        while len(all_samples) < num_examples and attempts < max_attempts:
            attempts += 1
            remaining = num_examples - len(all_samples)

            prompt = build_prompt(remaining)
            try:
                response_text = self._call_model(prompt, config)
                samples = self._parse_response(response_text)
                if samples:
                    all_samples.extend(samples)
            except Exception as e:
                print(f"Mistral generation failed: {e}")
                continue

        if len(all_samples) > num_examples:
            all_samples = all_samples[:num_examples]
        return all_samples

    def _parse_response(self, response_text: str) -> List[Sample]:
        """
        Parser for accurate entity position finding
        """

        valid_samples = []
        invalid_lines = []

        for line in response_text.split('\n'):
            line = line.strip()

            if not line or line.startswith('```'):
                continue
            if re.match(r'^\d+\.\s*', line):
                line = re.sub(r'^\d+\.\s*', '', line)

            if '||' in line:
                parts = [part.strip() for part in line.split('||')]

                if len(parts) >= 2:
                    sentence, entity_text = parts[:2]

                    if not sentence or not entity_text or not self.validate_entity(entity_text):
                        invalid_lines.append(line)
                        continue

                    if random.choice([True, False]):
                        entity_text, sentence = self._clean_text(entity_text, sentence)

                    start_pos = self._find_entity(sentence, entity_text)

                    if start_pos != -1:
                        end_pos = start_pos + len(entity_text)

                        if sentence[start_pos:end_pos] == entity_text:
                            valid_samples.append(Sample(
                                text=sentence,
                                tags=None,
                                entities=[Entity(
                                    start=start_pos,
                                    end=end_pos,
                                    text=entity_text,
                                    label="MOUNTAIN"
                                )],
                                metadata={
                                    'model': self.model_name,
                                    'timestamp': datetime.now().isoformat(),
                                    'parsing': 'regex_auto'
                                }
                            ))
                        else:
                            invalid_lines.append(line)
                    else:
                        invalid_lines.append(line)

        if len(invalid_lines) > 0:
            fallback_results = self._fallback_parsing(invalid_lines)
            if len(fallback_results) > 0:
                valid_samples.extend(fallback_results)

        return valid_samples

    def validate_entity(self, entity_text: str) -> bool:
        """
        Validate if entity meets criteria
        """
        if not entity_text:
            return False

        entity_lower = entity_text.lower()

        if self._validate_entity_words:
            for blocked_word in self._validate_entity_words:
                if blocked_word.lower() in entity_lower:
                    return False

        return True

    def _clean_text(self, entity_text: str, sentence: str) -> Tuple[str, str]:
        """
        Remove specified words from both entity text and sentence
        """

        cleaned_entity = entity_text.strip()
        cleaned_sentence = sentence

        for word in self._clean_text_words:
            end_pattern = r'\s+' + re.escape(word) + r'[\.!?,]?$'
            if re.search(end_pattern, cleaned_entity, re.IGNORECASE):
                entity_removed = re.sub(end_pattern, '', cleaned_entity, flags=re.IGNORECASE).strip()
                sentence_removed = re.sub(re.escape(cleaned_entity), entity_removed, cleaned_sentence,
                                          flags=re.IGNORECASE)
                cleaned_entity = entity_removed
                cleaned_sentence = sentence_removed

            start_pattern = r'^' + re.escape(word) + r'\s+'
            if re.match(start_pattern, cleaned_entity, re.IGNORECASE):
                entity_removed = re.sub(start_pattern, '', cleaned_entity, flags=re.IGNORECASE).strip()
                sentence_removed = re.sub(re.escape(cleaned_entity), entity_removed, cleaned_sentence,
                                          flags=re.IGNORECASE)
                cleaned_entity = entity_removed
                cleaned_sentence = sentence_removed

        return cleaned_entity, cleaned_sentence

    @staticmethod
    def _find_entity(sentence: str, entity_text: str) -> int:
        """
        Find entity with proper word boundaries
        """

        escaped_entity = re.escape(entity_text)

        pattern = r'\b' + escaped_entity + r'\b'
        match = re.search(pattern, sentence)
        if match:
            return match.start()

        return -1

    def _fallback_parsing(self, invalid_lines: List[str]) -> List[Sample]:
        """
        Fallback parsing for lines that didn't match the expected format
        """

        fallback_samples = []

        for line in invalid_lines:
            for mountain in self._known_mountains.copy():
                if mountain in line:
                    if random.choice([True, False]):
                        mountain, line = self._clean_text(mountain, line)
                    start_pos = line.find(mountain)
                    fallback_samples.append(Sample(
                            text=line,
                            tags=None,
                            entities=[Entity(
                                start=start_pos,
                                end=start_pos + len(mountain),
                                text=mountain,
                                label="MOUNTAIN"
                            )],
                            metadata={
                                'model': self.model_name,
                                'timestamp': datetime.now().isoformat(),
                                'parsing': 'fallback_known_mountain'
                            }
                        )
                    )
        return fallback_samples

    def _load_known_mountains(self, mountains_file: str) -> List[str]:
        """
        Load known mountains from text file
        """

        mountains = []

        try:
            if os.path.exists(mountains_file):
                with open(mountains_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            mountains.append(line)
                print(f"Loaded {len(mountains)} known mountains from {mountains_file}")
            else:
                mountains = self._get_base_mountains_list()
                print(f"Mountains file not found, using base list")
        except Exception as e:
            print(f"Error while loading mountains file: {e}. Using base list")
            mountains = self._get_base_mountains_list()
        return mountains

    @staticmethod
    def _get_base_mountains_list() -> List[str]:
        """
        Fallback mountain list in case file loading fails
        """

        return [
            'Mount Everest', 'K2', 'Matterhorn', 'Denali', 'Mount Fuji',
            'Kilimanjaro', 'Mont Blanc', 'Annapurna', 'Aconcagua', 'Mount Rainier',
            'Himalayas', 'Alps', 'Andes', 'Rocky Mountains', 'Atlas Mountains',
            'Kangchenjunga', 'Lhotse', 'Makalu', 'Cho Oyu', 'Dhaulagiri',
            'Manaslu', 'Nanga Parbat', 'Gasherbrum I', 'Broad Peak', 'Shishapangma',
            'Elbrus', 'Mount Whitney', 'Mount McKinley', 'Ama Dablam', 'Pumori'
        ]