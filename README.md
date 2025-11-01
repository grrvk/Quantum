# Quantum

## Projects

### Imatch
Satellite image matching system for feature correspondence detection.

- Pre-trained baseline models: **LoFTR** and **LightGlue**
- Trainable **SuperPoint** detectors with two approaches:
  - Self-matching: Unsupervised training on augmented views
  - Polygon keypoints: Supervised training from GeoJSON annotations
- End-to-end inference pipeline for feature matching

**See `Imatch/README.md` for detailed documentation.**

### NER
Named Entity Recognition system for extracting mountain entities from text.

- Fine-tuned **DistilBERT** model for NER tasks
- Synthetic dataset generation with **Mistral** via Ollama
- Training and inference workflows with evaluation metrics

**See `NER/README.md` for detailed documentation.**

