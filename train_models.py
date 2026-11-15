"""
Training entrypoint for Multimodal Thyroid Screening ML pipeline.
Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)
Delegates directly to canonical src/model_trainer.py to guarantee 100% artifact synchronization.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.model_trainer import train_and_evaluate_models

if __name__ == "__main__":
    train_and_evaluate_models()
