"""mir-core: shared Python package for MSc thesis MIR project.

Provides model definitions, preprocessing pipelines, dataset loaders,
evaluation metrics, postprocessing (DBN + particle filter), and training
utilities for beat, downbeat, and tempo tracking research.
"""

__version__ = "0.2.0"

# Re-export top-level API for convenience.
# Prefer direct submodule imports for AI-friendly context efficiency:
#   from mir_core.models.beatnet.crnn import BeatNetCRNN

from .models import (
    BeatNetCRNN, BeatNetBatch, BeatNetCRNNBatch, MultiHeadBeatNet,
    ResBlock, TCN, BockTCN,
    BEAST, BEASTBatch,
    GenreClassifier, GenreRouter, GENRE_LABELS,
)
from .preprocessing import PreProcessor, BeatNetPreProcessor, FPS, NUM_BANDS, FFT_SIZE, MASK_VALUE
from .postprocessing import DBNBeatTracker, ParticleFilterTracker, detect_beats
from .hub import load_model, list_models, ModelType, TrainingMethod, ModelSpec
