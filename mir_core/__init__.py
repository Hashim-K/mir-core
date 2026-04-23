"""mir-core: shared Python package for MSc thesis MIR project.

Provides model definitions, preprocessing pipelines, dataset loaders,
evaluation metrics, postprocessing (DBN + particle filter), and training
utilities for beat, downbeat, and tempo tracking research.
"""

__version__ = "0.2.0"

# Re-export top-level API for convenience.
# Prefer direct submodule imports for AI-friendly context efficiency:
#   from mir_core.models.beatnet.crnn import BeatNetCRNN

from .utils.hashing import canonical_json, stable_digest, stable_hash

try:
    from .models import (
        BeatNetCRNN, BeatNetBatch, BeatNetCRNNBatch, MultiHeadBeatNet,
        ResBlock, TCN, BockTCN,
        BEAST,
        GenreClassifier, GenreRouter, GENRE_LABELS,
    )
except ModuleNotFoundError:
    BeatNetCRNN = None
    BeatNetBatch = None
    BeatNetCRNNBatch = None
    MultiHeadBeatNet = None
    ResBlock = None
    TCN = None
    BockTCN = None
    BEAST = None
    GenreClassifier = None
    GenreRouter = None
    GENRE_LABELS = []

try:
    from .hub import load_model, list_models, ModelType, TrainingMethod, ModelSpec
except ModuleNotFoundError:
    load_model = None
    list_models = None
    ModelType = None
    TrainingMethod = None
    ModelSpec = None

try:
    from .preprocessing import (
        PreProcessor,
        BeatNetPreProcessor,
        FPS,
        NUM_BANDS,
        FFT_SIZE,
        MASK_VALUE,
    )
except ModuleNotFoundError:
    PreProcessor = None
    BeatNetPreProcessor = None
    FPS = 100
    NUM_BANDS = 12
    FFT_SIZE = 2048
    MASK_VALUE = -1

try:
    from .postprocessing import DBNBeatTracker, ParticleFilterTracker, detect_beats
except ModuleNotFoundError:
    DBNBeatTracker = None
    ParticleFilterTracker = None
    detect_beats = None
