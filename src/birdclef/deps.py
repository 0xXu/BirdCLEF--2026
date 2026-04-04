from typing import Iterable


try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import librosa
    import librosa.display
except ModuleNotFoundError:
    librosa = None

try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    gridspec = None
    plt = None

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

try:
    import timm
except ModuleNotFoundError:
    timm = None

try:
    import torchaudio.transforms as T
except ModuleNotFoundError:
    T = None

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError:
    roc_auc_score = None
    StratifiedKFold = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Iterable | None = None, **_: object):
        return iterable


def require_dependencies(*deps: tuple[str, object]) -> None:
    missing = [name for name, module in deps if module is None]
    if missing:
        raise ModuleNotFoundError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install them with `uv sync` before running this command."
        )
