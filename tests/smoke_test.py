"""Load the packaged models and predict all bundled CSV samples."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
app = runpy.run_path(str(root / "WineRating/app.py"))
for sample in sorted((root / "WineRating/File_data_test").glob("*.csv")):
    result = app["predict_grouped"](app["preprocess"](pd.read_csv(sample)))
    assert len(result) > 0, sample.name
    assert np.isfinite(result["PredictedRating"]).all(), sample.name
    assert set(result["ReliabilityGroup"]) == {"Low", "Medium", "High"}, sample.name
    print(f"{sample.name}: {len(result)} predictions OK")
