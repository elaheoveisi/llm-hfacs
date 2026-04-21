
    os.makedirs(path, exist_ok=True)


SEVERE_THRESHOLD = 4.0


def _binarize(weights: dict) -> dict:
    return {col: (2 if w >= SEVERE_THRESHOLD else 1) for col, w in weights.items()}


def make_five_class_target(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """5-class target:
      0 = Neither
      1 = Pure Error        (only error indicators active)
      2 = Pure Violation    (only violation indicators active)
      3 = Mixed Error-dominant   (both active, error wins)
      4 = Mixed Violation-dominant (both active, violation wins)

    Mixed resolution order:
      1) count of active indicators  -> higher wins
      2) weighted sum (binary * {1=not-severe, 2=severe}) -> higher wins
      3) tie_break config param ('error' -> class 3, 'violation' -> class 4)
    """
    hfacs = config_yaml['hfacs_categories']
    error_weights: dict = _binarize(hfacs['Error'])
    viol_weights: dict = _binarize(hfacs['Violation'])
