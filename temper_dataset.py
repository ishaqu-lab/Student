from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "default_dataset.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(42)
    for category in [
        "Teacher Feedback", "Course Content", "Examination Pattern",
        "Laboratory", "Library Facilities"
    ]:
        for source, target in [
            ("Positive", "Neutral"),
            ("Neutral", "Negative"),
            ("Negative", "Positive"),
        ]:
            idx = (df[(df["Category"] == category) & (df["Sentiment"] == source)]
                   .sample(8, random_state=int(rng.integers(0, 2**32 - 1))).index)
            df.loc[idx, "Sentiment"] = target
    df.to_csv(DATA_PATH, index=False)
    print("Applied the controlled 4% annotation-ambiguity adjustment.")

if __name__ == "__main__":
    main()
