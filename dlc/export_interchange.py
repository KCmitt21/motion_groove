"""Convert a DLC multi-index CSV into the pipeline's stable long CSV schema."""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    source = pd.read_csv(args.input, header=[0, 1, 2, 3], index_col=0)
    rows = []
    for frame_idx, row in source.iterrows():
        # Expected levels: scorer, individuals, bodyparts, coords.
        for instrument in ("guitar", "bass"):
            for point in ("body_center", "bridge", "neck_joint", "nut", "head_tip"):
                matches = [column for column in source.columns if instrument in column and point in column]
                by_coord = {column[-1]: row[column] for column in matches}
                rows.append({"frame_idx": int(frame_idx), "instrument": instrument, "keypoint": point,
                             "x": by_coord.get("x"), "y": by_coord.get("y"),
                             "score": by_coord.get("likelihood")})
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

