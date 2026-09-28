"""Plot the time series in every CSV under the datasets folder.

Usage:
    python scripts/plot_datasets.py                       # all CSVs -> figures/datasets/
    python scripts/plot_datasets.py --max-series 5        # limit channels per figure
    python scripts/plot_datasets.py --pattern "ETT-small/*.csv" --last 2000
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_time(col: pd.Series) -> pd.DatetimeIndex:
    sample = str(col.iloc[0])
    # e.g. "01.01.2020 00:00" (day-first)
    dayfirst = bool(re.match(r"^\d{2}\.\d{2}\.\d{4}", sample))
    return pd.DatetimeIndex(pd.to_datetime(col, format="mixed", dayfirst=dayfirst))


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV as a wide DataFrame indexed by time (one column per series)."""
    df = pd.read_csv(path, encoding_errors="replace")
    time_col = df.columns[0]
    df[time_col] = parse_time(df[time_col])

    # Long format (e.g. caiso: Date, load, zone) -> pivot to wide
    cat_cols = [c for c in df.columns[1:] if df[c].dtype == object]
    if cat_cols:
        value_cols = [c for c in df.columns[1:] if c not in cat_cols]
        df = df.pivot_table(index=time_col, columns=cat_cols[0], values=value_cols[0], aggfunc="mean")
        df.columns = [str(c) for c in df.columns]
    else:
        df = df.set_index(time_col)

    return df.sort_index().apply(pd.to_numeric, errors="coerce")


def plot_dataset(df: pd.DataFrame, title: str, out_path: Path, max_series: int, last: int | None):
    if last:
        df = df.iloc[-last:]
    cols = list(df.columns)
    shown = cols[:max_series]

    fig, axes = plt.subplots(len(shown), 1, figsize=(14, 1.6 * len(shown) + 0.8), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], shown):
        ax.plot(df.index, df[c], lw=0.6, color="tab:blue")
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    suffix = f" (showing {len(shown)}/{len(cols)} series)" if len(cols) > len(shown) else ""
    fig.suptitle(f"{title}  [{len(df)} steps, {df.index[0]} – {df.index[-1]}]{suffix}", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--pattern", default="**/*.csv", help="glob relative to data-dir")
    parser.add_argument("--out-dir", type=Path, default=Path("figures/datasets"))
    parser.add_argument("--max-series", type=int, default=8, help="max channels plotted per dataset")
    parser.add_argument("--last", type=int, default=None, help="plot only the last N time steps")
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob(args.pattern))
    if not paths:
        print(f"No CSV found: {args.data_dir / args.pattern}")
        return

    for path in paths:
        rel = path.relative_to(args.data_dir).with_suffix("")
        try:
            df = load_csv(path)
            out_path = args.out_dir / f"{str(rel).replace('/', '__')}.png"
            plot_dataset(df, str(rel), out_path, args.max_series, args.last)
            print(f"[ok] {path} -> {out_path}  shape={df.shape}")
        except Exception as e:
            print(f"[skip] {path}: {e}")


if __name__ == "__main__":
    main()
