#!/usr/bin/env python3

import pandas as pd


CSV_PATH = "/Users/ribas/clones/BigData_ForecastHackathon/sales_predictions.csv"
PARQUET_PATH = "/Users/ribas/clones/BigData_ForecastHackathon/sales_predictions.parquet"


def main() -> None:
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    df.to_parquet(PARQUET_PATH, engine="fastparquet", index=False)
    print(f"Wrote {PARQUET_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()


