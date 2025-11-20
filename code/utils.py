# ============================================
# utils.py — shared loader + cleaning functions
# For data mining / ARIMAX / forecasting projects
# ============================================

import pandas as pd
import numpy as np


# ----------------------------------------------------
# 1. FRED-style loader (DATE + value)
# ----------------------------------------------------
def load_fred(path, value_name="value"):
    """
    Load FRED-style time series:
    Columns usually like: DATE, PPIACO or DATE, CPIAUCSL
    """
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Identify date column
    if "date" not in df.columns:
        raise ValueError(f"No DATE column found in {path}")

    # If dataset has 2 columns: date + value
    value_col = [col for col in df.columns if col != "date"][0]

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.rename(columns={value_col: value_name})

    df = df.set_index("DATE").sort_index()
    return df[[value_name]]

import pandas as pd

def read_fred_monthly_to_quarterly_mean(path, value_col=None, dropna=True):

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Ensure date column exists
    if "date" not in df.columns:
        raise ValueError(f"No 'date' column found in {path}")

    # Auto-detect value column
    if value_col is None:
        non_date_cols = [c for c in df.columns if c != "date"]
        if not non_date_cols:
            raise ValueError(f"No value column found in {path}")
        value_col = non_date_cols[0]

    # Convert to proper types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop missing values if wanted
    if dropna:
        df = df.dropna(subset=[value_col])

    # Set index
    df = df.set_index("date").sort_index()

    # Monthly → Quarterly mean
    quarterly = df[value_col].resample("Q").mean().reset_index()

    return quarterly




# ----------------------------------------------------
# 2. Monthly wide → long → quarterly converter
# (Year, Jan, Feb, Mar, ...)
# ----------------------------------------------------
def monthly_wide_to_quarterly(path, value_name="value"):
    """
    Converts a typical BLS/BEA wide file into long format, then quarterly.
    """

    df = pd.read_csv(path)

    # Melt wide → long
    df_long = df.melt(id_vars="Year", var_name="Month", value_name=value_name)

    # Clean blanks
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
    df_long = df_long.dropna(subset=[value_name])

    # Convert month abbreviations → number
    df_long["Month"] = pd.to_datetime(df_long["Month"], format="%b").dt.month

    # Build date column
    df_long["date"] = pd.to_datetime(
        df_long[["Year", "Month"]].assign(day=1)
    )

    df_long = df_long.set_index("date").sort_index()

    # Convert monthly → quarterly mean
    quarterly = df_long[[value_name]].resample("Q").mean()

    return quarterly


# ----------------------------------------------------
# 3. Already-tidy CSV loader (has 'date' column)
# ----------------------------------------------------
def load_simple_series(path, value_name="value"):
    """
    Loads files where there is already a single date column and a single value column.
    """
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"Expected a date column in: {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Identify value column
    value_cols = [c for c in df.columns if c != "date"]
    if len(value_cols) != 1:
        raise ValueError("Cannot determine value column automatically.")

    df = df.rename(columns={value_cols[0]: value_name})

    return df[[value_name]]


# ----------------------------------------------------
# 4. Universal loader — auto-detects file type
# ----------------------------------------------------
def load_series(path, value_name="value", freq="auto"):
    """
    Automatically detects which loader to use.
    - FRED format (DATE + 1 value)
    - Wide monthly format (Year + months)
    - Simple tidy series (date + value)

    freq = "auto" | "monthly" | "quarterly"
    """

    head = pd.read_csv(path, nrows=5)
    cols = head.columns

    # FRED: DATE + Something
    if "DATE" in cols or "date" in cols:
        return load_fred(path, value_name)

    # WIDE monthly: Year + Jan, Feb, Mar...
    if "Year" in cols and any(m in cols for m in ["Jan", "Feb", "Mar"]):
        if freq == "quarterly":
            return monthly_wide_to_quarterly(path, value_name)
        else:
            # Return monthly instead of quarterly
            df_q = monthly_wide_to_quarterly(path, value_name)
            return df_q.resample("M").pad()

    # Already tidy: date + value
    if "date" in cols:
        return load_simple_series(path, value_name)

    raise ValueError(f"Cannot detect file type for: {path}")


# ----------------------------------------------------
# 5. Common transformations for macro models
# ----------------------------------------------------
def pct_change(df, periods=1):
    """Percent change helper with clean column handling."""
    name = df.columns[0] + "_pct"
    return df.pct_change(periods=periods).rename(columns={df.columns[0]: name})


def log_diff(df, periods=1):
    """Log difference (good for ARIMA regressors)."""
    col = df.columns[0]
    return np.log(df[col]).diff(periods=periods).to_frame(col + "_logdiff")


def to_quarterly(df):
    """Force quarterly frequency."""
    return df.resample("Q").mean()


def to_monthly(df):
    """Force monthly frequency (forward-fill)."""
    return df.resample("M").pad()


# ----------------------------------------------------
# 6. Combined loader for ARIMAX data bundles
# ----------------------------------------------------
def load_for_arimax(path, value_name="value"):
    """
    Automatically loads → monthly → logdiff → quarterly (typical ARIMAX cleaning)
    """
    raw = load_series(path, value_name=value_name)

    # Convert to monthly/quarterly consistency
    monthly = to_monthly(raw)
    q = to_quarterly(monthly)

    # Stationary transform
    stationary = log_diff(q)

    return q, stationary


# ----------------------------------------------------
# Clean wide → long → quarterly converter
# (Year, Jan, Feb, ..., Dec)
# ----------------------------------------------------
def read_wide_monthly_to_quarterly(path, value_name="value"):
    import pandas as pd

    df = pd.read_csv(path)

    # Melt wide to long (Year, Month → value)
    df_long = df.melt(id_vars="Year", var_name="Month", value_name=value_name)

    # Clean blanks
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
    df_long = df_long.dropna(subset=[value_name])

    # Convert month abbreviations → month number
    df_long["Month"] = pd.to_datetime(df_long["Month"], format="%b").dt.month

    # Build date index
    df_long["date"] = pd.to_datetime(
        df_long[["Year", "Month"]].assign(day=1)
    )

    df_long = df_long.set_index("date").sort_index()

    # Convert monthly → quarterly mean
    quarterly = df_long[[value_name]].resample("Q").mean().reset_index()

    return quarterly

def read_monthly_flow_to_quarterly_sum(path, value_name="value"):

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Auto-detect date column
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError(f"No date column in {path}")

    # Auto-detect value column
    value_cols = [c for c in df.columns if c not in [date_col]]
    if len(value_cols) != 1:
        raise ValueError("Cannot auto-detect value column.")
    raw_value_col = value_cols[0]

    # Parse date
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Clean numeric values
    df[raw_value_col] = pd.to_numeric(df[raw_value_col], errors="coerce")

    # Remove missing
    df = df.dropna(subset=[raw_value_col])

    # Index and sort
    df = df.set_index("date").sort_index()

    # Monthly → Quarterly SUM (because these are flow variables)
    df_q = df[raw_value_col].resample("Q").sum().reset_index()

    # Rename
    df_q = df_q.rename(columns={raw_value_col: value_name})

    return df_q[["date", value_name]]


def read_monthly_level_to_quarterly_mean(path, value_name="value"):
    """
    For tidy MONTHLY level/index series with:
        - observation_date  or  date
        - one numeric value column

    Returns quarterly MEAN (QE).
    Good for INDPRO, CPI-like indices, utilization, etc.
    """
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    date_col = "observation_date" if "observation_date" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError(f"No observation_date/date column found in {path}")

    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) != 1:
        raise ValueError("Cannot auto-detect value column.")
    raw_col = value_cols[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
    df = df.dropna(subset=[raw_col])
    df = df.set_index("date").sort_index()

    df_q = df[raw_col].resample("QE").mean().reset_index()
    df_q = df_q.rename(columns={raw_col: value_name})

    return df_q[["date", value_name]]


def read_census_period_value_excel_to_quarterly_sum(path, sheet_name="CIDR",
                                                    skiprows=7, value_name="value"):
    """
    For Census XLSX files with:
        - header row: Period, Value
        - monthly data like 'Jan-1959', 591

    Returns quarterly SUM (QE).
    Good for:
      - New Residential Construction (NRC.xlsx)
      - New Home Sales (SeriesReport-*.xlsx)
    """
    import pandas as pd

    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows)

    # Fix column names
    df.columns = [c.strip() for c in df.columns]

    # Validate columns
    if "Period" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"Expected 'Period' and 'Value' columns. Found: {df.columns}")

    df = df.rename(columns={"Period": "period", "Value": value_name})

    # Parse date
    df["date"] = pd.to_datetime(df["period"], format="%b-%Y", errors="coerce")

    # Clean numeric
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    df = df.dropna(subset=["date", value_name])
    df = df.set_index("date").sort_index()

    # Quarterly SUM (flow variable)
    df_q = df[value_name].resample("QE").sum().reset_index()

    return df_q[["date", value_name]]
