import pandas as pd
from typing import List, Optional, Union

def fix_stringified_date_lists(df, column="dates"):
    """
    Converts stringified Python lists into real lists.
    Example:
      "['12 March 2023', '9 Jan 2023']"
    becomes:
      ['12 March 2023', '9 Jan 2023']
    """
    import ast
    df[column] = df[column].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )
    return df


def deduplicate_date_lists(df, column="dates"):
    """
    Removes duplicate dates within each list.
    """
    df[column] = df[column].apply(
        lambda lst: list(dict.fromkeys(lst)) if isinstance(lst, list) else lst
    )
    return df



def parse_date_list(date_list: Union[List[str], float]) -> Optional[List[pd.Timestamp]]:
    """
    Convert a list of date strings into a list of pandas Timestamp objects.
    Handles NaN, non-list values, and parsing errors.

    Parameters
    ----------
    date_list : list of str or NaN
        The raw list of date strings.

    Returns
    -------
    list of pd.Timestamp or None
    """
    if isinstance(date_list, list):
        return pd.to_datetime(date_list, errors="coerce").tolist()
    return None


def add_date_features(df: pd.DataFrame, date_column: str = "dates") -> pd.DataFrame:
    """
    Given a DataFrame with a column containing lists of date strings,
    this function:
      - Parses each list into datetime objects
      - Extracts earliest and latest dates per row
      - Adds two new columns: 'earliest_date' and 'latest_date'

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the raw date lists.
    date_column : str
        The column name of the list-of-dates field.

    Returns
    -------
    pd.DataFrame
        The DataFrame enriched with parsed date columns.
    """

    # Parse date lists into datetime objects
    df["dates_parsed"] = df[date_column].apply(parse_date_list)

    # Extract earliest & latest per row
    df["earliest_date"] = df["dates_parsed"].apply(
        lambda lst: min(lst) if isinstance(lst, list) and len(lst) > 0 else pd.NaT
    )
    df["latest_date"] = df["dates_parsed"].apply(
        lambda lst: max(lst) if isinstance(lst, list) and len(lst) > 0 else pd.NaT
    )

    return df


def extract_latest_review(df):
    return df["latest_date"].max()

def extract_oldest_review(df):
    return df["earliest_date"].min()


def sort_by_latest(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    Sort the DataFrame by latest_date.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with a 'latest_date' column.
    ascending : bool
        Sort newest-to-oldest (False) or oldest-to-newest (True).

    Returns
    -------
    pd.DataFrame
    """
    return df.sort_values("latest_date", ascending=ascending)


def sort_by_earliest(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """
    Sort the DataFrame by earliest_date.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with an 'earliest_date' column.
    ascending : bool
        Sort oldest-to-newest (True) or newest-to-oldest (False).

    Returns
    -------
    pd.DataFrame
    """
    return df.sort_values("earliest_date", ascending=ascending)


def review_count(df):
    """
    For stringified lists or real lists, get how many dates exist.
    """
    return df["dates"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)

# Now add it to DataFrame:
def add_review_count(df):
    df["review_count"] = review_count(df)
    return df


def add_review_period(df):
    """
    Adds a new column showing the total review span in days:
    latest_date - earliest_date
    """
    df["review_period_days"] = (df["latest_date"] - df["earliest_date"]).dt.days
    return df


def explode_dates(df):
    """
    Turn each restaurant’s list of dates into a long format
    This is CRUCIAL for plotting time series or histograms.
    Converts:
        row = 1, dates = [d1, d2, d3]
    into:
        3 rows: (id, d1), (id, d2), (id, d3)
    """
    df_exploded = df.explode("dates_parsed").reset_index(drop=True)
    df_exploded = df_exploded[df_exploded["dates_parsed"].notna()]
    df_exploded.rename(columns={"dates_parsed": "review_date"}, inplace=True)
    return df_exploded


import matplotlib.pyplot as plt

def plot_review_history(df, name_column="name", restaurant_name=None):
    """
    Plot all review dates for a given restaurant (scatter plot).
    """
    df_ex = explode_dates(df)

    if restaurant_name:
        df_ex = df_ex[df_ex[name_column] == restaurant_name]

    plt.figure(figsize=(10, 4))
    plt.scatter(df_ex["review_date"], [1] * len(df_ex), alpha=0.5)
    plt.title(f"Review History for {restaurant_name}")
    plt.yticks([])
    plt.xlabel("Date")
    plt.show()


def plot_review_trend(df):
    '''
    Plot overall review volume over time
    This reveals if the restaurant is trending up or down.
    '''
    df_ex = explode_dates(df)

    df_daily = df_ex.groupby("review_date").size()

    plt.figure(figsize=(12, 5))
    plt.plot(df_daily.index, df_daily.values)
    plt.title("Total Review Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Reviews per Day")
    plt.show()
