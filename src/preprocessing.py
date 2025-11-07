import pandas as pd
from typing import List, Optional, Union

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
