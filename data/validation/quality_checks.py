import pandas as pd
from typing import List

def check_schema_match(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Check if the dataframe matches the expected schema."""
    if len(df.columns) != len(expected_columns):
        return False
    
    for c1, c2 in zip(df.columns, expected_columns):
        if c1 != c2:
            return False
            
    return True
