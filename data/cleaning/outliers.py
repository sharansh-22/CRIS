import polars as pl
import logging
from dataclasses import dataclass, field

@dataclass(slots=True)
class OutlierHandlingResult:
    dataframe: pl.DataFrame
    method: str
    diagnostics: dict[str, object] = field(default_factory=dict)

logger = logging.getLogger(__name__)

def handle_outliers(df: pl.DataFrame, columns: list[str], method: str = "none") -> OutlierHandlingResult:
    """Handle outliers in the specified numeric columns using polars expressions."""
    if df.is_empty() or method == "none":
        return OutlierHandlingResult(dataframe=df, method=method)

    diagnostics: dict[str, object] = {"columns": columns, "method": method, "adjusted_columns": []}

    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return OutlierHandlingResult(dataframe=df, method=method, diagnostics=diagnostics)

    for col in valid_cols:
        dtype = df.schema[col]
        if not dtype.is_numeric():
            raise TypeError(f"CRIS Constitution Violation: Column '{col}' is not numeric (found {dtype}). Silent coercion is forbidden.")

    if method == "clip":
        exprs = []
        for col in valid_cols:
            lower = df.select(pl.col(col).quantile(0.01)).item()
            upper = df.select(pl.col(col).quantile(0.99)).item()
            if lower is not None and upper is not None:
                exprs.append(pl.col(col).clip(lower_bound=lower, upper_bound=upper))
                diagnostics["adjusted_columns"].append({"column": col, "lower": float(lower), "upper": float(upper)})
        
        if exprs:
            df = df.with_columns(exprs)

    elif method == "winsorize":
        exprs = []
        for col in valid_cols:
            lower = df.select(pl.col(col).quantile(0.05)).item()
            upper = df.select(pl.col(col).quantile(0.95)).item()
            if lower is not None and upper is not None:
                exprs.append(pl.col(col).clip(lower_bound=lower, upper_bound=upper))
                diagnostics["adjusted_columns"].append({"column": col, "lower": float(lower), "upper": float(upper)})
        
        if exprs:
            df = df.with_columns(exprs)

    elif method == "robust_zscore_filter":
        for col in valid_cols:
            median_val = df.select(pl.col(col).median()).item()
            if median_val is None:
                continue
                
            mad_val = df.select((pl.col(col) - median_val).abs().median()).item()
            if mad_val is None or mad_val == 0.0:
                continue
                
            robust_z = 0.6745 * (pl.col(col) - median_val) / mad_val
            df = df.filter(robust_z.abs() <= 3.5)
            diagnostics["adjusted_columns"].append({"column": col, "median": float(median_val), "mad": float(mad_val)})

    elif method == "reject":
        raise ValueError(f"outlier rejection requested for column {columns}; configure a clipping policy instead")
    else:
        logger.warning(f"Unknown outlier method {method}. Returning original data.")
    
    return OutlierHandlingResult(dataframe=df, method=method, diagnostics=diagnostics)

