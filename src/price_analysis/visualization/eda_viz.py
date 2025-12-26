"""EDA visualization functions for price analysis."""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

logger: logging.Logger = logging.getLogger(__name__)


def plot_price_by_category(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    order: Sequence[str] | None = None,
    palette: str = "viridis",
    figsize: tuple[int, int] = (12, 7),
) -> plt.Figure:
    """Create box plot with strip plot overlay for price by category.

    Args:
        df: DataFrame with sale_price and category column
        category_col: Column name to group by on x-axis
        title: Plot title
        order: Optional ordering of categories
        palette: Seaborn color palette name
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    df_plot = df[df[category_col].notna() & df["sale_price"].notna()].copy()

    if df_plot.empty:
        logger.warning(f"No valid data for category column '{category_col}'")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    if order is None:
        order = (
            df_plot.groupby(category_col)["sale_price"].median().sort_values(ascending=False).index
        )
    else:
        order = [c for c in order if c in df_plot[category_col].values]

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df_plot,
        x=category_col,
        y=df_plot["sale_price"] / 1000,
        order=order,
        palette=palette,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x=category_col,
        y=df_plot["sale_price"] / 1000,
        order=order,
        color="black",
        alpha=0.2,
        size=3,
        ax=ax,
    )

    ax.set_xlabel(category_col.replace("_", " ").title())
    ax.set_ylabel("Sale Price ($k)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return fig


def plot_price_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "sale_price",
    hue_col: str | None = None,
    hue_order: Sequence[str] | None = None,
    show_regression: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (14, 7),
) -> plt.Figure:
    """Create scatter plot with optional hue coloring and regression line.

    Args:
        df: DataFrame with x, y, and optional hue columns
        x_col: Column for x-axis
        y_col: Column for y-axis (default: sale_price)
        hue_col: Optional column for color grouping
        hue_order: Optional ordering of hue categories
        show_regression: Whether to show overall regression line
        title: Plot title (auto-generated if None)
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    cols_needed = [x_col, y_col]
    if hue_col:
        cols_needed.append(hue_col)

    df_plot = df.dropna(subset=cols_needed).copy()

    if df_plot.empty:
        logger.warning(f"No valid data for columns {cols_needed}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    df_plot["y_k"] = df_plot[y_col] / 1000

    fig, ax = plt.subplots(figsize=figsize)

    scatter_kwargs = {
        "data": df_plot,
        "x": x_col,
        "y": "y_k",
        "alpha": 0.6,
        "s": 50,
        "ax": ax,
    }
    if hue_col:
        scatter_kwargs["hue"] = hue_col
        if hue_order:
            scatter_kwargs["hue_order"] = [h for h in hue_order if h in df_plot[hue_col].values]

    sns.scatterplot(**scatter_kwargs)

    if show_regression:
        sns.regplot(
            data=df_plot,
            x=x_col,
            y="y_k",
            scatter=False,
            color="black",
            line_kws={"linestyle": "--", "label": "Overall trend"},
            ax=ax,
        )

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(f"{y_col.replace('_', ' ').title()} ($k)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")

    if hue_col:
        ax.legend(
            title=hue_col.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left"
        )

    fig.tight_layout()
    return fig


def plot_lowess_curves(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "sale_price",
    title: str | None = None,
    frac: float = 0.3,
    figsize: tuple[int, int] = (16, 6),
) -> plt.Figure:
    """Create side-by-side LOWESS plots on log and linear scales.

    Args:
        df: DataFrame with x and y columns
        x_col: Column for x-axis
        y_col: Column for y-axis (default: sale_price)
        title: Base title for plots (auto-generated if None)
        frac: LOWESS smoothing fraction (0 to 1)
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    df_plot = df.dropna(subset=[x_col, y_col]).copy()

    if df_plot.empty:
        logger.warning(f"No valid data for columns [{x_col}, {y_col}]")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax in axes:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    df_plot = df_plot.sort_values(x_col)
    df_plot["y_k"] = df_plot[y_col] / 1000
    df_plot["log_y"] = np.log(df_plot[y_col])

    base_title = title or f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}"
    x_label = x_col.replace("_", " ").title()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.scatter(df_plot[x_col], df_plot["log_y"], alpha=0.3, s=20, label="Data")
    lowess_fit = lowess(df_plot["log_y"], df_plot[x_col], frac=frac)
    ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], "r-", linewidth=2, label="LOWESS")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Log({y_col.replace('_', ' ').title()})")
    ax.set_title(f"{base_title} (Log Scale)")
    ax.legend()

    ax = axes[1]
    ax.scatter(df_plot[x_col], df_plot["y_k"], alpha=0.3, s=20, label="Data")
    lowess_fit_linear = lowess(df_plot["y_k"], df_plot[x_col], frac=frac)
    ax.plot(lowess_fit_linear[:, 0], lowess_fit_linear[:, 1], "r-", linewidth=2, label="LOWESS")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{y_col.replace('_', ' ').title()} ($k)")
    ax.set_title(f"{base_title} (Linear Scale)")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_price_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    col_order: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    min_groups: int = 2,
) -> plt.Figure:
    """Create heatmap of median prices by two categorical variables.

    Args:
        df: DataFrame with sale_price and category columns
        row_col: Column for heatmap rows
        col_col: Column for heatmap columns
        col_order: Optional ordering of columns
        title: Plot title (auto-generated if None)
        figsize: Figure size tuple
        min_groups: Minimum number of non-null columns to include a row

    Returns:
        Matplotlib Figure object
    """
    df_plot = df.dropna(subset=[row_col, col_col, "sale_price"]).copy()

    if df_plot.empty:
        logger.warning(f"No valid data for columns [{row_col}, {col_col}, sale_price]")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    pivot = df_plot.pivot_table(
        values="sale_price",
        index=row_col,
        columns=col_col,
        aggfunc="median",
    )

    if col_order:
        col_order_valid = [c for c in col_order if c in pivot.columns]
        if col_order_valid:
            pivot = pivot[col_order_valid]

    pivot = pivot.dropna(thresh=min_groups)

    if pivot.empty:
        logger.warning(f"No rows with at least {min_groups} groups after filtering")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"No rows with >= {min_groups} groups",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot / 1000,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Median Price ($k)"},
        ax=ax,
    )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Median Price by {row_col.replace('_', ' ').title()} x {col_col.replace('_', ' ').title()} ($k)"
        )

    ax.set_xlabel(col_col.replace("_", " ").title())
    ax.set_ylabel(row_col.replace("_", " ").title())

    fig.tight_layout()
    return fig


def plot_faceted_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "sale_price",
    facet_col: str = "generation",
    col_wrap: int = 4,
    col_order: Sequence[str] | None = None,
    title: str | None = None,
) -> sns.FacetGrid:
    """Create faceted scatter plots with regression lines per facet.

    Args:
        df: DataFrame with x, y, and facet columns
        x_col: Column for x-axis
        y_col: Column for y-axis (default: sale_price)
        facet_col: Column to facet by
        col_wrap: Number of columns in the facet grid
        col_order: Optional ordering of facet columns
        title: Overall title for the facet grid

    Returns:
        Seaborn FacetGrid object
    """
    df_plot = df.dropna(subset=[x_col, y_col, facet_col]).copy()

    if df_plot.empty:
        logger.warning(f"No valid data for columns [{x_col}, {y_col}, {facet_col}]")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return sns.FacetGrid(pd.DataFrame({x_col: [], y_col: [], facet_col: []}))

    df_plot["y_k"] = df_plot[y_col] / 1000

    if col_order:
        col_order_valid = [c for c in col_order if c in df_plot[facet_col].values]
    else:
        col_order_valid = None

    g = sns.FacetGrid(
        df_plot,
        col=facet_col,
        col_wrap=col_wrap,
        height=4,
        aspect=1.2,
        col_order=col_order_valid,
    )
    g.map_dataframe(
        sns.regplot,
        x=x_col,
        y="y_k",
        scatter_kws={"alpha": 0.5, "s": 30},
        line_kws={"color": "red"},
    )
    g.set_axis_labels(x_col.replace("_", " ").title(), f"{y_col.replace('_', ' ').title()} ($k)")
    g.set_titles("{col_name}")

    if title:
        g.figure.suptitle(title, y=1.02, fontsize=14)

    g.figure.tight_layout()
    return g
