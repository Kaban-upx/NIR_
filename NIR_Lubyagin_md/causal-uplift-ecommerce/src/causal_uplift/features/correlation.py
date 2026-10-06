"""Контроль мультиколлинеарности (W4, SPEC §5.3).

Строит треугольную матрицу парных корреляций, отбирает признаки к исключению
при ``|r| > corr_threshold`` и выгружает артефакты для ВКР:
``reports/corr_matrix.png`` и ``reports/dropped_features.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.io_utils import save_json
from causal_uplift.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationReport:
    """Результат анализа мультиколлинеарности."""

    matrix: pd.DataFrame
    dropped: list[dict[str, object]] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    threshold: float = 0.95
    method: str = "pearson"

    @property
    def dropped_names(self) -> list[str]:
        return [str(d["feature"]) for d in self.dropped]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "n_features_before": len(self.kept) + len(self.dropped),
            "n_features_after": len(self.kept),
            "dropped": self.dropped,
            "kept": self.kept,
        }

    def top_pairs(self, n: int = 15) -> pd.DataFrame:
        """Самые скоррелированные пары — таблица для отчёта."""
        corr = self.matrix.abs()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = (
            corr.where(mask)
            .stack()
            .sort_values(ascending=False)
            .head(n)
            .rename("abs_r")
            .reset_index()
        )
        pairs.columns = ["feature_a", "feature_b", "abs_r"]
        return pairs


def correlation_matrix(
    df: pd.DataFrame, columns: Sequence[str], *, method: str = "pearson"
) -> pd.DataFrame:
    """Матрица парных корреляций по числовым колонкам."""
    numeric = df[list(columns)].select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        msg = "для матрицы корреляций нужно минимум два числовых признака"
        raise ValueError(msg)
    return numeric.corr(method=method)  # type: ignore[arg-type]


def select_decorrelated(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    threshold: float = 0.95,
    method: str = "pearson",
    protected: Sequence[str] = (),
) -> CorrelationReport:
    """Отобрать признаки, убрав по одному из каждой сильно скоррелированной пары.

    Из пары исключается признак с большей средней абсолютной корреляцией с
    остальными (он несёт меньше уникальной информации). ``protected``
    никогда не исключается — так сохраняются признаки, нужные содержательно
    (например ``age``).
    """
    corr = correlation_matrix(df, columns, method=method)
    abs_corr = corr.abs()
    mean_corr = abs_corr.mean(axis=1)

    protected_set = set(protected)
    # Порядок обхода: признаки с меньшей средней корреляцией рассматриваются
    # первыми, т.е. при конфликте выживает более «независимый».
    order = sorted(corr.columns, key=lambda c: (c not in protected_set, float(mean_corr[c]), c))

    kept: list[str] = []
    dropped: list[dict[str, object]] = []
    for feature in order:
        conflict: tuple[str, float] | None = None
        for other in kept:
            r = float(abs_corr.loc[feature, other])
            if r > threshold:
                conflict = (other, float(corr.loc[feature, other]))
                break
        if conflict is None or feature in protected_set:
            kept.append(feature)
        else:
            dropped.append(
                {
                    "feature": feature,
                    "correlated_with": conflict[0],
                    "r": round(conflict[1], 6),
                    "mean_abs_corr": round(float(mean_corr[feature]), 6),
                }
            )

    # Порядок колонок в витрине сохраняем исходным — удобнее читать отчёты
    kept_ordered = [c for c in corr.columns if c in set(kept)]
    if dropped:
        logger.info(
            "мультиколлинеарность: исключено %d признаков при |r| > %.2f (%s)",
            len(dropped),
            threshold,
            ", ".join(str(d["feature"]) for d in dropped[:8]),
        )
    else:
        logger.info("мультиколлинеарность: при |r| > %.2f исключать нечего", threshold)

    return CorrelationReport(
        matrix=corr,
        dropped=dropped,
        kept=kept_ordered,
        threshold=threshold,
        method=method,
    )


def plot_correlation_matrix(
    corr: pd.DataFrame,
    path: str | Path,
    *,
    title: str = "Треугольная матрица парных корреляций ковариат",
    annotate_threshold: float | None = None,
) -> Path:
    """Сохранить треугольную матрицу корреляций в PNG (артефакт для ВКР)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = corr.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    data = np.ma.masked_where(mask, corr.to_numpy())

    size = max(7.0, min(0.38 * n, 20.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick_size = 7 if n > 35 else 9
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=tick_size)
    ax.set_yticklabels(corr.index, fontsize=tick_size)
    ax.set_title(f"{title}\n(n = {n} признаков)", fontsize=11)

    if annotate_threshold is not None:
        # Подсветить пары выше порога: это и есть кандидаты на исключение
        strong = (np.abs(corr.to_numpy()) > annotate_threshold) & ~mask
        np.fill_diagonal(strong, False)
        for i, j in zip(*np.where(strong), strict=True):
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2
                )
            )

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="коэффициент корреляции")
    fig.tight_layout()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    logger.info("матрица корреляций: %s", target)
    return target


def save_correlation_artifacts(
    report: CorrelationReport,
    *,
    png_path: str | Path,
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> dict[str, Path]:
    """Выгрузить PNG, список исключённых признаков и (опционально) саму матрицу."""
    paths = {
        "png": plot_correlation_matrix(
            report.matrix, png_path, annotate_threshold=report.threshold
        ),
        "json": save_json(report.as_dict(), json_path),
    }
    if csv_path is not None:
        target = Path(csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        report.matrix.to_csv(target, encoding="utf-8")
        paths["csv"] = target
    return paths


__all__ = [
    "CorrelationReport",
    "correlation_matrix",
    "plot_correlation_matrix",
    "save_correlation_artifacts",
    "select_decorrelated",
]
