"""Тесты схем и контрактов данных (ETL-1, ETL-6).

Проверяется главное свойство карантина: некорректная строка не роняет прогон
и не попадает в расчёт — она уходит в отдельный файл с причиной.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.etl.schema import (
    CLIENTS_SCHEMA,
    FEATURE_STORE_RESERVED,
    PURCHASES_SCHEMA,
    UPLIFT_TRAIN_SCHEMA,
    FeatureStoreContract,
    cast_to_schema,
    covariate_columns,
    schema_summary,
)


def _purchases_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["c1", "c2", "c3", "c4"],
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "transaction_datetime": pd.to_datetime(
                ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
            ),
            "regular_points_received": [1.0, 0.0, 2.0, 1.0],
            "express_points_received": [0.0, 0.0, 0.0, 0.0],
            "regular_points_spent": [0.0, -5.0, 0.0, 0.0],
            "express_points_spent": [0.0, 0.0, 0.0, 0.0],
            "purchase_sum": [100.0, -20.0, 300.0, np.nan],
            "store_id": ["s1", "s1", "s2", "s2"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "product_quantity": [1.0, 1.0, 0.0, 2.0],
            "trn_sum_from_iss": [100.0, 0.0, 300.0, 0.0],
            "trn_sum_from_red": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_quarantine_separates_invalid_rows() -> None:
    valid, bad = PURCHASES_SCHEMA.validate_rows(_purchases_frame())

    # c1 валидна; c2 — отрицательная сумма; c3 — нулевое количество; c4 — NaN в сумме
    assert valid["client_id"].tolist() == ["c1"]
    assert sorted(bad["client_id"].tolist()) == ["c2", "c3", "c4"]
    assert "_quarantine_reason" in bad.columns
    assert bad["_quarantine_reason"].str.len().gt(0).all()
    # Причина называет конкретную колонку — иначе отчёт по карантину бесполезен
    assert "purchase_sum" in bad.loc[bad["client_id"] == "c2", "_quarantine_reason"].item()


def test_quarantine_reason_names_missing_required_value() -> None:
    df = _purchases_frame().head(1).assign(store_id=[None])
    valid, bad = PURCHASES_SCHEMA.validate_rows(df)
    assert valid.empty
    assert bad["_quarantine_reason"].item() == "пустое значение в store_id"


def test_nullable_column_allows_na_but_not_bad_value() -> None:
    df = pd.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "first_issue_date": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "first_redeem_date": pd.to_datetime([None, "2018-05-01"]),
            "age": [np.nan, 5.0],  # NaN допустим (nullable), 5 лет — нет
            "gender": ["F", "M"],
        }
    )
    valid, bad = CLIENTS_SCHEMA.validate_rows(df)
    assert valid["client_id"].tolist() == ["c1"]
    assert bad["client_id"].tolist() == ["c2"]


def test_unknown_gender_goes_to_quarantine() -> None:
    df = pd.DataFrame(
        {
            "client_id": ["c1"],
            "first_issue_date": pd.to_datetime(["2018-01-01"]),
            "first_redeem_date": pd.to_datetime([None]),
            "age": [33.0],
            "gender": ["X"],
        }
    )
    valid, bad = CLIENTS_SCHEMA.validate_rows(df)
    assert valid.empty
    assert "gender" in bad["_quarantine_reason"].item()


def test_treatment_flag_must_be_binary() -> None:
    df = pd.DataFrame(
        {"client_id": ["c1", "c2"], "treatment_flg": [1, 2], "target": [0, 1]}
    )
    valid, bad = UPLIFT_TRAIN_SCHEMA.validate_rows(df)
    assert valid["client_id"].tolist() == ["c1"]
    assert "treatment_flg" in bad["_quarantine_reason"].item()


def test_empty_frame_validates_without_error() -> None:
    empty = pd.DataFrame(columns=list(UPLIFT_TRAIN_SCHEMA.column_names))
    valid, bad = UPLIFT_TRAIN_SCHEMA.validate_rows(empty)
    assert valid.empty
    assert bad.empty


def test_cast_to_schema_turns_garbage_into_na() -> None:
    raw = pd.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "treatment_flg": ["1", "0"],
            "target": ["0", "не число"],
        }
    )
    out = cast_to_schema(raw, UPLIFT_TRAIN_SCHEMA)
    # Колонка с нечисловым значением не может быть int8 — остаётся float32 с NaN
    assert out["target"].isna().sum() == 1
    assert out["treatment_flg"].tolist() == [1, 0]


def test_cast_to_schema_parses_datetimes() -> None:
    raw = pd.DataFrame(
        {
            "client_id": ["c1"],
            "first_issue_date": ["2018-03-04"],
            "first_redeem_date": ["не дата"],
            "age": ["42"],
            "gender": ["F"],
        }
    )
    out = cast_to_schema(raw, CLIENTS_SCHEMA)
    assert out["first_issue_date"].dt.year.item() == 2018
    assert pd.isna(out["first_redeem_date"].item())
    assert out["age"].item() == pytest.approx(42.0)


def test_missing_columns_reported() -> None:
    assert PURCHASES_SCHEMA.missing_columns(pd.DataFrame({"client_id": ["c1"]}))
    assert not UPLIFT_TRAIN_SCHEMA.missing_columns(
        pd.DataFrame(columns=list(UPLIFT_TRAIN_SCHEMA.column_names))
    )


def test_schema_summary_covers_all_raw_tables() -> None:
    summary = schema_summary()
    assert set(summary) == {"clients", "purchases", "products", "uplift_train"}
    assert summary["purchases"]["key"] == ["client_id", "transaction_id", "product_id"]


# --------------------------------------------------------------- контракт витрины


def _store(n_covariates: int = 32, n_rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"x{i}": rng.normal(size=n_rows) for i in range(n_covariates)}
    return pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n_rows)],
            "treatment": rng.integers(0, 2, n_rows),
            "target": rng.integers(0, 2, n_rows),
            **data,
        }
    )


def test_valid_store_passes_contract() -> None:
    FeatureStoreContract().assert_valid(_store())


def test_contract_counts_only_covariates() -> None:
    df = _store(n_covariates=30)
    contract = FeatureStoreContract()
    # Служебные колонки не считаются ковариатами (критерий приёмки 3 — про X)
    assert len(contract.covariates(df)) == 30
    assert set(FEATURE_STORE_RESERVED) & set(df.columns) == {"client_id", "treatment", "target"}


def test_contract_rejects_too_few_covariates() -> None:
    violations = FeatureStoreContract().validate(_store(n_covariates=29))
    assert any("число ковариат" in v.rule for v in violations)


def test_contract_rejects_nan_and_inf_in_covariates() -> None:
    df = _store()
    df.loc[0, "x0"] = np.nan
    df.loc[1, "x1"] = np.inf
    rules = {v.rule for v in FeatureStoreContract().validate(df)}
    assert "NaN в X" in rules
    assert "inf в X" in rules


def test_contract_rejects_non_numeric_covariate() -> None:
    df = _store()
    df["x0"] = "строка"
    violations = FeatureStoreContract().validate(df)
    assert any("типы ковариат" in v.rule for v in violations)
    assert any("x0" in v.detail for v in violations)


def test_contract_rejects_duplicate_clients() -> None:
    df = _store()
    df.loc[1, "client_id"] = df.loc[0, "client_id"]
    assert any("уникальность" in v.rule for v in FeatureStoreContract().validate(df))


def test_contract_rejects_non_binary_treatment() -> None:
    df = _store()
    df.loc[0, "treatment"] = 7
    assert any(v.rule == "treatment" for v in FeatureStoreContract().validate(df))


def test_contract_requires_revenue_when_outcome_is_revenue() -> None:
    violations = FeatureStoreContract(outcome="revenue").validate(_store())
    assert any(v.rule == "исход" for v in violations)


def test_assert_valid_message_lists_every_violation() -> None:
    df = _store(n_covariates=5)
    df.loc[0, "x0"] = np.nan
    with pytest.raises(ValueError, match="ETL-6") as exc:
        FeatureStoreContract().assert_valid(df)
    assert "число ковариат" in str(exc.value)
    assert "NaN в X" in str(exc.value)


def test_covariate_columns_excludes_reserved_and_extra() -> None:
    df = _store(n_covariates=3)
    df["split"] = "train"
    df["true_tau"] = 0.1
    assert covariate_columns(df) == ["x0", "x1", "x2"]
    assert covariate_columns(df, extra_exclude=("x0",)) == ["x1", "x2"]
