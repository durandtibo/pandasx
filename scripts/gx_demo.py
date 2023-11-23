from __future__ import annotations

import logging
from pathlib import Path

import great_expectations as gx
from great_expectations.validator.validator import Validator

from scripts.demo import create_dataframe

logger = logging.getLogger(__name__)


def add_expectations(validator: Validator) -> None:
    validator.expect_table_column_count_to_be_between(min_value=7)
    validator.expect_table_row_count_to_be_between(min_value=10000, max_value=999999999)

    validator.expect_column_values_to_not_be_null("datetime", result_format="SUMMARY")
    validator.expect_column_values_to_match_strftime_format(
        "datetime", strftime_format="%Y-%m-%d %H:%M:%S", result_format="SUMMARY"
    )

    validator.expect_column_values_to_not_be_null("discrete", result_format="SUMMARY")
    validator.expect_column_values_to_be_between(
        "discrete", min_value=0.0, max_value=1000.0, result_format="SUMMARY"
    )
    validator.expect_column_mean_to_be_between(
        "discrete", min_value=495.0, max_value=505.0, result_format="SUMMARY"
    )
    validator.expect_column_median_to_be_between(
        "discrete", min_value=495.0, max_value=505.0, result_format="SUMMARY"
    )

    validator.expect_column_values_to_not_be_null("bool", mostly=0.59, result_format="SUMMARY")
    validator.expect_column_values_to_be_null("bool", mostly=0.39, result_format="SUMMARY")
    validator.expect_column_values_to_be_in_set(
        "bool", value_set={True, False}, result_format="SUMMARY"
    )

    validator.expect_column_values_to_not_be_null("cauchy", mostly=0.79, result_format="SUMMARY")
    validator.expect_column_values_to_be_null("cauchy", mostly=0.19, result_format="SUMMARY")
    validator.expect_column_median_to_be_between(
        "cauchy", min_value=-1.0, max_value=1.0, result_format="SUMMARY"
    )
    validator.expect_column_values_to_be_between("cauchy", min_value=0.0, result_format="SUMMARY")

    validator.expect_column_values_to_not_be_null("float", mostly=0.19, result_format="SUMMARY")
    validator.expect_column_values_to_be_null("float", mostly=0.79, result_format="SUMMARY")
    validator.expect_column_mean_to_be_between(
        "float", min_value=0.9, max_value=1.1, result_format="SUMMARY"
    )
    validator.expect_column_median_to_be_between(
        "float", min_value=0.9, max_value=1.1, result_format="SUMMARY"
    )
    validator.expect_column_stdev_to_be_between("float", min_value=2.9, max_value=3.1)

    validator.expect_column_values_to_not_be_null("int", mostly=0.39, result_format="SUMMARY")
    validator.expect_column_values_to_be_null("int", mostly=0.59, result_format="SUMMARY")
    validator.expect_column_values_to_be_in_set(
        "int", value_set={i for i in range(10)}, result_format="SUMMARY"
    )

    validator.expect_column_values_to_not_be_null("str", mostly=0.79, result_format="SUMMARY")
    validator.expect_column_values_to_be_null("str", mostly=0.18, result_format="SUMMARY")
    validator.expect_column_values_to_be_in_set(
        "str", value_set={"A", "B", "C"}, result_format="SUMMARY"
    )


def save_dataframe() -> Path:
    path = Path.cwd().joinpath("tmp/data.csv")
    logger.info(f"Creating data and saving it in {path}")
    df = create_dataframe(nrows=100000)
    df.to_csv(path)
    return path


def main() -> None:
    logger.info("Create context")
    context = gx.get_context()

    path = save_dataframe()
    validator = context.sources.pandas_default.read_csv(path.as_uri())
    add_expectations(validator)
    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name="my_quickstart_checkpoint",
        validator=validator,
    )
    checkpoint_result = checkpoint.run()
    context.view_validation_result(checkpoint_result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
