"""Budget allocation across groups."""

import pandas as pd
import pytest

from TsT.debiasing.allocation import budgets_from_ids, plan_groups, proportional_budgets, uniform_budgets

VSI_FORMATS = {"mc": 2490, "num": 2640}
VSI_TYPES = {
    "object_size_estimation": 953,
    "object_abs_distance": 834,
    "object_rel_distance": 710,
    "object_counting": 565,
    "obj_appearance_order": 618,
    "object_rel_direction_medium": 378,
    "object_rel_direction_hard": 373,
    "object_rel_direction_easy": 217,
    "room_size_estimation": 288,
    "route_planning": 194,
}


@pytest.mark.parametrize(
    "budget,expected",
    [
        (200, {"mc": 97, "num": 103}),
        (500, {"mc": 243, "num": 257}),
        (1000, {"mc": 485, "num": 515}),
        (2500, {"mc": 1213, "num": 1287}),
    ],
)
def test_per_format_budgets_match_the_paper_runs(budget, expected):
    assert proportional_budgets(VSI_FORMATS, budget) == expected


def test_largest_remainder_sums_to_budget():
    counts = {"a": 1, "b": 1, "c": 1}
    out = proportional_budgets(counts, 2)
    assert sum(out.values()) == 2 and out == {"a": 1, "b": 1, "c": 0}


def test_uniform_54_matches_the_paper_total():
    b = uniform_budgets(VSI_TYPES, 0.54)
    assert sum(b.values()) == 2770 and sum(VSI_TYPES.values()) == 5130


def _df():
    rows = []
    for t, n in {"x": 30, "y": 12, "z": 5}.items():
        rows += [{"question_type": t, "question_format": "mc" if t != "z" else "num"} for _ in range(n)]
    df = pd.DataFrame(rows)
    df["id"] = [f"q{i}" for i in range(len(df))]
    return df


def test_plan_per_type_caps_at_the_group_floor():
    groups = plan_groups(_df(), "per_type", frac=0.5, min_per_group=10)
    assert [(g.name, g.size, g.budget) for g in groups] == [("x", 30, 15), ("y", 12, 2)]  # z: 5 <= floor


def test_plan_per_type_from_id_list():
    df = _df()
    budgets = budgets_from_ids(df, ["q0", "q1", "q2", "q31", "q45"], "id")
    assert budgets == {"x": 3, "y": 1, "z": 1}
    groups = plan_groups(df, "per_type", budgets=budgets, min_per_group=2)
    assert {g.name: g.budget for g in groups} == {"x": 3, "y": 1, "z": 1}
    with pytest.raises(ValueError, match="not in the benchmark"):
        budgets_from_ids(df, ["nope"], "id")


def test_plan_per_format_and_global():
    df = _df()
    groups = plan_groups(df, "per_format", budget=10)
    assert [(g.name, g.size, g.budget) for g in groups] == [("mc", 42, 9), ("num", 5, 1)]
    groups = plan_groups(df, "per_format", budgets={"mc": 3})
    assert [(g.name, g.budget) for g in groups] == [("mc", 3)]
    (g,) = plan_groups(df, "global", budget=7)
    assert (g.name, g.size, g.budget) == ("all", len(df), 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alloc="per_format"),
        dict(alloc="per_format", budget=5, frac=0.1),
        dict(alloc="per_type", budget=5),
        dict(alloc="per_type"),
        dict(alloc="per_type", frac=0.1, budgets={"x": 1}),
        dict(alloc="per_type", budgets={"nope": 1}),
        dict(alloc="per_format", budget=5, budgets={"mc": 1}),
        dict(alloc="per_format", budgets={"mc": 1000}),
        dict(alloc="global", budget=10_000),
        dict(alloc="bogus", budget=1),
    ],
)
def test_plan_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        plan_groups(_df(), **kwargs)
