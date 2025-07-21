import pytest
from TsT import utils


# Test get_t2d returns a Text2Digits instance and is cached
def test_get_t2d_singleton():
    t2d1 = utils.get_t2d()
    t2d2 = utils.get_t2d()
    assert t2d1 is t2d2
    from text2digits import text2digits

    assert isinstance(t2d1, text2digits.Text2Digits)


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("  Hello world.  ", "hello"),
        ("Test. ", "test"),
        ("42 is the answer", "42"),
        (" 42. ", "42"),
        ("MixedCase Word.", "mixedcase"),
        (" 123.45 ", "123.45"),
        ("foo.bar ", "foo.bar"),
    ],
)
def test_fuzzy_cleanup(pred, expected):
    assert utils.fuzzy_cleanup(pred) == expected


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("forty two", 42.0),
        ("  123.45  ", 123.45),
        ("one hundred", 100.0),
        ("2,000", 2000.0),
        ("  3.14159  ", 3.14159),
        ("zero", 0.0),
        ("twenty one.", 21.0),
    ],
)
def test_fuzzy_cleanup_numeric(pred, expected):
    assert utils.fuzzy_cleanup_numeric(pred) == expected


# Edge case: input that cannot be converted should raise ValueError
@pytest.mark.parametrize(
    "pred",
    [
        "not a number",
        "",
        "foo bar",
    ],
)
def test_fuzzy_cleanup_numeric_invalid(pred):
    with pytest.raises(ValueError):
        utils.fuzzy_cleanup_numeric(pred)
