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
        ("47.1%", 47.1),
        ("$99.99", 99.99),
        ("$15.25", 15.25),
        ("99 cents", 99),
        ("-42", -42.0),
        ("minus forty two", -42.0),
        ("-$123.45", -123.45),
        ("-47.1%", -47.1),
        ("€1,234.56", 1234.56),
        ("1,234.56 dollars", 1234.56),
        ("£0.99", 0.99),
        ("0.99p", 0.99),
        ("twenty-five percent", 25.0),
        ("twenty five percent", 25.0),
        # hard
        ("$15.25, including three $5 bills and one quarter.", 15.25),
        # stress tests - multiple numbers, complex sentences. take the first number
        ("The price is $42.99 but we also have a $29.95 option.", 42.99),
        ("I bought 3 apples for $1.50 each and 2 oranges for $0.75 each.", 3.0),
        ("twenty-three people attended, but only fifteen stayed for dinner.", 23.0),
        ("In 2023, we sold 1,456 units at $19.99 each for a total of $29,084.44.", 2023.0),
        ("The temperature dropped from ninety-five degrees to thirty-two degrees.", 95.0),
        ("Call me at 555-123-4567 or email me about the $250 offer.", 555.0),
        ("Order #12345: 3x widgets at $15.99, 2x gadgets at $8.50.", 12345.0),
        ("We need forty-two volunteers and have a budget of $5,000.", 42.0),
        ("The meeting is at 3:30 PM in room 101 with a $25 entrance fee.", 3.0),
        ("Version 2.1.4 costs $99.99 but version 1.0 is free.", 2.1),
        ("My PIN is 1234 and my balance is $567.89.", 1234.0),
        ("Score: Team A 21, Team B 18. Prize money: $1,000.", 21.0),
        # edge cases with punctuation and formatting
        ("Price: $1,234.56!", 1234.56),
        ("(Cost: $99.99)", 99.99),
        ("Total = $42.50 + tax", 42.50),
        ("Amount: $15.25; discount: 10%", 15.25),
        ("[Price] $123.45 {Final}", 123.45),
        # mixed currency and text numbers
        ("twenty dollars and fifty cents", 20.0),
        ("$5 and thirty-five cents", 5.0),
        ("one hundred fifty pounds sterling", 150.0),
        # decimals in different positions
        ("3.14159 is pi, but 2.71828 is e", 3.14159),
        ("0.5 cups of flour and 1.25 cups of sugar", 0.5),
        ("The rate is 0.075% annually", 0.075),
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


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("minus one hundred", -100.0),
        ("negative one hundred", -100.0),
        ("minus 42", -42.0),
        ("negative 42", -42.0),
        ("minus $123.45", -123.45),
        ("negative 47.1%", -47.1),
        ("minus 1,234.56 dollars", -1234.56),
        ("negative £0.99", -0.99),
        ("minus 0.99p", -0.99),
        ("minus twenty-five percent", -25.0),
        ("negative twenty five percent", -25.0),
    ],
)
def test_fuzzy_cleanup_numeric_negative(pred, expected):
    assert utils.fuzzy_cleanup_numeric(pred) == expected
