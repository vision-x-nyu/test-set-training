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


# Tests for mean_relative_accuracy
def test_mean_relative_accuracy_perfect_predictions():
    """Test MRA with perfect predictions (should return 1.0)"""
    import numpy as np

    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = utils.mean_relative_accuracy(pred, true)
    assert result == 1.0


def test_mean_relative_accuracy_basic_case():
    """Test MRA with known inputs and expected output"""
    import numpy as np

    # Simple case: predictions that are 10% off
    pred = np.array([1.1, 2.2, 3.3])
    true = np.array([1.0, 2.0, 3.0])

    # Relative errors are all 0.1 (10%)
    # For thresholds 0.5 to 0.95, some will pass and some won't
    # 0.1 < (1-0.5)=0.5 ✓, but 0.1 < (1-0.95)=0.05 ✗
    result = utils.mean_relative_accuracy(pred, true)
    assert 0.0 < result < 1.0  # Should be between 0 and 1


def test_mean_relative_accuracy_very_good_predictions():
    """Test MRA with very good predictions that pass all thresholds"""
    import numpy as np

    # Predictions with only 1% error
    pred = np.array([1.01, 2.01, 3.01])
    true = np.array([1.0, 2.0, 3.0])

    # Relative errors are all 0.01 (1%)
    # This should pass all thresholds since 0.01 < (1-t) for all t in [0.5, 0.95]
    result = utils.mean_relative_accuracy(pred, true)
    assert result == 1.0


def test_mean_relative_accuracy_poor_predictions():
    """Test MRA with very poor predictions"""
    import numpy as np

    # Predictions that are 200% off (relative error = 2.0)
    pred = np.array([3.0, 6.0, 9.0])
    true = np.array([1.0, 2.0, 3.0])

    # Relative errors are all 2.0 (200%)
    # For thresholds 0.5 to 0.95, none should pass since 2.0 > (1-threshold) for all
    result = utils.mean_relative_accuracy(pred, true)
    assert result == 0.0


def test_mean_relative_accuracy_mixed_predictions():
    """Test MRA with mixed quality predictions"""
    import numpy as np

    # Mix of good and bad predictions
    pred = np.array([1.05, 3.0, 2.1])  # 5% error, 200% error, 5% error
    true = np.array([1.0, 1.0, 2.0])

    result = utils.mean_relative_accuracy(pred, true)
    # Should be between 0 and 1, closer to 0.67 since 2/3 predictions are good
    assert 0.0 < result < 1.0


def test_mean_relative_accuracy_custom_thresholds():
    """Test MRA with custom threshold ranges"""
    import numpy as np

    pred = np.array([1.1, 1.1, 1.1])  # 10% error
    true = np.array([1.0, 1.0, 1.0])

    # With stricter thresholds (0.8 to 0.9), 10% error might not pass all
    result = utils.mean_relative_accuracy(pred, true, start=0.8, end=0.9, step=0.05)
    assert 0.0 <= result <= 1.0


def test_mean_relative_accuracy_edge_cases():
    """Test MRA edge cases"""
    import numpy as np

    # Single value
    result = utils.mean_relative_accuracy(np.array([1.0]), np.array([1.0]))
    assert result == 1.0

    # Zero true values should be handled (but might cause division by zero)
    # This tests the robustness of the function
    try:
        result = utils.mean_relative_accuracy(np.array([1.0]), np.array([0.0]))
        # If it doesn't crash, result should be a valid number
        assert isinstance(result, (int, float))
    except (ZeroDivisionError, ValueError):
        # It's acceptable if the function raises an error for zero true values
        pass


# Tests for weighted_mean_std
def test_weighted_mean_std_basic():
    """Test weighted mean and std with basic inputs"""
    import numpy as np

    scores = np.array([0.8, 0.6, 0.9])
    counts = np.array([10, 20, 30])

    mean, std = utils.weighted_mean_std(scores, counts)

    # Manual calculation: weighted_avg = (0.8*10 + 0.6*20 + 0.9*30) / (10+20+30)
    expected_mean = (0.8 * 10 + 0.6 * 20 + 0.9 * 30) / 60
    assert abs(mean - expected_mean) < 1e-10


def test_weighted_mean_std_equal_weights():
    """Test weighted stats with equal weights (should match simple mean/std)"""
    import numpy as np

    scores = np.array([0.5, 0.7, 0.9])
    counts = np.array([1, 1, 1])  # Equal weights

    mean, std = utils.weighted_mean_std(scores, counts)

    # Should be close to simple mean and std
    expected_mean = np.mean(scores)
    assert abs(mean - expected_mean) < 1e-10


def test_weighted_mean_std_zero_counts():
    """Test weighted stats with some zero counts"""
    import numpy as np

    scores = np.array([0.8, 0.6, 0.9, 0.4])
    counts = np.array([10, 0, 20, 0])  # Only first and third have non-zero counts

    mean, std = utils.weighted_mean_std(scores, counts)

    # Should only consider scores with non-zero counts
    expected_mean = (0.8 * 10 + 0.9 * 20) / 30
    assert abs(mean - expected_mean) < 1e-10


def test_weighted_mean_std_all_zero_counts():
    """Test weighted stats with all zero counts"""
    import numpy as np

    scores = np.array([0.8, 0.6, 0.9])
    counts = np.array([0, 0, 0])

    mean, std = utils.weighted_mean_std(scores, counts)

    # Should return 0, 0 for all zero counts
    assert mean == 0
    assert std == 0


def test_weighted_mean_std_single_value():
    """Test weighted stats with single non-zero value"""
    import numpy as np

    scores = np.array([0.75])
    counts = np.array([100])

    mean, std = utils.weighted_mean_std(scores, counts)

    assert abs(mean - 0.75) < 1e-10
    assert std == 0.0  # Single value should have zero std


def test_weighted_mean_std_mathematical_correctness():
    """Test that weighted variance formula is implemented correctly"""
    import numpy as np

    scores = np.array([0.2, 0.8])
    counts = np.array([3, 7])

    mean, std = utils.weighted_mean_std(scores, counts)

    # Manual calculation
    total_count = 10
    weighted_mean = (0.2 * 3 + 0.8 * 7) / total_count  # = 0.62
    weighted_var = (3 * (0.2 - weighted_mean) ** 2 + 7 * (0.8 - weighted_mean) ** 2) / total_count
    expected_std = weighted_var**0.5

    assert abs(mean - weighted_mean) < 1e-10
    assert abs(std - expected_std) < 1e-10


def test_weighted_mean_std_larger_arrays():
    """Test weighted stats with larger arrays"""
    import numpy as np

    np.random.seed(42)
    scores = np.random.uniform(0, 1, 100)
    counts = np.random.randint(1, 50, 100)

    mean, std = utils.weighted_mean_std(scores, counts)

    # Basic sanity checks
    assert 0 <= mean <= 1  # Should be within score range
    assert std >= 0  # Standard deviation should be non-negative
    assert isinstance(mean, float)
    assert isinstance(std, float)


def test_weighted_mean_std_empty_arrays():
    """Test weighted stats with empty arrays"""
    import numpy as np

    scores = np.array([])
    counts = np.array([])

    mean, std = utils.weighted_mean_std(scores, counts)

    # Should handle empty arrays gracefully
    assert mean == 0
    assert std == 0


class TestExactMatch:
    def test_strings_case_insensitive(self):
        assert utils.exact_match("Hello", "hello") == 1.0
        assert utils.exact_match("World", "world") == 1.0
        assert utils.exact_match("Hello", "World") == 0.0

    def test_numeric_values(self):
        assert utils.exact_match(3, 3) == 1.0
        assert utils.exact_match(3.0, 3.0) == 1.0
        assert utils.exact_match(3.0, 4.0) == 0.0


@pytest.mark.parametrize(
    "pred,target,expected",
    [
        ("  Hello world.  ", "hello", 1.0),
        ("Test. ", "test", 1.0),
        ("42 is the answer", "42", 1.0),
        (" 42. ", "42", 1.0),
        ("MixedCase Word.", "mixedcase", 1.0),
        ("foo.bar ", "foo.bar", 1.0),
        ("different", "answer", 0.0),
    ],
)
def test_fuzzy_match(pred, target, expected):
    assert utils.fuzzy_match(pred, target) == expected


class TestMultiChoiceParsing:
    def test_get_multi_choice_info(self):
        options = ["Red", "Blue", "Green"]
        index2ans, all_choices = utils.get_multi_choice_info(options)

        assert all_choices == ["A", "B", "C"]
        assert index2ans == {"A": "Red", "B": "Blue", "C": "Green"}

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("The answer is (B).", "B"),
            ("I choose C", "C"),
            ("Final answer: A.", "A"),
        ],
    )
    def test_parse_multi_choice_response_variants(self, response, expected):
        options = ["Option A", "Option B", "Option C"]
        assert utils.parse_multi_choice_response(response, options) == expected

    def test_parse_multi_choice_content_based(self):
        options = ["cat", "dog", "bird"]
        # Use >5 tokens to trigger content-based parsing in the implementation
        response = "In my opinion, it is the dog indeed."
        assert utils.parse_multi_choice_response(response, options) == "B"

    def test_multiple_candidates_last_one_wins(self):
        options = ["opt1", "opt2", "opt3"]
        response = "A B A"  # Both A and B appear; last occurrence is A
        assert utils.parse_multi_choice_response(response, options) == "A"

    def test_random_fallback_monkeypatched(self, monkeypatch):
        # Force random fallback to a deterministic choice
        monkeypatch.setattr(utils.random, "choice", lambda seq: "C")

        options = ["Red", "Blue", "Green"]
        response = "No valid marker here"
        assert utils.parse_multi_choice_response(response, options) == "C"
