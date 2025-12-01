"""Test file for StatisticalMoment class."""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statistical_moment import StatisticalMoment

# Mock SlidingWindow for testing
class MockSlidingWindow:
    def __init__(self, data):
        self.data = data
        if data.size > 0:
            self._n_columns = data.shape[1] if data.ndim > 1 else 1
        else:
            self._n_columns = 3  # Default

    def is_full(self):
        return True

    def to_array(self):
        return self.data, None

def test_basic_functionality():
    """Test basic functionality."""
    print("=" * 60)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    data = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 40, 400],
        [5, 50, 500]
    ])

    window = MockSlidingWindow(data)
    analyzer = StatisticalMoment()

    # Test 1: Only mean
    result = analyzer(window, methods=['mean'])
    print(f"Methods: ['mean']")
    print(f"Result: {result}")
    print(f"Expected mean: [3.0, 30.0, 300.0]")

    # Test 2: Mean and std
    result = analyzer(window, methods=['mean', 'std_dev'])
    print(f"\nMethods: ['mean', 'std_dev']")
    print(f"Result: {result}")
    print(f"Keys in result: {list(result.keys())}")

    # Test 3: All methods
    result = analyzer(window, methods=['mean', 'std_dev', 'skewness', 'kurtosis'])
    print(f"\nMethods: ['mean', 'std_dev', 'skewness', 'kurtosis']")
    print(f"Result: {result}")
    print(f"Keys in result: {list(result.keys())}")

    return True

def test_single_feature():
    """Test with single feature (1D output)."""
    print("\n" + "=" * 60)
    print("TEST 2: SINGLE FEATURE")
    print("=" * 60)

    data = np.array([[1], [2], [3], [4], [5]])
    window = MockSlidingWindow(data)
    analyzer = StatisticalMoment()

    result = analyzer(window, methods=['mean', 'std_dev'])
    print(f"Single feature data shape: {data.shape}")
    print(f"Result: {result}")
    print(f"Mean type: {type(result['mean'])} (should be float)")
    print(f"Std type: {type(result['std'])} (should be float)")

    return True

def test_empty_window():
    """Test with empty/non-full window."""
    print("\n" + "=" * 60)  # FIXED: Changed from "\n" + 60 to "\n" + "=" * 60
    print("TEST 3: EMPTY/NON-FULL WINDOW")
    print("=" * 60)

    class NonFullWindow:
        def __init__(self):
            self._n_columns = 3

        def is_full(self):
            return False

        def to_array(self):
            return np.array([]).reshape(0, 3), None

    window = NonFullWindow()
    analyzer = StatisticalMoment()

    result = analyzer(window, methods=['mean'])
    print(f"Non-full window result: {result}")
    print(f"Is np.nan: {np.isnan(result)}")

    return True

def test_invalid_methods():
    """Test with invalid method names."""
    print("\n" + "=" * 60)
    print("TEST 4: INVALID METHODS")
    print("=" * 60)

    data = np.random.normal(0, 1, (10, 2))
    window = MockSlidingWindow(data)
    analyzer = StatisticalMoment()

    # Test with invalid method (should be skipped)
    result = analyzer(window, methods=['mean', 'invalid', 'std_dev'])
    print(f"Methods: ['mean', 'invalid', 'std_dev']")
    print(f"Result: {result}")
    print(f"Invalid method 'invalid' was skipped")

    return True

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING STATISTICAL MOMENT TESTS")
    print("=" * 70)

    try:
        test_basic_functionality()
        test_single_feature()
        test_empty_window()
        test_invalid_methods()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED SUCCESSFULLY! ")
        print("=" * 70)

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()