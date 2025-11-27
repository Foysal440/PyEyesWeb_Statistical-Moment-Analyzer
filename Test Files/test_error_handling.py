"""Error handling and validation tests for StatisticalMoment class."""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statistical_moment import StatisticalMoment, compute_statistical_moments


def test_invalid_inputs():
    """Test error handling with invalid inputs."""
    print("=" * 60)
    print("TEST 1: INVALID INPUT HANDLING")
    print("=" * 60)

    # Test invalid methods
    try:
        analyzer = StatisticalMoment(methods=['invalid_method'])
        print(" Should have raised ValueError for invalid method")
    except ValueError as e:
        print(f" Correctly caught invalid method: {e}")

    # Test empty methods list
    try:
        analyzer = StatisticalMoment(methods=[])
        print(" Should have raised ValueError for empty methods")
    except ValueError as e:
        print(f" Correctly caught empty methods: {e}")

    # Test non-list methods
    try:
        analyzer = StatisticalMoment(methods="mean")
        print(" Should have raised TypeError for non-list methods")
    except TypeError as e:
        print(f" Correctly caught non-list methods: {e}")


def test_insufficient_data():
    """Test behavior with insufficient data."""
    print("\n" + "=" * 60)
    print("TEST 2: INSUFFICIENT DATA")
    print("=" * 60)

    # Single sample (should fail)
    try:
        single_sample = np.array([[1, 2, 3]])
        result = compute_statistical_moments(single_sample, ['mean'])
        print(" Should have failed with single sample")
        print(f"   But got result: {result}")
    except ValueError as e:
        print(f" Correctly caught single sample error: {e}")

    # Empty array
    try:
        empty_data = np.array([]).reshape(0, 2)
        result = compute_statistical_moments(empty_data, ['mean'])
        print(" Should have failed with empty data")
        print(f"   But got result: {result}")
    except ValueError as e:
        print(f" Correctly caught empty data error: {e}")

    # 1D array (should now work with the fix)
    try:
        data_1d = np.array([1, 2, 3, 4, 5])
        result = compute_statistical_moments(data_1d, ['mean'])
        print(f" 1D array handled correctly")
        print(f"   Result: {result}")
    except Exception as e:
        print(f" 1D array failed: {e}")


def test_method_combinations():
    """Test various method combinations."""
    print("\n" + "=" * 60)
    print("TEST 3: METHOD COMBINATIONS")
    print("=" * 60)

    data = np.random.normal(0, 1, (50, 2))

    combinations = [
        ['mean'],
        ['mean', 'std_dev'],
        ['skewness', 'kurtosis'],
        ['mean', 'std_dev', 'skewness', 'kurtosis']
    ]

    for methods in combinations:
        result = compute_statistical_moments(data, methods)
        print(f"Methods {methods}:")
        print(f"  Result keys: {[k for k in result.keys() if k not in ['sample_size', 'feature_dimension']]}")
        print(f"  All requested methods present: {all(method in result for method in methods)}")


def test_numeric_stability():
    """Test with extreme values for numeric stability."""
    print("\n" + "=" * 60)
    print("TEST 4: NUMERIC STABILITY")
    print("=" * 60)

    # Very large values
    large_data = np.array([[1e10, 2e10], [3e10, 4e10], [5e10, 6e10]])
    result_large = compute_statistical_moments(large_data, ['mean', 'std_dev'])
    print("Large values (1e10 scale):")
    print(f"  Mean: {result_large['mean']}")
    print(f"  Std Dev: {result_large['std_dev']}")

    # Very small values
    small_data = np.array([[1e-10, 2e-10], [3e-10, 4e-10], [5e-10, 6e-10]])
    result_small = compute_statistical_moments(small_data, ['mean', 'std_dev'])
    print("\nSmall values (1e-10 scale):")
    print(f"  Mean: {result_small['mean']}")
    print(f"  Std Dev: {result_small['std_dev']}")

    # Mixed scale
    mixed_data = np.array([[1e-5, 1e5], [2e-5, 2e5], [3e-5, 3e5]])
    result_mixed = compute_statistical_moments(mixed_data, ['mean', 'std_dev'])
    print("\nMixed scale (1e-5 and 1e5):")
    print(f"  Mean: {result_mixed['mean']}")
    print(f"  Std Dev: {result_mixed['std_dev']}")


if __name__ == "__main__":
    test_invalid_inputs()
    test_insufficient_data()
    test_method_combinations()
    test_numeric_stability()