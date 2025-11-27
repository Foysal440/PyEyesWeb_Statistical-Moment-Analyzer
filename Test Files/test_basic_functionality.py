"""Basic functionality tests for StatisticalMoment class."""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statistical_moment import StatisticalMoment, compute_statistical_moments


def test_basic_computations():
    """Test basic statistical computations with simple data."""
    print("=" * 60)
    print("TEST 1: BASIC STATISTICAL COMPUTATIONS")
    print("=" * 60)

    # Simple test data - 2D points
    data = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])

    # Test all methods
    analyzer = StatisticalMoment(methods=['mean', 'std_dev', 'skewness', 'kurtosis'])
    result = compute_statistical_moments(data, ['mean', 'std_dev', 'skewness', 'kurtosis'])

    print("Input data:")
    print(data)
    print(f"\nShape: {data.shape}")

    print("\nResults:")
    print(f"Mean: {result['mean']}")
    print(f"Std Dev: {result['std_dev']}")
    print(f"Skewness: {result['skewness']}")
    print(f"Kurtosis: {result['kurtosis']}")
    print(f"Sample Size: {result['sample_size']}")
    print(f"Feature Dimension: {result['feature_dimension']}")

    # Expected results verification
    expected_mean = [3.0, 30.0]
    computed_mean = result['mean']

    print(f"\nVerification:")
    print(f"Expected mean: {expected_mean}")
    print(f"Computed mean: {computed_mean}")
    print(f"Mean calculation correct: {np.allclose(expected_mean, computed_mean)}")


def test_single_method():
    """Test using only one statistical method."""
    print("\n" + "=" * 60)
    print("TEST 2: SINGLE METHOD COMPUTATION")
    print("=" * 60)

    data = np.random.normal(0, 1, (50, 2))

    # Test only mean
    result = compute_statistical_moments(data, ['mean'])
    print("Methods requested: ['mean']")
    print("Available in result:", [key for key in result.keys() if key not in ['sample_size', 'feature_dimension']])
    print(f"Mean: {result['mean']}")

    # Test only standard deviation
    result = compute_statistical_moments(data, ['std_dev'])
    print("\nMethods requested: ['std_dev']")
    print("Available in result:", [key for key in result.keys() if key not in ['sample_size', 'feature_dimension']])
    print(f"Std Dev: {result['std_dev']}")


def test_different_data_shapes():
    """Test with different data dimensions."""
    print("\n" + "=" * 60)
    print("TEST 3: DIFFERENT DATA DIMENSIONS")
    print("=" * 60)

    # 1 feature
    data_1d = np.random.normal(0, 1, (100, 1))
    result_1d = compute_statistical_moments(data_1d, ['mean', 'std_dev'])
    print(f"1 feature - Shape: {data_1d.shape}")
    print(f"Mean: {result_1d['mean']}, Std Dev: {result_1d['std_dev']}")

    # 5 features
    data_5d = np.random.normal(0, 1, (100, 5))
    result_5d = compute_statistical_moments(data_5d, ['mean', 'std_dev'])
    print(f"\n5 features - Shape: {data_5d.shape}")
    print(f"Mean: {result_5d['mean']}")
    print(f"Std Dev: {result_5d['std_dev']}")


if __name__ == "__main__":
    test_basic_computations()
    test_single_method()
    test_different_data_shapes()