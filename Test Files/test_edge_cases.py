"""Edge case tests for StatisticalMoment class."""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statistical_moment import StatisticalMoment, compute_statistical_moments


def test_small_samples():
    """Test with very small sample sizes."""
    print("=" * 60)
    print("TEST 1: SMALL SAMPLE SIZES")
    print("=" * 60)

    # 2 samples (minimum required)
    data_2 = np.array([[1, 10], [2, 20]])
    result_2 = compute_statistical_moments(data_2, ['mean', 'std_dev'])
    print(f"2 samples - Shape: {data_2.shape}")
    print(f"Mean: {result_2['mean']}")
    print(f"Std Dev: {result_2['std_dev']}")
    print(f"Sample Size: {result_2['sample_size']}")

    # 3 samples
    data_3 = np.array([[1, 10], [2, 20], [3, 30]])
    result_3 = compute_statistical_moments(data_3, ['mean', 'std_dev', 'skewness'])
    print(f"\n3 samples - Shape: {data_3.shape}")
    print(f"Mean: {result_3['mean']}")
    print(f"Skewness: {result_3['skewness']}")


def test_constant_data():
    """Test with constant (zero variance) data."""
    print("\n" + "=" * 60)
    print("TEST 2: CONSTANT DATA")
    print("=" * 60)

    # All values same
    constant_data = np.ones((10, 3)) * 5.0
    result = compute_statistical_moments(constant_data, ['mean', 'std_dev', 'skewness', 'kurtosis'])

    print("Constant data (all values = 5.0):")
    print(f"Mean: {result['mean']}")
    print(f"Std Dev: {result['std_dev']}")  # Should be [0, 0, 0]
    print(f"Skewness: {result['skewness']}")  # May be NaN or specific value
    print(f"Kurtosis: {result['kurtosis']}")  # May be NaN or specific value


def test_normal_distribution():
    """Test with normally distributed data."""
    print("\n" + "=" * 60)
    print("TEST 3: NORMAL DISTRIBUTION PROPERTIES")
    print("=" * 60)

    np.random.seed(42)  # For reproducible results
    normal_data = np.random.normal(0, 1, (1000, 2))
    result = compute_statistical_moments(normal_data, ['mean', 'std_dev', 'skewness', 'kurtosis'])

    print("Normal distribution (μ=0, σ=1) with 1000 samples:")
    print(f"Mean (should be ~0): {[f'{x:.3f}' for x in result['mean']]}")
    print(f"Std Dev (should be ~1): {[f'{x:.3f}' for x in result['std_dev']]}")
    print(f"Skewness (should be ~0): {[f'{x:.3f}' for x in result['skewness']]}")
    print(f"Kurtosis (should be ~0): {[f'{x:.3f}' for x in result['kurtosis']]}")


def test_skewed_data():
    """Test with intentionally skewed data."""
    print("\n" + "=" * 60)
    print("TEST 4: SKEWED DISTRIBUTION")
    print("=" * 60)

    # Create positively skewed data (exponential distribution)
    skewed_data = np.random.exponential(2, (500, 1))
    result = compute_statistical_moments(skewed_data, ['mean', 'std_dev', 'skewness', 'kurtosis'])

    print("Exponential distribution (positively skewed):")
    print(f"Mean: {result['mean'][0]:.3f}")
    print(f"Std Dev: {result['std_dev'][0]:.3f}")
    print(f"Skewness: {result['skewness'][0]:.3f} (should be positive)")
    print(f"Kurtosis: {result['kurtosis'][0]:.3f}")


if __name__ == "__main__":
    test_small_samples()
    test_constant_data()
    test_normal_distribution()
    test_skewed_data()