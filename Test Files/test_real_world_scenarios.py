"""Real-world scenario tests for StatisticalMoment class."""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statistical_moment import StatisticalMoment, compute_statistical_moments


# Mock SlidingWindow for testing
class MockSlidingWindow:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self._n_columns = data.shape[1]
        self.current_index = 0

    def is_full(self):
        return self.current_index >= self.window_size

    def to_array(self):
        if self.current_index <= self.window_size:
            return self.data[:self.current_index], None
        else:
            return self.data[self.current_index - self.window_size:self.current_index], None

    def add_data(self, new_data):
        if self.current_index < len(self.data):
            self.current_index += 1


def test_sensor_data_analysis():
    """Test with simulated sensor data."""
    print("=" * 60)
    print("TEST 1: SENSOR DATA ANALYSIS")
    print("=" * 60)

    # Simulate accelerometer data (x, y, z coordinates)
    np.random.seed(123)
    time = np.linspace(0, 10, 500)

    # Create synthetic accelerometer data with noise
    x = 2 * np.sin(2 * np.pi * 0.5 * time) + 0.5 * np.random.normal(0, 1, 500)
    y = 1.5 * np.cos(2 * np.pi * 0.3 * time) + 0.3 * np.random.normal(0, 1, 500)
    z = 0.8 * np.sin(2 * np.pi * 0.8 * time) + 0.2 * np.random.normal(0, 1, 500)

    sensor_data = np.column_stack([x, y, z])

    # Create analyzer
    analyzer = StatisticalMoment(
        methods=['mean', 'std_dev', 'skewness', 'kurtosis'],
        output_interpretation=True
    )

    # Create sliding window
    window = MockSlidingWindow(sensor_data, window_size=100)
    window.current_index = 100  # Fill the window

    # Analyze the data
    result = analyzer(window)

    print("Sensor Data Analysis (Accelerometer - x, y, z):")
    print(f"Sample Size: {result['sample_size']}")
    print(f"Features: {result['feature_dimension']}")
    print(f"\nMean: {[f'{x:.3f}' for x in result['mean']]}")
    print(f"Std Dev: {[f'{x:.3f}' for x in result['std_dev']]}")
    print(f"Skewness: {[f'{x:.3f}' for x in result['skewness']]}")
    print(f"Kurtosis: {[f'{x:.3f}' for x in result['kurtosis']]}")

    if 'interpretation' in result:
        print(f"\nInterpretations:")
        for method, interp in result['interpretation'].items():
            print(f"  {method}: {interp}")


def test_ecg_signal_analysis():
    """Test with simulated ECG-like signal data."""
    print("\n" + "=" * 60)
    print("TEST 2: ECG-LIKE SIGNAL ANALYSIS")
    print("=" * 60)

    # Simulate ECG signal with P, QRS, T waves
    t = np.linspace(0, 2, 1000)

    # Synthetic ECG components
    p_wave = 0.1 * np.exp(-((t - 0.2) / 0.02) ** 2)
    qrs_complex = 1.0 * np.exp(-((t - 0.3) / 0.01) ** 2)
    t_wave = 0.3 * np.exp(-((t - 0.5) / 0.05) ** 2)

    ecg_signal = p_wave + qrs_complex + t_wave + 0.05 * np.random.normal(0, 1, 1000)

    # Create analyzer for ECG analysis
    analyzer = StatisticalMoment(
        methods=['mean', 'std_dev', 'skewness'],
        output_interpretation=True
    )

    # Analyze different segments
    segments = [
        ("Baseline", ecg_signal[0:200]),
        ("P-Wave", ecg_signal[180:220]),
        ("QRS-Complex", ecg_signal[280:320]),
        ("T-Wave", ecg_signal[480:520])
    ]

    for segment_name, segment_data in segments:
        # Reshape for single feature
        segment_2d = segment_data.reshape(-1, 1)
        window = MockSlidingWindow(segment_2d, window_size=len(segment_2d))
        window.current_index = len(segment_2d)

        result = compute_statistical_moments(segment_2d, ['mean', 'std_dev', 'skewness'])

        print(f"\n{segment_name} (n={len(segment_data)}):")
        print(f"  Mean: {result['mean'][0]:.4f}")
        print(f"  Std Dev: {result['std_dev'][0]:.4f}")
        print(f"  Skewness: {result['skewness'][0]:.4f}")


def test_movement_patterns():
    """Test with human movement pattern data."""
    print("\n" + "=" * 60)
    print("TEST 3: HUMAN MOVEMENT PATTERNS")
    print("=" * 60)

    # Simulate different movement patterns
    np.random.seed(456)

    # Walking pattern (cyclic)
    time = np.linspace(0, 10, 500)
    walking_x = 0.5 * np.sin(2 * np.pi * 1.0 * time)  # 1 Hz walking
    walking_y = 0.3 * np.cos(2 * np.pi * 1.0 * time)

    # Running pattern (higher frequency)
    running_x = 0.8 * np.sin(2 * np.pi * 2.0 * time)  # 2 Hz running
    running_y = 0.6 * np.cos(2 * np.pi * 2.0 * time)

    # Random movement (exploration)
    random_x = np.cumsum(np.random.normal(0, 0.1, 500))
    random_y = np.cumsum(np.random.normal(0, 0.1, 500))

    movements = [
        ("Walking", np.column_stack([walking_x, walking_y])),
        ("Running", np.column_stack([running_x, running_y])),
        ("Random", np.column_stack([random_x, random_y]))
    ]

    analyzer = StatisticalMoment(
        methods=['mean', 'std_dev', 'skewness', 'kurtosis'],
        output_interpretation=True
    )

    for movement_name, movement_data in movements:
        window = MockSlidingWindow(movement_data, window_size=100)
        window.current_index = 100

        result = analyzer(window)

        print(f"\n{movement_name} Pattern:")
        print(f"  Mean: {[f'{x:.3f}' for x in result['mean']]}")
        print(f"  Std Dev: {[f'{x:.3f}' for x in result['std_dev']]}")
        print(f"  Skewness: {[f'{x:.3f}' for x in result['skewness']]}")
        print(f"  Kurtosis: {[f'{x:.3f}' for x in result['kurtosis']]}")


if __name__ == "__main__":
    test_sensor_data_analysis()
    test_ecg_signal_analysis()
    test_movement_patterns()