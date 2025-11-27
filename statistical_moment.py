"""Statistical moments analysis module for signal processing.

This module provides tools for computing various statistical moments from
multivariate signal data. Statistical moments characterize the shape and
properties of probability distributions and are fundamental in signal analysis.

The available statistical moments include:
1. Mean - Central tendency of the data
2. Standard Deviation - Dispersion around the mean
3. Skewness - Asymmetry of the distribution
4. Kurtosis - Tailedness of the distribution

Typical use cases include:
1. Signal characterization and feature extraction
2. Quality assessment of sensor data
3. Motion pattern analysis in movement data
4. Anomaly detection in time series data
5. Distribution analysis in multivariate signals

References
----------
1. Pearson, K. (1895). Contributions to the Mathematical Theory of Evolution.
2. Fisher, R. A. (1925). Statistical Methods for Research Workers.
"""

import numpy as np
from scipy import stats
from pyeyesweb.data_models.sliding_window import SlidingWindow
from pyeyesweb.data_models.thread_safe_buffer import ThreadSafeHistoryBuffer
from pyeyesweb.utils.validators import validate_boolean, validate_list, validate_integer


class StatisticalMoment:
    """Real time statistical moments analyzer for signal data.

    This class computes various statistical moments (mean, standard deviation,
    skewness, kurtosis) from sliding window data to characterize signal
    distributions and properties.

    Parameters
    ----------
    methods : list of str
        List of statistical methods to compute. Available options:
        'mean', 'std_dev', 'skewness', 'kurtosis'
    sensitivity : int, optional
        Size of the statistical history buffer. Larger values provide
        more temporal context but increase memory usage. Must be positive
        integer between 1 and 10,000 (default: 100).
    output_interpretation : bool, optional
        If True, outputs statistical interpretation as categorical labels.
        Must be boolean (default: True).
    random_state : int or None, optional
        Random seed for reproducible computations. If None, uses default
        sampling (default: None).
    """

    def __init__(self, methods, sensitivity=100, output_interpretation=True,
                 random_state=None):
        self.methods = validate_list(methods, 'methods',
                                     valid_options=['mean', 'std_dev', 'skewness', 'kurtosis'],
                                     min_length=1)  # Require at least 1 method
        sensitivity = validate_integer(sensitivity, 'sensitivity', min_val=1, max_val=10000)
        self.output_interpretation = validate_boolean(output_interpretation, 'output_interpretation')

        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError("random_state must be integer or None")
            if random_state < 0:
                raise ValueError("random_state must be non-negative")
        self.random_state = random_state

        # Initialize history buffers for each method
        self.history_buffers = {
            'mean': ThreadSafeHistoryBuffer(maxlen=sensitivity),
            'std_dev': ThreadSafeHistoryBuffer(maxlen=sensitivity),
            'skewness': ThreadSafeHistoryBuffer(maxlen=sensitivity),
            'kurtosis': ThreadSafeHistoryBuffer(maxlen=sensitivity)
        }

    def compute_statistical_moments(self, data: np.ndarray) -> dict:
        """Compute statistical moments for multivariate data.

        Parameters
        ----------
        data : np.ndarray
            Multivariate data array of shape (n_samples, n_features).
            Must contain at least 2 samples.

        Returns
        -------
        dict
            Dictionary containing computed statistical moments.
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got {data.ndim}D")

        n_samples, n_features = data.shape

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")

        if n_features < 1:
            raise ValueError("Data must have at least 1 feature")

        result = {}

        # Compute requested statistical moments
        for method in self.methods:
            if method == 'mean':
                values = np.mean(data, axis=0).tolist()
                result['mean'] = values
                self.history_buffers['mean'].append(values)

            elif method == 'std_dev':
                values = np.std(data, axis=0, ddof=1).tolist()
                result['std_dev'] = values
                self.history_buffers['std_dev'].append(values)

            elif method == 'skewness':
                values = stats.skew(data, axis=0).tolist()
                result['skewness'] = values
                self.history_buffers['skewness'].append(values)

            elif method == 'kurtosis':
                values = stats.kurtosis(data, axis=0).tolist()
                result['kurtosis'] = values
                self.history_buffers['kurtosis'].append(values)

        return result

    def interpret_statistical_moments(self, moments: dict) -> dict:
        """Interpret statistical moments with categorical labels.

        Parameters
        ----------
        moments : dict
            Dictionary containing statistical moments.

        Returns
        -------
        dict
            Dictionary containing interpretations.
        """
        interpretation = {}

        if 'skewness' in moments:
            skew_interpretations = []
            for skew_val in moments['skewness']:
                if skew_val > 1:
                    skew_interpretations.append("HIGHLY POSITIVE SKEW")
                elif skew_val > 0.5:
                    skew_interpretations.append("MODERATE POSITIVE SKEW")
                elif skew_val > -0.5:
                    skew_interpretations.append("APPROXIMATELY SYMMETRIC")
                elif skew_val > -1:
                    skew_interpretations.append("MODERATE NEGATIVE SKEW")
                else:
                    skew_interpretations.append("HIGHLY NEGATIVE SKEW")
            interpretation['skewness'] = skew_interpretations

        if 'kurtosis' in moments:
            kurt_interpretations = []
            for kurt_val in moments['kurtosis']:
                if kurt_val > 3:
                    kurt_interpretations.append("LEPTOKURTIC (HEAVY-TAILED)")
                elif kurt_val > 1:
                    kurt_interpretations.append("MESOKURTIC (NORMAL-LIKE)")
                else:
                    kurt_interpretations.append("PLATYKURTIC (LIGHT-TAILED)")
            interpretation['kurtosis'] = kurt_interpretations

        return interpretation

    def compute_statistics(self, signals: SlidingWindow) -> dict:
        """Compute statistical analysis for multivariate signals.

        Parameters
        ----------
        signals : SlidingWindow
            Sliding window buffer containing multivariate signal data.

        Returns
        -------
        dict
            Dictionary containing statistical metrics.
        """
        if not signals.is_full():
            return self._create_empty_result(signals._n_columns)

        data, _ = signals.to_array()
        n_samples, n_features = data.shape

        if n_samples < 2:
            return self._create_empty_result(n_features)

        try:
            moments = self.compute_statistical_moments(data)

            interpretation = {}
            if self.output_interpretation:
                interpretation = self.interpret_statistical_moments(moments)

            result = {
                "sample_size": n_samples,
                "feature_dimension": n_features
            }
            result.update(moments)
            if interpretation:
                result["interpretation"] = interpretation

            return result

        except Exception as e:
            return self._create_error_result(n_features)

    def _create_empty_result(self, n_features: int) -> dict:
        """Create empty result dictionary with NaN values."""
        result = {
            "sample_size": 0,
            "feature_dimension": n_features
        }

        for method in self.methods:
            result[method] = [np.nan] * n_features

        if self.output_interpretation:
            result["interpretation"] = {}
            for method in self.methods:
                if method in ['skewness', 'kurtosis']:
                    result["interpretation"][method] = ["NO DATA"] * n_features

        return result

    def _create_error_result(self, n_features: int) -> dict:
        """Create error result dictionary."""
        result = {
            "sample_size": n_features,  # Keep original dimension for context
            "feature_dimension": n_features
        }

        for method in self.methods:
            result[method] = [np.nan] * n_features

        if self.output_interpretation:
            result["interpretation"] = {"error": "COMPUTATION ERROR"}

        return result

    def get_temporal_statistics(self) -> dict:
        """Get temporal statistics from history buffers.

        Returns
        -------
        dict
            Dictionary containing temporal analysis.
        """
        temporal_stats = {}

        for method in self.methods:
            history = self.history_buffers[method].get_all()
            if len(history) < 2:
                temporal_stats[method] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "trend": np.nan,
                    "stability": np.nan,
                    "history_length": len(history)
                }
                continue

            # Convert history to numpy array for analysis
            history_array = np.array(history)

            # Compute statistics across time for each feature
            mean_vals = np.mean(history_array, axis=0)
            std_vals = np.std(history_array, axis=0)

            # Compute trend for each feature
            x = np.arange(len(history))
            trends = []
            for feature_idx in range(history_array.shape[1]):
                if len(history) > 1:
                    trend = np.polyfit(x, history_array[:, feature_idx], 1)[0]
                    trends.append(trend)
                else:
                    trends.append(np.nan)

            temporal_stats[method] = {
                "mean": mean_vals.tolist(),
                "std": std_vals.tolist(),
                "trend": trends,
                "stability": (std_vals / np.abs(mean_vals)).tolist() if np.any(mean_vals != 0) else [np.inf] * len(mean_vals),
                "history_length": len(history)
            }

        return temporal_stats

    def __call__(self, sliding_window: SlidingWindow) -> dict:
        """Compute and optionally display statistical metrics.

        Parameters
        ----------
        sliding_window : SlidingWindow
            Buffer containing multivariate data to analyze.

        Returns
        -------
        dict
            Dictionary containing statistical metrics.
        """
        result = self.compute_statistics(sliding_window)

        if not np.isnan(result["sample_size"]) and result["sample_size"] > 0:
            print(f"Statistical Moments (n={result['sample_size']}):")
            for method in self.methods:
                if method in result and not np.isnan(result[method][0]):
                    values = result[method]
                    if len(values) == 1:
                        print(f"  {method}: {values[0]:.4f}")
                    else:
                        print(f"  {method}: {[f'{v:.4f}' for v in values]}")

                    if self.output_interpretation and 'interpretation' in result and method in result['interpretation']:
                        interpretations = result['interpretation'][method]
                        if len(interpretations) == 1:
                            print(f"    Interpretation: {interpretations[0]}")
                        else:
                            print(f"    Interpretation: {interpretations}")

        return result

    def reset_history(self):
        """Clear the statistical history buffers."""
        for buffer in self.history_buffers.values():
            buffer.clear()

    def get_history(self, method: str) -> np.ndarray:
        """Get the complete history of statistical values for a specific method.

        Parameters
        ----------
        method : str
            Statistical method to get history for.

        Returns
        -------
        np.ndarray
            Array of all stored statistical values.
        """
        if method not in self.history_buffers:
            raise ValueError(f"Invalid method '{method}'. Available methods: {list(self.history_buffers.keys())}")

        return np.array(self.history_buffers[method].get_all())


def compute_statistical_moments(data: np.ndarray, methods: list) -> dict:
    """One-time statistical moments computation for static datasets.

    Parameters
    ----------
    data : np.ndarray
        Multivariate data array of shape (n_samples, n_features).
    methods : list of str
        List of statistical methods to compute.

    Returns
    -------
    dict
        Statistical moments computation results.
    """
    analyzer = StatisticalMoment(
        methods=methods,
        sensitivity=1,
        output_interpretation=True
    )

    class MockSlidingWindow:
        def __init__(self, data):
            self.data = data
            # Handle 1D arrays by reshaping to 2D
            if data.ndim == 1:
                self.data = data.reshape(-1, 1)
            self._n_columns = self.data.shape[1]

        def is_full(self):
            return True

        def to_array(self):
            return self.data, None

    window = MockSlidingWindow(data)
    return analyzer.compute_statistics(window)