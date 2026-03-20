"""Regularization schemes for pixelised source reconstruction.

Provides regularization matrix construction for arbitrary mesh topologies,
generalising the existing rectangular-only implementation in
``regularization_matrix_pixel.py``.

Supported schemes:
- Constant gradient: penalises differences between neighbouring pixels
  with a uniform coefficient
- Constant curvature: penalises second-order differences
- Adaptive brightness: varies the regularization coefficient per pixel
  based on an estimate of the source brightness
"""

import numpy as np

__all__ = [
    "RegularizationBase",
    "ConstantRegularization",
    "GradientRegularization",
    "CurvatureRegularization",
    "AdaptiveBrightnessRegularization",
]


class RegularizationBase:
    """Abstract base class for regularization schemes."""

    def regularization_matrix(self, mesh):
        """Construct the regularization matrix U for a given mesh.

        The regularization term in the linear system is lambda * a^T U a,
        where a is the vector of source pixel amplitudes.

        :param mesh: a MeshBase instance providing num_pixels and neighbor_list()
        :return: (num_pixels, num_pixels) numpy array
        """
        raise NotImplementedError


class ConstantRegularization(RegularizationBase):
    """Zeroth-order (identity) regularization.

    Penalises the amplitude of each pixel uniformly:
    R = coefficient * I

    :param coefficient: regularization strength (lambda)
    """

    def __init__(self, coefficient=1.0):
        self.coefficient = float(coefficient)

    def regularization_matrix(self, mesh):
        n = mesh.num_pixels
        return self.coefficient * np.eye(n)


class GradientRegularization(RegularizationBase):
    """First-order (gradient) regularization based on mesh adjacency.

    Penalises differences between neighbouring pixel amplitudes:
    sum_{i~j} (a_i - a_j)^2

    This works for any mesh topology via the neighbour list.

    :param coefficient: regularization strength (lambda)
    """

    def __init__(self, coefficient=1.0):
        self.coefficient = float(coefficient)

    def regularization_matrix(self, mesh):
        n = mesh.num_pixels
        neighbors = mesh.neighbor_list()
        U = np.zeros((n, n))
        for i in range(n):
            nbrs = neighbors[i]
            n_nbrs = len(nbrs)
            U[i, i] += n_nbrs
            for j in nbrs:
                U[i, j] -= 1.0
        return self.coefficient * U


class CurvatureRegularization(RegularizationBase):
    """Second-order (curvature) regularization based on mesh adjacency.

    For each pixel i with neighbours {j}, penalises the discrete Laplacian:
    sum_i (n_i * a_i - sum_{j~i} a_j)^2

    where n_i is the number of neighbours of pixel i.

    :param coefficient: regularization strength (lambda)
    """

    def __init__(self, coefficient=1.0):
        self.coefficient = float(coefficient)

    def regularization_matrix(self, mesh):
        n = mesh.num_pixels
        neighbors = mesh.neighbor_list()

        # Build the Laplacian operator L, then U = L^T L
        L = np.zeros((n, n))
        for i in range(n):
            nbrs = neighbors[i]
            n_nbrs = len(nbrs)
            L[i, i] = n_nbrs
            for j in nbrs:
                L[i, j] = -1.0
        U = L.T @ L
        return self.coefficient * U


class AdaptiveBrightnessRegularization(RegularizationBase):
    """Adaptive regularization that varies smoothing strength by pixel brightness.

    Bright pixels are regularised less (allowing fine detail), while faint
    pixels are regularised more (suppressing noise). The per-pixel
    coefficient is:

        lambda_i = outer_coefficient * (signal_i / signal_max)^(-signal_scale)

    clamped to [inner_coefficient, outer_coefficient].

    This follows the approach of Nightingale & Dye (2015) as used in
    PyAutoLens.

    :param inner_coefficient: minimum regularization coefficient (for brightest pixels)
    :param outer_coefficient: maximum regularization coefficient (for faintest pixels)
    :param signal_scale: exponent controlling how steeply the coefficient
        varies with brightness (higher = more contrast)
    :param pixel_signals: 1d array of per-pixel brightness estimates.
        If None, must be set via set_pixel_signals() before building the matrix.
    """

    def __init__(
        self,
        inner_coefficient=0.01,
        outer_coefficient=1.0,
        signal_scale=1.0,
        pixel_signals=None,
    ):
        self.inner_coefficient = float(inner_coefficient)
        self.outer_coefficient = float(outer_coefficient)
        self.signal_scale = float(signal_scale)
        self._pixel_signals = pixel_signals

    def set_pixel_signals(self, pixel_signals):
        """Set or update the per-pixel brightness estimates.

        :param pixel_signals: 1d array of brightness values, one per source pixel
        """
        self._pixel_signals = np.asarray(pixel_signals, dtype=float)

    def _pixel_coefficients(self, n_pixels):
        """Compute per-pixel regularization coefficients.

        :param n_pixels: number of source pixels
        :return: 1d array of coefficients
        """
        if self._pixel_signals is None:
            return np.full(n_pixels, self.outer_coefficient)

        signals = self._pixel_signals
        if len(signals) != n_pixels:
            raise ValueError(
                f"pixel_signals length ({len(signals)}) does not match "
                f"number of mesh pixels ({n_pixels})."
            )

        signal_max = np.max(np.abs(signals))
        if signal_max == 0:
            return np.full(n_pixels, self.outer_coefficient)

        # Normalise to [0, 1]
        normed = np.abs(signals) / signal_max
        # Avoid division by zero for zero-signal pixels
        normed = np.clip(normed, 1e-10, 1.0)

        coefficients = self.outer_coefficient * normed ** (-self.signal_scale)
        coefficients = np.clip(
            coefficients, self.inner_coefficient, self.outer_coefficient
        )
        return coefficients

    def regularization_matrix(self, mesh):
        n = mesh.num_pixels
        neighbors = mesh.neighbor_list()
        coeffs = self._pixel_coefficients(n)

        U = np.zeros((n, n))
        for i in range(n):
            nbrs = neighbors[i]
            for j in nbrs:
                # Use the average coefficient of the two pixels
                avg_coeff = 0.5 * (coeffs[i] + coeffs[j])
                U[i, i] += avg_coeff
                U[i, j] -= avg_coeff
        return U
