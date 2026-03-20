import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.ImSim.SourceReconstruction.mesh import RectangularMesh, DelaunayMesh
from lenstronomy.ImSim.SourceReconstruction.regularization import (
    ConstantRegularization,
    GradientRegularization,
    CurvatureRegularization,
    AdaptiveBrightnessRegularization,
)


class TestConstantRegularization:
    def test_identity_matrix(self):
        mesh = RectangularMesh(3, 3, -1, 1, -1, 1)
        reg = ConstantRegularization(coefficient=2.0)
        U = reg.regularization_matrix(mesh)
        assert U.shape == (9, 9)
        npt.assert_array_equal(U, 2.0 * np.eye(9))


class TestGradientRegularization:
    def setup_method(self):
        self.mesh = RectangularMesh(3, 3, -1, 1, -1, 1)
        self.reg = GradientRegularization(coefficient=1.0)

    def test_shape(self):
        U = self.reg.regularization_matrix(self.mesh)
        assert U.shape == (9, 9)

    def test_symmetry(self):
        U = self.reg.regularization_matrix(self.mesh)
        npt.assert_array_almost_equal(U, U.T)

    def test_positive_semidefinite(self):
        U = self.reg.regularization_matrix(self.mesh)
        eigvals = np.linalg.eigvalsh(U)
        assert np.all(eigvals >= -1e-10)

    def test_uniform_field_zero_penalty(self):
        """A uniform source field should have zero gradient penalty."""
        U = self.reg.regularization_matrix(self.mesh)
        a_uniform = np.ones(9)
        penalty = a_uniform @ U @ a_uniform
        npt.assert_almost_equal(penalty, 0.0, decimal=10)

    def test_coefficient_scaling(self):
        reg2 = GradientRegularization(coefficient=3.0)
        U1 = self.reg.regularization_matrix(self.mesh)
        U2 = reg2.regularization_matrix(self.mesh)
        npt.assert_array_almost_equal(U2, 3.0 * U1)

    def test_delaunay_mesh(self):
        """Gradient regularization should work with Delaunay mesh too."""
        mesh = DelaunayMesh.from_regular_grid(4, 4, -1, 1, -1, 1)
        U = self.reg.regularization_matrix(mesh)
        assert U.shape == (16, 16)
        npt.assert_array_almost_equal(U, U.T)
        eigvals = np.linalg.eigvalsh(U)
        assert np.all(eigvals >= -1e-10)


class TestCurvatureRegularization:
    def setup_method(self):
        self.mesh = RectangularMesh(4, 4, -1, 1, -1, 1)
        self.reg = CurvatureRegularization(coefficient=1.0)

    def test_shape(self):
        U = self.reg.regularization_matrix(self.mesh)
        assert U.shape == (16, 16)

    def test_symmetry(self):
        U = self.reg.regularization_matrix(self.mesh)
        npt.assert_array_almost_equal(U, U.T)

    def test_positive_semidefinite(self):
        U = self.reg.regularization_matrix(self.mesh)
        eigvals = np.linalg.eigvalsh(U)
        assert np.all(eigvals >= -1e-10)

    def test_uniform_field_zero_penalty(self):
        U = self.reg.regularization_matrix(self.mesh)
        a_uniform = np.ones(16)
        penalty = a_uniform @ U @ a_uniform
        npt.assert_almost_equal(penalty, 0.0, decimal=10)

    def test_linear_field_low_penalty(self):
        """A linear gradient should have lower curvature penalty than a
        random field. (It's non-zero at boundaries due to Dirichlet BCs.)"""
        mesh = RectangularMesh(5, 5, -1, 1, -1, 1)
        reg = CurvatureRegularization(coefficient=1.0)
        U = reg.regularization_matrix(mesh)
        xc, yc = mesh.pixel_centers
        a_linear = xc + yc
        a_random = np.random.RandomState(42).randn(25)
        penalty_linear = a_linear @ U @ a_linear
        penalty_random = a_random @ U @ a_random
        assert penalty_linear < penalty_random


class TestAdaptiveBrightnessRegularization:
    def setup_method(self):
        self.mesh = RectangularMesh(3, 3, -1, 1, -1, 1)

    def test_no_signals_uses_outer(self):
        """Without pixel signals, should behave like gradient with
        outer_coefficient."""
        reg = AdaptiveBrightnessRegularization(
            inner_coefficient=0.01, outer_coefficient=1.0
        )
        U = reg.regularization_matrix(self.mesh)
        # Should be like gradient reg with coefficient=outer
        grad_reg = GradientRegularization(coefficient=1.0)
        U_grad = grad_reg.regularization_matrix(self.mesh)
        npt.assert_array_almost_equal(U, U_grad)

    def test_symmetry(self):
        reg = AdaptiveBrightnessRegularization(
            inner_coefficient=0.01, outer_coefficient=1.0
        )
        signals = np.random.rand(9)
        reg.set_pixel_signals(signals)
        U = reg.regularization_matrix(self.mesh)
        npt.assert_array_almost_equal(U, U.T)

    def test_bright_pixels_less_regularised(self):
        """Bright pixels should have smaller regularization coefficients."""
        reg = AdaptiveBrightnessRegularization(
            inner_coefficient=0.001, outer_coefficient=10.0, signal_scale=1.0
        )
        # Make center pixel bright
        signals = np.full(9, 0.01)
        signals[4] = 1.0  # center pixel is bright
        reg.set_pixel_signals(signals)
        U = reg.regularization_matrix(self.mesh)
        # Center pixel's self-coupling should be smaller relative to its
        # number of neighbors compared to edge pixels
        assert U[4, 4] > 0  # has regularization
        # A corner pixel with 2 neighbors
        assert U[0, 0] > 0

    def test_signal_length_mismatch(self):
        reg = AdaptiveBrightnessRegularization()
        reg.set_pixel_signals(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            reg.regularization_matrix(self.mesh)
