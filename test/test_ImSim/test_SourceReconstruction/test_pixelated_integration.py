"""Integration tests for the full pixelated source reconstruction pipeline.

Tests the end-to-end flow: mesh + regularization + mapper + light profile
integration with ImageLinearFit.
"""

import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.ImSim.SourceReconstruction.mesh import RectangularMesh, DelaunayMesh
from lenstronomy.ImSim.SourceReconstruction.regularization import (
    GradientRegularization,
    CurvatureRegularization,
    AdaptiveBrightnessRegularization,
)
from lenstronomy.ImSim.SourceReconstruction.mapper import Mapper
from lenstronomy.ImSim.SourceReconstruction.pixelated_operator import PixelatedOperator
from lenstronomy.LightModel.Profiles.pixelated import Pixelated
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestMapper:
    def setup_method(self):
        self.lens_model = LensModel(lens_model_list=["SIS"])
        self.kwargs_lens = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
        self.mesh = RectangularMesh(
            nx=5, ny=5, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5
        )
        x_img = np.linspace(-2, 2, 20)
        y_img = np.linspace(-2, 2, 20)
        xx, yy = np.meshgrid(x_img, y_img)
        self.x_img = xx.ravel()
        self.y_img = yy.ravel()

    def test_mapping_matrix_shape(self):
        mapper = Mapper(self.mesh, self.lens_model, self.x_img, self.y_img)
        F = mapper.mapping_matrix(self.kwargs_lens)
        assert F.shape == (400, 25)

    def test_mapping_matrix_non_negative(self):
        mapper = Mapper(self.mesh, self.lens_model, self.x_img, self.y_img)
        F = mapper.mapping_matrix(self.kwargs_lens)
        assert np.all(F >= 0)

    def test_mapping_matrix_row_sum(self):
        """Each image pixel maps to at most one source pixel for rectangular."""
        mapper = Mapper(self.mesh, self.lens_model, self.x_img, self.y_img)
        F = mapper.mapping_matrix(self.kwargs_lens)
        row_sums = np.sum(F, axis=1)
        # Each row should sum to either 0 (outside mesh) or 1 (inside)
        for s in row_sums:
            assert s == pytest.approx(0.0, abs=1e-10) or s == pytest.approx(
                1.0, abs=1e-10
            )

    def test_sparse_mapping_matrix(self):
        mapper = Mapper(self.mesh, self.lens_model, self.x_img, self.y_img)
        F_dense = mapper.mapping_matrix(self.kwargs_lens)
        F_sparse = mapper.mapping_matrix_sparse(self.kwargs_lens)
        npt.assert_array_almost_equal(F_dense, F_sparse.toarray())


class TestPixelatedProfile:
    def setup_method(self):
        self.mesh = RectangularMesh(
            nx=4, ny=4, x_min=-1, x_max=1, y_min=-1, y_max=1
        )
        self.reg = GradientRegularization(coefficient=1.0)
        self.profile = Pixelated(mesh=self.mesh, regularization=self.reg)

    def test_n_pixels(self):
        assert self.profile.n_pixels == 16

    def test_function(self):
        amp = np.ones(16) * 2.0
        xc, yc = self.mesh.pixel_centers
        flux = self.profile.function(xc, yc, amp)
        npt.assert_array_almost_equal(flux, np.full(16, 2.0))

    def test_function_split_length(self):
        amp = np.ones(16)
        xc, yc = self.mesh.pixel_centers
        basis = self.profile.function_split(xc, yc, amp)
        assert len(basis) == 16

    def test_function_split_sum(self):
        """Sum of all basis functions weighted by amps should equal function()."""
        amp = np.random.rand(16)
        xc, yc = self.mesh.pixel_centers
        basis = self.profile.function_split(xc, yc, amp=np.ones(16))
        total = sum(amp[j] * basis[j] for j in range(16))
        expected = self.profile.function(xc, yc, amp)
        npt.assert_array_almost_equal(total, expected)

    def test_regularization_matrix(self):
        U = self.profile.regularization_matrix()
        assert U.shape == (16, 16)


class TestPixelatedLightModel:
    def test_registration(self):
        """PIXELATED profile should be registered in LightModel."""
        mesh = RectangularMesh(3, 3, -1, 1, -1, 1)
        reg = GradientRegularization()
        source_model = LightModel(
            light_model_list=["PIXELATED"],
            profile_kwargs_list=[{"mesh": mesh, "regularization": reg}],
        )
        assert source_model.profile_type_list[0] == "PIXELATED"

    def test_functions_split(self):
        mesh = RectangularMesh(3, 3, -1, 1, -1, 1)
        reg = GradientRegularization()
        source_model = LightModel(
            light_model_list=["PIXELATED"],
            profile_kwargs_list=[{"mesh": mesh, "regularization": reg}],
        )
        xc, yc = mesh.pixel_centers
        kwargs_source = [{"amp": np.ones(9), "reg_strength": 1.0}]
        response, n = source_model.functions_split(xc, yc, kwargs_source)
        assert n == 9
        assert len(response) == 9

    def test_num_param_linear(self):
        mesh = RectangularMesh(4, 4, -1, 1, -1, 1)
        reg = GradientRegularization()
        source_model = LightModel(
            light_model_list=["PIXELATED"],
            profile_kwargs_list=[{"mesh": mesh, "regularization": reg}],
        )
        kwargs_source = [{"amp": np.ones(16), "reg_strength": 1.0}]
        assert source_model.num_param_linear(kwargs_source) == 16

    def test_update_linear(self):
        mesh = RectangularMesh(3, 3, -1, 1, -1, 1)
        reg = GradientRegularization()
        source_model = LightModel(
            light_model_list=["PIXELATED"],
            profile_kwargs_list=[{"mesh": mesh, "regularization": reg}],
        )
        kwargs_source = [{"amp": np.zeros(9), "reg_strength": 1.0}]
        param = np.arange(9, dtype=float)
        kwargs_out, i = source_model.update_linear(param, 0, kwargs_source)
        npt.assert_array_equal(kwargs_out[0]["amp"], param)
        assert i == 9


class TestPixelatedOperator:
    def setup_method(self):
        self.mesh = RectangularMesh(
            nx=5, ny=5, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5
        )
        self.reg = GradientRegularization(coefficient=1.0)
        self.lens_model = LensModel(lens_model_list=["SIS"])
        self.kwargs_lens = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]

        x_img = np.linspace(-2, 2, 10)
        y_img = np.linspace(-2, 2, 10)
        xx, yy = np.meshgrid(x_img, y_img)
        self.x_img = xx.ravel()
        self.y_img = yy.ravel()
        self.mapper = Mapper(self.mesh, self.lens_model, self.x_img, self.y_img)

    def test_response_matrix(self):
        op = PixelatedOperator(
            self.mapper, self.mesh, self.reg, image_shape=(10, 10)
        )
        F = op.response_matrix(self.kwargs_lens)
        assert F.shape == (100, 25)

    def test_solve_returns_dict(self):
        op = PixelatedOperator(
            self.mapper, self.mesh, self.reg, image_shape=(10, 10)
        )
        data = np.random.rand(10, 10)
        noise = np.ones((10, 10)) * 0.1
        result = op.solve(data, noise, self.kwargs_lens, reg_strength=1.0)
        assert "amplitudes" in result
        assert "model_image" in result
        assert "covariance" in result
        assert len(result["amplitudes"]) == 25

    def test_solve_recovers_simple_source(self):
        """With noiseless data, the solver should recover the source."""
        op = PixelatedOperator(
            self.mapper, self.mesh, self.reg, image_shape=(10, 10)
        )
        # Create a simple source and generate mock data
        true_amps = np.zeros(25)
        true_amps[12] = 1.0  # central pixel
        F = op.response_matrix(self.kwargs_lens)
        mock_data = F @ true_amps
        noise = np.ones(100) * 0.01

        result = op.solve(mock_data, noise, self.kwargs_lens, reg_strength=0.001)
        # The brightest reconstructed pixel should be near the center
        assert np.argmax(result["amplitudes"]) == 12


class TestDelaunayIntegration:
    def test_delaunay_with_gradient_regularization(self):
        mesh = DelaunayMesh.from_regular_grid(4, 4, -1, 1, -1, 1)
        reg = GradientRegularization(coefficient=1.0)
        profile = Pixelated(mesh=mesh, regularization=reg)

        xc, yc = mesh.pixel_centers
        amp = np.ones(mesh.num_pixels)
        flux = profile.function(xc, yc, amp)
        npt.assert_array_almost_equal(flux, np.ones(mesh.num_pixels))

    def test_delaunay_with_adaptive_regularization(self):
        mesh = DelaunayMesh.from_regular_grid(4, 4, -1, 1, -1, 1)
        reg = AdaptiveBrightnessRegularization(
            inner_coefficient=0.01, outer_coefficient=1.0
        )
        signals = np.random.rand(mesh.num_pixels)
        reg.set_pixel_signals(signals)
        U = reg.regularization_matrix(mesh)
        assert U.shape == (mesh.num_pixels, mesh.num_pixels)
        npt.assert_array_almost_equal(U, U.T)

    def test_delaunay_from_image_positions(self):
        lens_model = LensModel(lens_model_list=["SIS"])
        kwargs_lens = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
        x_img = np.linspace(-2, 2, 10)
        y_img = np.linspace(-2, 2, 10)
        xx, yy = np.meshgrid(x_img, y_img)
        mesh = DelaunayMesh.from_image_positions(
            xx.ravel(), yy.ravel(), kwargs_lens, lens_model
        )
        assert mesh.num_pixels > 100  # 100 image pixels + boundary
