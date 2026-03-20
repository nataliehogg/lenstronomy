import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.ImSim.SourceReconstruction.mesh import RectangularMesh, DelaunayMesh


class TestRectangularMesh:
    def setup_method(self):
        self.mesh = RectangularMesh(
            nx=5, ny=4, x_min=-1.0, x_max=1.0, y_min=-0.8, y_max=0.8
        )

    def test_num_pixels(self):
        assert self.mesh.num_pixels == 20

    def test_pixel_centers_shape(self):
        xc, yc = self.mesh.pixel_centers
        assert len(xc) == 20
        assert len(yc) == 20

    def test_pixel_centers_range(self):
        xc, yc = self.mesh.pixel_centers
        assert np.min(xc) > -1.0
        assert np.max(xc) < 1.0
        assert np.min(yc) > -0.8
        assert np.max(yc) < 0.8

    def test_neighbor_list_length(self):
        neighbors = self.mesh.neighbor_list()
        assert len(neighbors) == 20

    def test_corner_pixel_neighbors(self):
        neighbors = self.mesh.neighbor_list()
        # Bottom-left corner (index 0) should have 2 neighbors
        assert len(neighbors[0]) == 2

    def test_interior_pixel_neighbors(self):
        neighbors = self.mesh.neighbor_list()
        # Interior pixel should have 4 neighbors
        # Pixel at (ix=2, iy=1) -> index = 1*5 + 2 = 7
        assert len(neighbors[7]) == 4

    def test_interpolation_inside(self):
        # Point at the center of the grid
        pix_idx, weights = self.mesh.interpolation_weights(
            np.array([0.0]), np.array([0.0])
        )
        assert len(pix_idx[0]) == 1
        npt.assert_almost_equal(weights[0][0], 1.0)

    def test_interpolation_outside(self):
        # Point far outside the grid
        pix_idx, weights = self.mesh.interpolation_weights(
            np.array([10.0]), np.array([10.0])
        )
        assert len(pix_idx[0]) == 0

    def test_pixel_index(self):
        # Center of bottom-left pixel
        dx = 2.0 / 5  # pixel width in x
        dy = 1.6 / 4  # pixel width in y
        x = -1.0 + dx / 2
        y = -0.8 + dy / 2
        idx = self.mesh.pixel_index(np.array([x]), np.array([y]))
        assert idx[0] == 0

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            RectangularMesh(nx=1, ny=1, x_min=-1, x_max=1, y_min=-1, y_max=1)


class TestDelaunayMesh:
    def setup_method(self):
        # Create a simple grid of nodes
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        xx, yy = np.meshgrid(x, y)
        self.mesh = DelaunayMesh(xx.ravel(), yy.ravel())

    def test_num_pixels(self):
        assert self.mesh.num_pixels == 25

    def test_pixel_centers(self):
        xc, yc = self.mesh.pixel_centers
        assert len(xc) == 25
        assert len(yc) == 25

    def test_neighbor_list(self):
        neighbors = self.mesh.neighbor_list()
        assert len(neighbors) == 25
        # Each node should have at least 2 neighbors
        for nbrs in neighbors:
            assert len(nbrs) >= 2

    def test_neighbor_symmetry(self):
        neighbors = self.mesh.neighbor_list()
        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                assert i in neighbors[j], (
                    f"Asymmetric adjacency: {i} -> {j} but not {j} -> {i}"
                )

    def test_interpolation_inside(self):
        pix_idx, weights = self.mesh.interpolation_weights(
            np.array([0.0]), np.array([0.0])
        )
        # Should find an enclosing triangle with 3 vertices
        assert len(pix_idx[0]) == 3
        npt.assert_almost_equal(np.sum(weights[0]), 1.0)

    def test_interpolation_at_node(self):
        # Point exactly at a node should have weight ~1 for that node
        xc, yc = self.mesh.pixel_centers
        pix_idx, weights = self.mesh.interpolation_weights(
            np.array([xc[0]]), np.array([yc[0]])
        )
        assert len(pix_idx[0]) == 3
        npt.assert_almost_equal(np.sum(weights[0]), 1.0)

    def test_interpolation_outside(self):
        pix_idx, weights = self.mesh.interpolation_weights(
            np.array([10.0]), np.array([10.0])
        )
        assert len(pix_idx[0]) == 0

    def test_from_regular_grid(self):
        mesh = DelaunayMesh.from_regular_grid(
            nx=4, ny=4, x_min=-1, x_max=1, y_min=-1, y_max=1
        )
        assert mesh.num_pixels == 16

    def test_too_few_nodes(self):
        with pytest.raises(ValueError):
            DelaunayMesh(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_barycentric_weights_sum(self):
        """Barycentric weights should sum to 1 for points inside triangles."""
        x_test = np.linspace(-0.8, 0.8, 10)
        y_test = np.linspace(-0.8, 0.8, 10)
        xx, yy = np.meshgrid(x_test, y_test)
        pix_idx, weights = self.mesh.interpolation_weights(xx.ravel(), yy.ravel())
        for i in range(len(xx.ravel())):
            if len(pix_idx[i]) > 0:
                npt.assert_almost_equal(np.sum(weights[i]), 1.0, decimal=10)
