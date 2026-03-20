"""Mapper: builds the response matrix mapping source pixels to image pixels.

The mapper takes image-plane pixel coordinates, ray-traces them through the
lens model to the source plane, and computes the sparse mapping matrix F
(N_image x N_source) that relates source pixel amplitudes to observed image
pixel values. Each column of F, when convolved with the PSF, gives the
image-plane response of a single source pixel.
"""

import numpy as np

__all__ = ["Mapper"]


class Mapper:
    """Builds the mapping/response matrix from a source mesh to image pixels.

    :param mesh: MeshBase instance defining the source-plane pixelisation
    :param lens_model: LensModel instance
    :param image_x: 1d array of image-plane x coordinates
    :param image_y: 1d array of image-plane y coordinates
    """

    def __init__(self, mesh, lens_model, image_x, image_y):
        self._mesh = mesh
        self._lens_model = lens_model
        self._image_x = np.asarray(image_x, dtype=float).ravel()
        self._image_y = np.asarray(image_y, dtype=float).ravel()

    @property
    def n_image_pixels(self):
        """int: number of image-plane pixels."""
        return len(self._image_x)

    @property
    def n_source_pixels(self):
        """int: number of source-plane pixels."""
        return self._mesh.num_pixels

    def mapping_matrix(self, kwargs_lens):
        """Compute the mapping matrix F (N_image x N_source).

        F[i, j] is the weight with which source pixel j contributes to
        image pixel i, before PSF convolution.

        :param kwargs_lens: lens model keyword arguments
        :return: (N_image, N_source) numpy array
        """
        # Ray-trace image pixels to source plane
        x_src, y_src = self._lens_model.ray_shooting(
            self._image_x, self._image_y, kwargs_lens
        )

        # Compute interpolation weights
        pixel_indices, weights = self._mesh.interpolation_weights(x_src, y_src)

        # Build the mapping matrix
        n_img = self.n_image_pixels
        n_src = self.n_source_pixels
        F = np.zeros((n_img, n_src))
        for i in range(n_img):
            idx = pixel_indices[i]
            w = weights[i]
            if len(idx) > 0:
                F[i, idx] = w
        return F

    def mapping_matrix_sparse(self, kwargs_lens):
        """Compute the mapping matrix as a scipy sparse CSR matrix.

        More memory-efficient for large problems.

        :param kwargs_lens: lens model keyword arguments
        :return: scipy.sparse.csr_matrix of shape (N_image, N_source)
        """
        from scipy import sparse

        x_src, y_src = self._lens_model.ray_shooting(
            self._image_x, self._image_y, kwargs_lens
        )
        pixel_indices, weights = self._mesh.interpolation_weights(x_src, y_src)

        rows = []
        cols = []
        data = []
        for i in range(self.n_image_pixels):
            idx = pixel_indices[i]
            w = weights[i]
            if len(idx) > 0:
                rows.extend([i] * len(idx))
                cols.extend(idx)
                data.extend(w)

        F = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_image_pixels, self.n_source_pixels),
        )
        return F
