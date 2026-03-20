"""Pixelated source light profile for regularised source reconstruction.

This profile represents the source as a set of pixels on a mesh (rectangular
or Delaunay triangulation) in the source plane. Each pixel acts as a linear
basis function whose amplitude is solved for via regularised linear inversion.

The mesh geometry and regularization scheme are set at construction time via
profile_kwargs. At evaluation time, source-plane coordinates (already
ray-traced from the image plane) are mapped to source pixels using the
mesh's interpolation scheme.
"""

import numpy as np

__all__ = ["Pixelated"]


class Pixelated:
    """Pixelated source profile for use with the LightModel framework.

    The mesh and regularization are configuration, passed at construction.
    The linear parameter is ``amp`` (array of source pixel amplitudes).
    The non-linear parameter ``reg_strength`` controls the regularization
    strength and can be fixed or sampled.

    :param mesh: MeshBase instance (RectangularMesh or DelaunayMesh)
    :param regularization: RegularizationBase instance
    """

    param_names = ["amp", "reg_strength"]
    lower_limit_default = {"reg_strength": 1e-6}
    upper_limit_default = {"reg_strength": 1e6}

    def __init__(self, mesh, regularization):
        self._mesh = mesh
        self._regularization = regularization
        self._n_pixels = mesh.num_pixels

    @property
    def mesh(self):
        """MeshBase: the source-plane mesh."""
        return self._mesh

    @property
    def regularization(self):
        """RegularizationBase: the regularization scheme."""
        return self._regularization

    @property
    def n_pixels(self):
        """int: number of source pixels."""
        return self._n_pixels

    def function(self, x, y, amp, reg_strength=1.0):
        """Evaluate the pixelated source at given source-plane positions.

        :param x: 1d array of source-plane x coordinates (ray-traced)
        :param y: 1d array of source-plane y coordinates (ray-traced)
        :param amp: 1d array of pixel amplitudes (length = n_pixels)
        :param reg_strength: regularization strength (unused in evaluation)
        :return: 1d array of surface brightness values
        """
        amp = np.atleast_1d(amp)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))

        pixel_indices, weights = self._mesh.interpolation_weights(x, y)

        flux = np.zeros_like(x)
        for i in range(len(x)):
            idx = pixel_indices[i]
            w = weights[i]
            if len(idx) > 0:
                flux[i] = np.sum(w * amp[idx])
        return flux

    def function_split(self, x, y, amp, reg_strength=1.0):
        """Return the response of each source pixel as a separate basis function.

        For each source pixel j, returns a 1d array giving the interpolation
        weight of pixel j at each input position. This is the j-th column of
        the mapping matrix.

        :param x: 1d array of source-plane x coordinates (ray-traced)
        :param y: 1d array of source-plane y coordinates (ray-traced)
        :param amp: array of amplitudes (values are ignored; set amp=1 for
            unit-amplitude basis functions)
        :param reg_strength: regularization strength (unused here)
        :return: list of 1d arrays, one per source pixel
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))

        pixel_indices, weights = self._mesh.interpolation_weights(x, y)

        # Build columns of mapping matrix
        n_pts = len(x)
        basis = [np.zeros(n_pts) for _ in range(self._n_pixels)]
        for i in range(n_pts):
            idx = pixel_indices[i]
            w = weights[i]
            for k in range(len(idx)):
                basis[idx[k]][i] += w[k]
        return basis

    def regularization_matrix(self):
        """Construct the regularization matrix for this profile's mesh.

        :return: (n_pixels, n_pixels) numpy array
        """
        return self._regularization.regularization_matrix(self._mesh)
