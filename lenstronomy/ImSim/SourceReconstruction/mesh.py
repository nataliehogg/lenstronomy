"""Mesh classes for pixelised source plane reconstruction.

Provides abstract base class and concrete implementations for defining
the source plane pixelisation geometry used in regularised linear inversions.

Currently supported mesh types:
- Rectangular: uniform rectangular grid
- Delaunay: adaptive triangulation based on scipy.spatial.Delaunay
"""

import abc
import numpy as np

__all__ = ["RectangularMesh", "DelaunayMesh"]


class MeshBase(abc.ABC):
    """Abstract base class for source plane meshes.

    A mesh defines a pixelisation of the source plane: the geometry of the
    cells/pixels, their centres, their adjacency structure, and how to
    interpolate a source-plane position into pixel weights.
    """

    @property
    @abc.abstractmethod
    def num_pixels(self):
        """int: total number of source pixels/cells."""
        pass

    @property
    @abc.abstractmethod
    def pixel_centers(self):
        """tuple of (x, y) arrays: coordinates of each pixel centre."""
        pass

    @abc.abstractmethod
    def neighbor_list(self):
        """Return adjacency structure.

        :return: list of lists, where neighbor_list()[i] gives the indices
            of pixels adjacent to pixel i.
        """
        pass

    @abc.abstractmethod
    def interpolation_weights(self, x, y):
        """Compute which pixel(s) each source-plane point falls in and the
        associated interpolation weights.

        :param x: 1d array of source-plane x coordinates
        :param y: 1d array of source-plane y coordinates
        :return: tuple (pixel_indices, weights) where:
            - pixel_indices is a list of arrays, one per input point, giving
              the pixel indices that contribute
            - weights is a list of arrays giving the corresponding weights
              (summing to 1 for points inside the mesh)
        """
        pass


class RectangularMesh(MeshBase):
    """Uniform rectangular grid mesh for the source plane.

    Pixels are indexed in row-major order: pixel index = iy * nx + ix,
    where ix runs from 0 to nx-1 (left to right) and iy from 0 to ny-1
    (bottom to top).

    :param nx: number of pixels in x direction
    :param ny: number of pixels in y direction
    :param x_min: left edge of the grid in arcsec
    :param x_max: right edge of the grid in arcsec
    :param y_min: bottom edge of the grid in arcsec
    :param y_max: top edge of the grid in arcsec
    """

    def __init__(self, nx, ny, x_min, x_max, y_min, y_max):
        if nx < 2 or ny < 2:
            raise ValueError("Rectangular mesh requires nx >= 2 and ny >= 2.")
        self._nx = int(nx)
        self._ny = int(ny)
        self._x_min = float(x_min)
        self._x_max = float(x_max)
        self._y_min = float(y_min)
        self._y_max = float(y_max)
        self._dx = (self._x_max - self._x_min) / self._nx
        self._dy = (self._y_max - self._y_min) / self._ny

        # Pixel centres
        x_centers = self._x_min + (np.arange(self._nx) + 0.5) * self._dx
        y_centers = self._y_min + (np.arange(self._ny) + 0.5) * self._dy
        self._xc, self._yc = np.meshgrid(x_centers, y_centers)
        self._xc = self._xc.ravel()
        self._yc = self._yc.ravel()

    @property
    def nx(self):
        """int: number of pixels in x direction."""
        return self._nx

    @property
    def ny(self):
        """int: number of pixels in y direction."""
        return self._ny

    @property
    def pixel_width_x(self):
        """float: pixel width in x direction (arcsec)."""
        return self._dx

    @property
    def pixel_width_y(self):
        """float: pixel width in y direction (arcsec)."""
        return self._dy

    @property
    def num_pixels(self):
        return self._nx * self._ny

    @property
    def pixel_centers(self):
        return self._xc, self._yc

    def neighbor_list(self):
        neighbors = []
        for idx in range(self.num_pixels):
            iy, ix = divmod(idx, self._nx)
            nbrs = []
            if ix > 0:
                nbrs.append(iy * self._nx + (ix - 1))
            if ix < self._nx - 1:
                nbrs.append(iy * self._nx + (ix + 1))
            if iy > 0:
                nbrs.append((iy - 1) * self._nx + ix)
            if iy < self._ny - 1:
                nbrs.append((iy + 1) * self._nx + ix)
            neighbors.append(nbrs)
        return neighbors

    def interpolation_weights(self, x, y):
        """Nearest-grid-point assignment for rectangular mesh.

        Each source-plane point is assigned to its enclosing pixel with
        weight 1. Points outside the grid get empty arrays.

        :param x: 1d array of source-plane x coordinates
        :param y: 1d array of source-plane y coordinates
        :return: (pixel_indices, weights) lists of arrays
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))

        # Fractional pixel coordinates
        fx = (x - self._x_min) / self._dx - 0.5
        fy = (y - self._y_min) / self._dy - 0.5

        # Nearest pixel
        ix = np.clip(np.round(fx).astype(int), 0, self._nx - 1)
        iy = np.clip(np.round(fy).astype(int), 0, self._ny - 1)

        # Check bounds
        inside = (
            (x >= self._x_min)
            & (x <= self._x_max)
            & (y >= self._y_min)
            & (y <= self._y_max)
        )

        pixel_indices = []
        weights = []
        for i in range(len(x)):
            if inside[i]:
                pixel_indices.append(np.array([iy[i] * self._nx + ix[i]]))
                weights.append(np.array([1.0]))
            else:
                pixel_indices.append(np.array([], dtype=int))
                weights.append(np.array([]))
        return pixel_indices, weights

    def pixel_index(self, x, y):
        """Return the pixel index for given source-plane coordinates.

        :param x: x coordinate(s) in source plane
        :param y: y coordinate(s) in source plane
        :return: integer pixel index array (-1 for points outside the grid)
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        ix = np.floor((x - self._x_min) / self._dx).astype(int)
        iy = np.floor((y - self._y_min) / self._dy).astype(int)
        ix = np.clip(ix, 0, self._nx - 1)
        iy = np.clip(iy, 0, self._ny - 1)
        idx = iy * self._nx + ix
        outside = (
            (x < self._x_min)
            | (x > self._x_max)
            | (y < self._y_min)
            | (y > self._y_max)
        )
        idx[outside] = -1
        return idx


class DelaunayMesh(MeshBase):
    """Delaunay triangulation mesh for the source plane.

    The mesh is built from a set of node positions in the source plane.
    Each triangle defines a pixel/cell. Source-plane positions are
    interpolated using barycentric coordinates within their enclosing
    triangle.

    The node positions can be provided directly, or generated automatically
    on a regular grid. For adaptive meshes, nodes can be placed at the
    ray-traced positions of image-plane pixels (traced through the lens
    model), concentrating resolution where magnification is high.

    :param x_nodes: 1d array of x coordinates of mesh nodes
    :param y_nodes: 1d array of y coordinates of mesh nodes
    """

    def __init__(self, x_nodes, y_nodes):
        from scipy.spatial import Delaunay

        self._x_nodes = np.asarray(x_nodes, dtype=float)
        self._y_nodes = np.asarray(y_nodes, dtype=float)
        if len(self._x_nodes) < 3:
            raise ValueError("Delaunay mesh requires at least 3 nodes.")
        points = np.column_stack([self._x_nodes, self._y_nodes])
        self._tri = Delaunay(points)
        self._neighbors_cache = None

    @property
    def num_pixels(self):
        """Number of pixels equals number of nodes (vertex-based reconstruction)."""
        return len(self._x_nodes)

    @property
    def pixel_centers(self):
        return self._x_nodes.copy(), self._y_nodes.copy()

    @property
    def triangulation(self):
        """scipy.spatial.Delaunay: the underlying triangulation object."""
        return self._tri

    def neighbor_list(self):
        if self._neighbors_cache is not None:
            return self._neighbors_cache
        n = self.num_pixels
        neighbor_sets = [set() for _ in range(n)]
        for simplex in self._tri.simplices:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbor_sets[simplex[i]].add(simplex[j])
        self._neighbors_cache = [sorted(s) for s in neighbor_sets]
        return self._neighbors_cache

    def interpolation_weights(self, x, y):
        """Barycentric interpolation within Delaunay triangles.

        For each source-plane point, find the enclosing triangle and compute
        barycentric weights for the three vertices. Points outside the
        convex hull of the nodes get empty arrays.

        :param x: 1d array of source-plane x coordinates
        :param y: 1d array of source-plane y coordinates
        :return: (pixel_indices, weights) lists of arrays
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        points = np.column_stack([x, y])

        simplex_indices = self._tri.find_simplex(points)

        pixel_indices = []
        weights = []
        for i in range(len(x)):
            si = simplex_indices[i]
            if si == -1:
                # Outside the convex hull
                pixel_indices.append(np.array([], dtype=int))
                weights.append(np.array([]))
            else:
                vertex_ids = self._tri.simplices[si]
                # Compute barycentric coordinates
                tri_points = self._tri.points[vertex_ids]
                bary = self._barycentric(
                    points[i, 0], points[i, 1], tri_points
                )
                pixel_indices.append(vertex_ids)
                weights.append(bary)
        return pixel_indices, weights

    @staticmethod
    def _barycentric(px, py, tri_points):
        """Compute barycentric coordinates of point (px, py) in triangle.

        :param px: x coordinate of point
        :param py: y coordinate of point
        :param tri_points: (3, 2) array of triangle vertex coordinates
        :return: array of 3 barycentric weights
        """
        x0, y0 = tri_points[0]
        x1, y1 = tri_points[1]
        x2, y2 = tri_points[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-30:
            return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        return np.array([w0, w1, w2])

    @classmethod
    def from_regular_grid(cls, nx, ny, x_min, x_max, y_min, y_max):
        """Create a Delaunay mesh from a uniform rectangular grid of nodes.

        :param nx: number of nodes in x direction
        :param ny: number of nodes in y direction
        :param x_min: left boundary
        :param x_max: right boundary
        :param y_min: bottom boundary
        :param y_max: top boundary
        :return: DelaunayMesh instance
        """
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(x, y)
        return cls(xx.ravel(), yy.ravel())

    @classmethod
    def from_image_positions(cls, x_image, y_image, kwargs_lens, lens_model,
                             n_boundary=20, boundary_margin=0.1):
        """Create a Delaunay mesh by ray-tracing image pixel positions to the
        source plane, concentrating resolution where magnification is high.

        A boundary ring of nodes is added around the convex hull of the
        traced positions to stabilise the reconstruction at the edges.

        :param x_image: 1d array of image-plane x coordinates
        :param y_image: 1d array of image-plane y coordinates
        :param kwargs_lens: lens model keyword arguments
        :param lens_model: LensModel instance
        :param n_boundary: number of boundary nodes to add
        :param boundary_margin: fractional margin to expand the boundary
        :return: DelaunayMesh instance
        """
        x_src, y_src = lens_model.ray_shooting(x_image, y_image, kwargs_lens)

        # Add boundary nodes
        cx, cy = np.mean(x_src), np.mean(y_src)
        rx = (np.max(x_src) - np.min(x_src)) / 2 * (1 + boundary_margin)
        ry = (np.max(y_src) - np.min(y_src)) / 2 * (1 + boundary_margin)
        r = max(rx, ry)
        angles = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
        x_boundary = cx + r * np.cos(angles)
        y_boundary = cy + r * np.sin(angles)

        x_all = np.concatenate([x_src, x_boundary])
        y_all = np.concatenate([y_src, y_boundary])
        return cls(x_all, y_all)
