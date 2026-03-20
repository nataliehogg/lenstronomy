"""Pixelated source reconstruction operator.

Combines the Mapper, a PSF convolution, a regularization scheme, and a
linear solver into a single class that solves for source pixel amplitudes
given data and a lens model.

This provides the core engine for the PIXELATED light profile type and
can also be used standalone.
"""

import numpy as np

__all__ = ["PixelatedOperator"]


class PixelatedOperator:
    """Solves for optimal source pixel amplitudes via regularised linear inversion.

    Solves the system:
        (F^T C_D^{-1} F + lambda * U) a = F^T C_D^{-1} d

    where:
    - F is the response matrix (mapping + PSF convolution)
    - C_D is the data covariance (diagonal: noise variance per pixel)
    - U is the regularization matrix
    - lambda is the regularization strength
    - d is the data vector
    - a is the vector of source pixel amplitudes

    :param mapper: Mapper instance
    :param mesh: MeshBase instance
    :param regularization: RegularizationBase instance
    :param psf_kernel: 2d PSF kernel array (can be None for no convolution)
    :param image_shape: tuple (ny, nx) shape of the image
    """

    def __init__(self, mapper, mesh, regularization, psf_kernel=None,
                 image_shape=None):
        self._mapper = mapper
        self._mesh = mesh
        self._regularization = regularization
        self._psf_kernel = psf_kernel
        self._image_shape = image_shape

    def response_matrix(self, kwargs_lens):
        """Build the full response matrix: mapping matrix with PSF convolution.

        Each column j of the response matrix is the image you would see if
        only source pixel j had unit amplitude, after lensing and PSF convolution.

        :param kwargs_lens: lens model keyword arguments
        :return: (N_image, N_source) numpy array
        """
        F = self._mapper.mapping_matrix(kwargs_lens)

        if self._psf_kernel is not None and self._image_shape is not None:
            F = self._convolve_columns(F)
        return F

    def _convolve_columns(self, F):
        """Convolve each column of the mapping matrix with the PSF.

        :param F: (N_image, N_source) mapping matrix
        :return: (N_image, N_source) convolved response matrix
        """
        from scipy.signal import fftconvolve

        ny, nx = self._image_shape
        n_src = F.shape[1]
        F_conv = np.zeros_like(F)
        kernel = self._psf_kernel

        ky, kx = kernel.shape
        # Padding for convolution output to match original image size
        pad_y = ky // 2
        pad_x = kx // 2

        for j in range(n_src):
            col_2d = F[:, j].reshape(ny, nx)
            convolved = fftconvolve(col_2d, kernel, mode="same")
            F_conv[:, j] = convolved.ravel()
        return F_conv

    def solve(self, data, noise_map, kwargs_lens, reg_strength=None,
              subtract_model=None):
        """Solve for source pixel amplitudes.

        :param data: 2d or 1d array of observed image data
        :param noise_map: 2d or 1d array of noise RMS per pixel
        :param kwargs_lens: lens model keyword arguments
        :param reg_strength: float, regularization strength lambda.
            If None, uses default coefficient from the regularization object.
        :param subtract_model: optional 2d or 1d array to subtract from data
            before solving (e.g., lens light model)
        :return: dict with keys:
            - 'amplitudes': 1d array of source pixel amplitudes
            - 'model_image': 1d array of the reconstructed image (F @ a)
            - 'covariance': (N_source, N_source) covariance matrix of amplitudes
            - 'regularization_matrix': the U matrix used
            - 'response_matrix': the F matrix used
            - 'log_evidence': Bayesian log-evidence for model comparison
        """
        d = np.asarray(data, dtype=float).ravel()
        sigma = np.asarray(noise_map, dtype=float).ravel()

        if subtract_model is not None:
            d = d - np.asarray(subtract_model, dtype=float).ravel()

        # Inverse variance weights
        C_D_inv = 1.0 / sigma**2

        # Build response matrix
        F = self.response_matrix(kwargs_lens)

        # Build regularization matrix
        U = self._regularization.regularization_matrix(self._mesh)

        if reg_strength is None:
            reg_strength = 1.0

        # Compute M = F^T C_D^{-1} F
        # Efficiently: M_ij = sum_k F_ki * C_D_inv_k * F_kj
        FC = F * C_D_inv[:, np.newaxis]  # F weighted by inverse variance
        M = FC.T @ F

        # Compute b = F^T C_D^{-1} d
        b = FC.T @ d

        # Solve (M + lambda * U) a = b
        M_reg = M + reg_strength * U
        amplitudes, cov = self._solve_regularised(M_reg, b)

        # Reconstructed image
        model_image = F @ amplitudes

        # Log-evidence (Suyu et al. 2006)
        log_evi = self._log_evidence(
            d, model_image, C_D_inv, amplitudes, M, U, reg_strength
        )

        return {
            "amplitudes": amplitudes,
            "model_image": model_image,
            "covariance": cov,
            "regularization_matrix": U,
            "response_matrix": F,
            "log_evidence": log_evi,
        }

    @staticmethod
    def _solve_regularised(M_reg, b):
        """Solve the regularised linear system.

        :param M_reg: (N_source, N_source) matrix M + lambda * U
        :param b: (N_source,) data vector
        :return: (amplitudes, covariance_matrix)
        """
        import sys

        cond = np.linalg.cond(M_reg)
        if cond < 5 / sys.float_info.epsilon:
            try:
                cov = np.linalg.inv(M_reg)
                amplitudes = cov @ b
            except np.linalg.LinAlgError:
                n = M_reg.shape[0]
                amplitudes = np.zeros(n)
                cov = np.zeros_like(M_reg)
        else:
            n = M_reg.shape[0]
            amplitudes = np.zeros(n)
            cov = np.zeros_like(M_reg)
        return amplitudes, cov

    @staticmethod
    def _log_evidence(d, model, C_D_inv, amplitudes, M, U, reg_strength):
        """Compute the Bayesian log-evidence for regularization strength selection.

        log(E) = -0.5 * chi^2 - 0.5 * lambda * a^T U a
                 + 0.5 * N_s * log(lambda)
                 + 0.5 * log(det(U))
                 - 0.5 * log(det(M + lambda * U))
                 + const

        :return: float, log-evidence
        """
        residual = d - model
        chi2 = np.sum(residual**2 * C_D_inv)

        reg_term = amplitudes @ (U @ amplitudes)

        n_src = len(amplitudes)
        M_reg = M + reg_strength * U

        sign_reg, logdet_reg = np.linalg.slogdet(M_reg)
        if sign_reg <= 0:
            return -1e15

        # For the log(det(U)) term, handle potential singularity
        sign_U, logdet_U = np.linalg.slogdet(U)
        if sign_U <= 0:
            # Regularization matrix is singular (e.g., identity-based);
            # use a pseudodet via eigenvalues
            eigvals = np.linalg.eigvalsh(U)
            pos_eigvals = eigvals[eigvals > 1e-30]
            logdet_U = np.sum(np.log(pos_eigvals)) if len(pos_eigvals) > 0 else 0.0

        log_evi = (
            -0.5 * chi2
            - 0.5 * reg_strength * reg_term
            + 0.5 * n_src * np.log(reg_strength)
            + 0.5 * logdet_U
            - 0.5 * logdet_reg
        )
        return log_evi

    def response_split(self, kwargs_lens):
        """Return response matrix columns as a list of 1d arrays.

        This matches the interface expected by ``LinearBasis.functions_split()``
        so that each source pixel acts as one linear basis function.

        :param kwargs_lens: lens model keyword arguments
        :return: list of 1d arrays (one per source pixel), length N_source
        """
        F = self.response_matrix(kwargs_lens)
        return [F[:, j] for j in range(F.shape[1])]
