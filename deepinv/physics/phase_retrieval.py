from __future__ import annotations
from functools import partial
import math
from typing import Any

import numpy as np
import torch

from deepinv.optim.phase_retrieval import spectral_methods
from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.physics.structured_random import StructuredRandom
from deepinv.utils.decorators import _deprecated_alias


class PhaseRetrieval(Physics):
    r"""
    Phase Retrieval base class corresponding to the operator

    .. math::

        \forw{x} = |Bx|^2.

    The linear operator :math:`B` is defined by a :class:`deepinv.physics.LinearPhysics` object.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param deepinv.physics.forward.LinearPhysics B: the linear forward operator.
    """

    def __init__(
        self,
        B: LinearPhysics,
        measurement: str = "intensity",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "Phase Retrieval"
        self.measurement = measurement

        self.B = B

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Applies the forward operator to the input x.

        Note here the operation includes the modulus operation.

        :param torch.Tensor x: signal/image.
        """
        if self.measurement == "intensity":
            return self.B(x, **kwargs).abs().square()
        elif self.measurement == "amplitude":
            return self.B(x, **kwargs).abs()
        else:
            raise ValueError(
                f"measurement must be either 'intensity' or 'amplitude', got {self.measurement}"
            )

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes a initial reconstruction for the image :math:`x` from the measurements :math:`y` using :class:`deepinv.optim.phase_retrieval.spectral_methods`.

        :param torch.Tensor y: measurements.
        :return: (:class:`torch.Tensor`) an initial reconstruction for image :math:`x`.
        """
        return spectral_methods(y, self, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.B_adjoint(y, **kwargs)

    def B_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.B.A_adjoint(y, **kwargs)

    def B_dagger(self, y):
        r"""
        Computes the linear pseudo-inverse of :math:`B`.

        :param torch.Tensor y: measurements.
        :return: (:class:`torch.Tensor`) the reconstruction image :math:`x`.
        """
        return self.B.A_dagger(y)

    def forward(self, x, **kwargs):
        r"""
        Applies the phase retrieval measurement operator, i.e. :math:`y = \noise{|Bx|^2}` (with noise :math:`N` and/or sensor non-linearities).

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) noisy measurements
        """
        return self.sensor(self.noise(self.A(x, **kwargs)))

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` at the input x, defined as:

        .. math::

            A_{vjp}(x, v) = 2 \overline{B}^{\top} \text{diag}(Bx) v.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (:class:`torch.Tensor`) the VJP product between :math:`v` and the Jacobian.
        """
        if self.measurement == "intensity":
            return 2 * self.B_adjoint(self.B(x) * v)
        elif self.measurement == "amplitude":
            return 2 * self.B_adjoint(self.B(x) / 2 / (self.B(x).abs() + 1e-10) * v)
        else:
            raise ValueError(
                f"measurement must be either 'intensity' or 'amplitude', got {self.measurement}"
            )

    def release_memory(self):
        del self.B
        torch.cuda.empty_cache()
        return


class RandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Random Phase Retrieval forward operator. Creates a random :math:`m \times n` sampling matrix :math:`B` where :math:`n` is the number of elements of the signal and :math:`m` is the number of measurements.

    This class generates a random i.i.d. Gaussian matrix

    .. math::

        B_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param int m: number of measurements.
    :param tuple img_size: shape (C, H, W) of inputs.
    :param str mode: ``gaussian`` for a Gaussian matrix, ``unitary`` for a haar matrix, ``circulant`` for a circulant matrix, ``product`` for a product of Gaussian matrices. Default is ``gaussian``.
    :param int | None product: If ``mode=product``, the number of Gaussian matrices to be multiplied. Default is ``None``.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.dtype dtype: Forward matrix is stored as a dtype. Default is torch.cfloat.
    :param str device: Device to store the forward matrix.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> from deepinv.physics import RandomPhaseRetrieval
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=6, img_size=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[3.8405, 2.2588, 0.0146, 3.0864, 1.8075, 0.1518]])

    """

    def __init__(
        self,
        m: int,
        img_size,
        mode: str = "gaussian",
        product: int | None = None,
        channelwise: bool = False,
        dtype=torch.complex64,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        self.m = m
        self.img_size = img_size
        self.oversampling_ratio = m / torch.prod(torch.tensor(img_size))
        self.channelwise = channelwise
        self.dtype = dtype
        self.device = device
        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            assert rng.device == torch.device(device), (
                f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {self.device}."
            )
            self.rng = rng

        B = CompressedSensing(
            m=m,
            img_size=img_size,
            mode=mode,
            product=product,
            channelwise=channelwise,
            dtype=dtype,
            device=device,
            rng=self.rng,
        )
        super().__init__(B, **kwargs)
        self.register_buffer("initial_random_state", self.rng.get_state())
        self.name = "Random Phase Retrieval"
        self.to(device)

    def get_A_squared_mean(self):
        return self.B._A.var() + self.B._A.mean() ** 2


def compare(img_size: tuple, output_size: tuple) -> str:
    r"""
    Compare input and output shape to determine the sampling mode.

    :param tuple img_size: Input shape (C, H, W).
    :param tuple output_size: Output shape (C, H, W).

    :return: The sampling mode in ["oversampling","undersampling","equisampling].
    """
    h_in = img_size[1]
    w_in = img_size[2]
    h_out = output_size[1]
    w_out = output_size[2]
    if h_in == h_out and w_in == w_out:
        return "equisampling"
    elif h_in <= h_out and w_in <= w_out:
        return "oversampling"
    elif h_in >= h_out and w_in >= w_out:
        return "undersampling"
    else:
        raise ValueError(
            "Does not support different sampling schemes on height and width."
        )


class StructuredRandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Structured random phase retrieval model corresponding to the operator

    .. math::

        A(x) = |\prod_{i=1}^N (F D_i) x|^2,

    where :math:`F` is the Discrete Fourier Transform (DFT) matrix, and :math:`D_i` are diagonal matrices with elements of unit norm and random phases, and :math:`N` refers to the number of layers. It is also possible to replace :math:`x` with :math:`Fx` as an additional 0.5 layer.

    For oversampling, we first pad the input signal with zeros to match the output shape and pass it to :math:`A(x)`. For undersampling, we first pass the signal in its original shape to :math:`A(x)` and trim the output signal to match the output shape.

    The phase of the diagonal elements of the matrices :math:`D_i` are drawn from a uniform distribution in the interval :math:`[0, 2\pi]`.

    :param tuple img_size: shape (C, H, W) of inputs.
    :param tuple output_size: shape (C, H, W) of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math:`F` transform is included, i.e., :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`.
    :param str transform: structured transform to use. Default is 'fft'.
    :param str diagonal_mode: sampling distribution for the diagonal elements. Default is 'uniform_phase'.
    :param bool shared_weights: if True, the same diagonal matrix is used for all layers. Default is False.
    :param torch.dtype dtype: Signals are processed in dtype. Default is torch.cfloat.
    :param str device: Device for computation. Default is `cpu`.
    """

    def __init__(
        self,
        img_size: tuple,
        output_size: tuple,
        middle_size: tuple | None = None,
        n_layers: float | None = None,
        transforms: list[str] | None = None,
        diagonals: list[list[str]] | None = None,  # in the order of math
        diagonal_config: dict | None = None,
        manual_spectrum: str | torch.Tensor = "unit",
        pad_powers_of_two=False,
        shared_weights=False,
        explicit_matrix=False,
        include_zero=False,
        measurement: str = "intensity",
        dtype=torch.complex64,
        device="cpu",
        verbose=False,
        **kwargs,
    ):
        if n_layers is not None:
            assert int(n_layers * 2) - n_layers * 2 == 0, (
                f"The number of layers must be a half-integer, got {n_layers}"
            )
            if transforms is not None:
                assert len(transforms) == math.ceil(n_layers), (
                    f"The number of transforms must match the number of layers, got {len(transforms)} and {n_layers}"
                )
            else:
                transforms = ["fourier2"] * math.ceil(n_layers)
            if diagonals is not None:
                assert len(diagonals) == math.floor(n_layers), (
                    f"The number of diagonals must match the number of layers, got {len(diagonals)} and {n_layers}"
                )
            else:
                diagonals = [["unit", "rademacher"]] * math.floor(n_layers)
        else:
            if transforms is not None:
                if diagonals is not None:
                    assert len(transforms) - len(diagonals) in [0, 1], (
                        f"The number of transforms must be equal to or one more than the number of diagonals, got {len(transforms)} and {len(diagonals)}"
                    )
                    n_layers = (len(transforms) + len(diagonals)) / 2
                else:
                    diagonals = [["unit", "rademacher"]] * len(
                        transforms
                    )  # integer layers
                    n_layers = len(transforms)
            else:
                if diagonals is not None:
                    transforms = ["fourier2"] * len(diagonals)  # half-integer layers
                    n_layers = len(diagonals)
                else:
                    n_layers = 2
                    transforms = ["fourier2"] * n_layers
                    diagonals = [["unit", "rademacher"]] * n_layers

        if manual_spectrum != "unit":
            assert all(diag[0] == "unit" for diag in diagonals), (
                "Manual spectrum should only be used with unit diagonals"
            )

        if diagonal_config is None:
            diagonal_config = {}

        # model shape
        self.img_size = img_size
        self.output_size = output_size
        self.mode = compare(img_size, output_size)
        if middle_size is None:
            for transform in transforms:
                if "hadamard" in transform:
                    pad_powers_of_two = True
            if pad_powers_of_two is True:
                middle_size = 2 ** math.ceil(
                    math.log2(max(img_size[1], output_size[1]))
                )
                self.middle_size = (1, middle_size, middle_size)
            elif self.mode == "oversampling":
                self.middle_size = self.output_size
            else:
                self.middle_size = self.img_size
        else:
            self.middle_size = middle_size
        if verbose:
            print(f"middle shape: {self.middle_size}")

        self.n = torch.prod(torch.tensor(self.img_size))
        self.m = torch.prod(torch.tensor(self.output_size))
        self.oversampling_ratio = self.m / self.n
        self.n_layers = n_layers
        # flip transforms and diagonals order
        self.transforms = transforms[::-1]
        self.diagonals = diagonals[::-1]
        self.diagonal_config = diagonal_config
        self.diagonal_config["alpha"] = self.oversampling_ratio
        self.diagonal_config["include_zero"] = include_zero
        self.structure = self.get_structure(self.n_layers)
        self.shared_weights = shared_weights

        self.dtype = dtype
        self.device = device

        self.spectrum = manual_spectrum

        B = StructuredRandom(
            img_size=self.img_size,
            output_size=self.output_size,
            middle_size=self.middle_size,
            n_layers=self.n_layers,
            spectrum=self.spectrum,
            transforms=self.transforms,
            diagonals=self.diagonals,
            diagonal_config=self.diagonal_config,
            explicit_matrix=explicit_matrix,
            dtype=self.dtype,
            device=self.device,
            # **kwargs,
        )

        super().__init__(B, measurement, **kwargs)
        self.name = "Structured Random Phase Retrieval"
        self.to(device)

    def B_dagger(self, y):
        return self.B.A_adjoint(y)

    def get_A_squared_mean(self):
        if self.n_layers == 0.5:
            print(
                "warning: computing the mean of the squared operator for a single Fourier transform."
            )
            return None
        return self.diagonals[0].var() + self.diagonals[0].mean() ** 2

    @staticmethod
    def get_structure(n_layers) -> str:
        r"""Returns the structure of the operator as a string.

        :param float n_layers: number of layers.

        :return: (str) the structure of the operator, e.g., "FDFD".
        """
        return "FD" * math.floor(n_layers) + "F" * (n_layers % 1 == 0.5)

    def get_singular_values(self):
        r"""Returns the singular values of the forward matrix.

        :return: (torch.Tensor) the singular values.
        """
        return self.B.get_singular_values()

    def get_forward_matrix(self):
        r"""Returns the forward matrix.

        :return: (torch.Tensor) the forward matrix.
        """
        return self.B.forward_matrix

    def partial_forward(self, x, n_layers):
        return self.B.partial_forward(x, n_layers)

    def partial_inverse(self, y, n_layers):
        return self.B.partial_inverse(y, n_layers)

    def get_adversarial(self, n_layers=None, trimmed=True, mag="delta"):
        if n_layers is None:
            n_layers = self.n_layers - 1
        return self.B.get_adversarial(n_layers, trimmed, mag)


class PtychographyLinearOperator(LinearPhysics):
    r"""
    Forward linear operator for phase retrieval in ptychography.

    Models multiple applications of the shifted probe and Fourier transform on an input image.

    This operator performs multiple 2D Fourier transforms on the probe function applied to the shifted input image according to specific offsets, and concatenates them.
    The probe function is applied element by element to the input image.

    .. math::

        B = \left[ \begin{array}{c} B_1 \\ B_2 \\ \vdots \\ B_{n_{\text{img}}} \end{array} \right],
        B_l = F \text{diag}(p) T_l, \quad l = 1, \dots, n_{\text{img}},

    where :math:`F` is the 2D Fourier transform, :math:`\text{diag}(p)` is associated with the probe :math:`p` and :math:`T_l` is a 2D shift.

    :param tuple img_size: Shape of the input image (height, width).
    :param None, torch.Tensor probe: A tensor of shape ``img_size`` representing the probe function. If ``None``, a disk probe is generated with :func:`deepinv.physics.phase_retrieval.build_probe` with disk shape and radius 10.
    :param None, torch.Tensor shifts: A 2D array of shape ``(N, 2)`` corresponding to the ``N`` shift positions for the probe. If ``None``, shifts are generated with :func:`deepinv.physics.phase_retrieval.generate_shifts` with ``N=25``.
    :param torch.device, str device: Device "cpu" or "gpu".

    """

    def __init__(
        self,
        img_size,
        probe=None,
        shifts=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device
        self.img_size = img_size

        if shifts is None:
            self.n_img = 25
            shifts = generate_shifts(img_size=img_size, n_img=self.n_img)
        else:
            self.n_img = len(shifts)

        self.register_buffer("shifts", shifts)

        if probe is None:
            probe = build_probe(
                img_size=img_size, type="disk", probe_radius=10, device=device
            )

        self.register_buffer("init_probe", probe.clone())

        probe = probe / self.get_overlap_img(self.shifts).mean().sqrt()
        probe = torch.cat(
            [self.shift(probe, x_shift, y_shift) for x_shift, y_shift in self.shifts],
            dim=0,
        ).unsqueeze(0)

        self.register_buffer("probe", probe)
        self.to(device)

    def A(self, x, **kwargs):
        """
        Applies the forward operator to the input image ``x`` by shifting the probe,
        multiplying element-wise, and performing a 2D Fourier transform.

        :param torch.Tensor x: Input image tensor.
        :return: Concatenated Fourier transformed tensors after applying shifted probes.
        """
        op_fft2 = partial(torch.fft.fft2, norm="ortho")
        return op_fft2(self.probe * x)

    def A_adjoint(self, y, **kwargs):
        """
        Applies the adjoint operator to ``y``.

        :param torch.Tensor y: Transformed image data tensor of size (batch_size, n_img, height, width).
        :return: Reconstructed image tensor.
        """
        op_ifft2 = partial(torch.fft.ifft2, norm="ortho")
        return (self.probe * op_ifft2(y)).sum(dim=1).unsqueeze(1)

    def shift(self, x, x_shift, y_shift, pad_zeros=True):
        """
        Applies a shift to the tensor ``x`` by ``x_shift`` and ``y_shift``.

        :param torch.Tensor x: Input tensor.
        :param int x_shift: Shift in x-direction.
        :param int y_shift: Shift in y-direction.
        :param bool pad_zeros: If True, pads shifted regions with zeros.
        :return: Shifted tensor.
        """
        x = torch.roll(x, (x_shift, y_shift), dims=(-2, -1))

        if pad_zeros:
            if x_shift < 0:
                x[..., x_shift:, :] = 0
            elif x_shift > 0:
                x[..., 0:x_shift, :] = 0
            if y_shift < 0:
                x[..., :, y_shift:] = 0
            elif y_shift > 0:
                x[..., :, 0:y_shift] = 0
        return x

    def get_overlap_img(self, shifts):
        """
        Computes the overlapping image intensities from probe shifts, used for normalization.

        :param torch.Tensor shifts: Tensor of probe shifts.
        :return: Tensor representing the overlap image.
        """
        overlap_img = torch.zeros_like(self.init_probe, dtype=torch.float32)
        for x_shift, y_shift in shifts:
            overlap_img += torch.abs(self.shift(self.init_probe, x_shift, y_shift)) ** 2
        return overlap_img


class Ptychography(PhaseRetrieval):
    r"""
    Ptychography forward operator.

    Corresponding to the operator

    .. math::

         \forw{x} = \left| Bx \right|^2

    where :math:`B` is the linear forward operator defined by a :class:`deepinv.physics.PtychographyLinearOperator` object.

    :param tuple img_size: Shape of the input image.
    :param None, torch.Tensor probe: A tensor of shape ``img_size`` representing the probe function.
        If None, a disk probe is generated with ``deepinv.physics.phase_retrieval.build_probe`` function.
    :param None, torch.Tensor shifts: A 2D array of shape (``n_img``, 2) corresponding to the shifts for the probe.
        If None, shifts are generated with ``deepinv.physics.phase_retrieval.generate_shifts`` function.
    :param torch.device, str device: Device "cpu" or "gpu".

    |sep|

    :Examples:

    >>> from deepinv.physics import Ptychography
    >>> import torch
    >>> img_size = (1, 64, 64)  # input image
    >>> physics = Ptychography(img_size=img_size)
    >>> x = torch.randn(img_size, dtype=torch.cfloat)
    >>> y = physics(x)  # Apply the Ptychography forward operator
    >>> print(y.shape) # 25 probe positions by default
    torch.Size([1, 25, 64, 64])
    """

    @_deprecated_alias(in_shape="img_size")
    def __init__(
        self,
        img_size=None,
        probe=None,
        shifts=None,
        device="cpu",
        **kwargs,
    ):
        B = PtychographyLinearOperator(
            img_size=img_size,
            probe=probe,
            shifts=shifts,
            device=device,
        )
        self.probe = B.probe
        self.shifts = B.shifts
        self.device = device
        self.img_size = img_size
        super().__init__(B, **kwargs)
        self.name = f"Ptychography_PR"
        self.to(device)


def build_probe(img_size, type="disk", probe_radius=10, device="cpu"):
    """
    Builds a probe based on the specified type and radius.

    :param tuple img_size: Shape of the input image.
    :param str type: Type of probe shape, e.g., "disk".
    :param int probe_radius: Radius of the probe shape.
    :param torch.device device: Device "cpu" or "gpu".
    :return: Tensor representing the constructed probe.
    """
    if type == "disk" or type is None:
        x = torch.arange(img_size[1], dtype=torch.float64)
        y = torch.arange(img_size[2], dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        probe = torch.zeros(img_size, device=device)
        probe[
            torch.sqrt((X - img_size[1] // 2) ** 2 + (Y - img_size[2] // 2) ** 2)
            .unsqueeze(0)
            .expand(img_size[0], -1, -1)
            < probe_radius
        ] = 1
    else:
        raise NotImplementedError(f"Probe type {type} not implemented")
    return probe


def generate_shifts(
    img_size: Any, n_img: int = 25, fov: int | None = None
) -> torch.Tensor:
    """
    Generates the array of probe shifts across the image.
    Based on probe radius and field of view.

    :param img_size: Size of the image.
    :param int n_img: Number of shifts (must be a perfect square).
    :param int fov: Field of view for shift computation.
    :return: Array of (x, y) shifts.
    """
    if fov is None:
        fov = img_size[-1]
    start_shift = -fov // 2
    end_shift = fov // 2

    if n_img != int(np.sqrt(n_img)) ** 2:
        raise ValueError("n_img needs to be a perfect square")

    side_n_img = int(np.sqrt(n_img))
    shifts = torch.linspace(start_shift, end_shift, side_n_img).to(torch.int32)
    y_shifts, x_shifts = torch.meshgrid(shifts, shifts, indexing="ij")
    return torch.concatenate(
        [x_shifts.reshape(n_img, 1), y_shifts.reshape(n_img, 1)], dim=1
    )
