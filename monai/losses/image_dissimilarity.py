# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union
import math

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.networks.layers import gaussian_1d, separable_filtering
from monai.utils import LossReduction, deprecated_arg
from monai.utils.module import look_up_option


def make_rectangular_kernel(kernel_size: int) -> torch.Tensor:
    return torch.ones(kernel_size)


def make_triangular_kernel(kernel_size: int) -> torch.Tensor:
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)


def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma=sigma, truncated=kernel_size // 2, approx="sampled", normalize=False) * (
        2.5066282 * sigma
    )
    return kernel[:kernel_size]


kernel_dict = {
    "rectangular": make_rectangular_kernel,
    "triangular": make_triangular_kernel,
    "gaussian": make_gaussian_kernel,
}


def parzen_windowing(
    pred: torch.Tensor, target: torch.Tensor, kernel_type: str, num_bins: int, preterm: torch.Tensor, bin_centers: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if kernel_type == "gaussian":
        pred_weight = parzen_windowing_gaussian(pred, preterm, bin_centers)
        target_weight = parzen_windowing_gaussian(target, preterm, bin_centers)
    elif kernel_type == "b-spline":
        # a third order BSpline kernel is used for the pred image intensity PDF.
        pred_weight = parzen_windowing_b_spline(pred, order=3, num_bins=num_bins)
        # a zero order (box car) BSpline kernel is used for the target image intensity PDF.
        target_weight = parzen_windowing_b_spline(target, order=0, num_bins=num_bins)
    else:
        raise ValueError
    return pred_weight, target_weight


def parzen_windowing_b_spline(
    img: torch.Tensor, order: int, num_bins: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parzen windowing with b-spline kernel (adapted from ITK)

    Args:
        img: the shape should be B[NDHW].
        order: int.
    """

    # Compute binsize for the histograms.
    #
    # The binsize for the image intensities needs to be adjusted so that
    # we can avoid dealing with boundary conditions using the cubic
    # spline as the Parzen window.  We do this by increasing the size
    # of the bins so that the joint histogram becomes "padded" at the
    # borders. Because we are changing the binsize,
    # we also need to shift the minimum by the padded amount in order to
    # avoid minimum values filling in our padded region.
    #
    # Note that there can still be non-zero bin values in the padded region,
    # it's just that these bins will never be a central bin for the Parzen
    # window.
    _max, _min = torch.max(img), torch.min(img)
    padding = 2
    bin_size = (_max - _min) / (num_bins - 2 * padding)
    norm_min = torch.div(_min, bin_size) - padding

    # assign bin/window index to each voxel
    window_term = torch.div(img, bin_size) - norm_min  # B[NDHW]
    # make sure the extreme values are in valid (non-padded) bins
    window_term = torch.clamp(window_term, padding, num_bins - padding - 1)  # B[NDHW]
    window_term = window_term.reshape(window_term.shape[0], -1, 1)  # (batch, num_sample, 1)
    bins = torch.arange(num_bins, device=window_term.device).reshape(1, 1, -1)  # (1, 1, num_bins)
    sample_bin_matrix = torch.abs(bins - window_term)  # (batch, num_sample, num_bins)

    # b-spleen kernel
    # (4 - 6 * abs ** 2 + 3 * abs ** 3) / 6 when 0 <= abs < 1
    # (2 - abs) ** 3 / 6 when 1 <= abs < 2
    weight = torch.zeros_like(sample_bin_matrix, dtype=torch.float)  # (batch, num_sample, num_bins)
    if order == 0:
        weight = weight + (sample_bin_matrix < 0.5) + (sample_bin_matrix == 0.5) * 0.5
    elif order == 3:
        weight = (
            weight + (4 - 6 * sample_bin_matrix**2 + 3 * sample_bin_matrix**3) * (sample_bin_matrix < 1) / 6
        )
        weight = weight + (2 - sample_bin_matrix) ** 3 * (sample_bin_matrix >= 1) * (sample_bin_matrix < 2) / 6
    else:
        raise ValueError(f"Do not support b-spline {order}-order parzen windowing")

    weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bins)

    return weight


def parzen_windowing_gaussian(
    img: torch.Tensor, preterm: torch.Tensor, bin_centers: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parzen windowing with gaussian kernel (adapted from DeepReg implementation)
    Note: the input is expected to range between 0 and 1
    Args:
        img: the shape should be B[NDHW].
    """
    img = torch.clamp(img, 0, 1)
    img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
    weight = torch.exp(
        -preterm.to(img) * (img - bin_centers.to(img)) ** 2
    )  # (batch, num_sample, num_bin)
    weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)

    return weight


def weights_to_joint_probability(wp: torch.Tensor, wt: torch.Tensor, smooth_nr: int, smooth_dr: int):
    """
    Computes prediction and target joint distribution from Parzen window density estimation weights.
    Args:
        wp: prediction weights (batch, num_sample, 1)
        wt: target weights (batch, num_sample, 1)
        smooth_nr: a small constant added to the numerator to avoid nan.
        smooth_dr: a small constant added to the denominator to avoid nan.
    """
    ppt = torch.bmm(wp.permute(0, 2, 1), wt.to(wp)) + smooth_nr
    ppt = ppt.div(ppt.sum() + smooth_dr)  # p(pred, target)
    pp = ppt.sum(dim=1, keepdim=True)  # p(pred)
    pt = ppt.sum(dim=2, keepdim=True)  # p(target)

    return ppt, pp, pt


class LocalNormalizedCrossCorrelationLoss(_Loss):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.

    Adapted from:
        https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    @deprecated_arg(name="ndim", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = "rectangular",
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        ndim: Optional[int] = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.

        .. deprecated:: 0.6.0
            ``ndim`` is deprecated, use ``spatial_dims``.
        """
        super().__init__(reduction=LossReduction(reduction).value)

        if ndim is not None:
            spatial_dims = ndim
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        _kernel = look_up_option(kernel_type, kernel_dict)
        self.kernel = _kernel(self.kernel_size)
        self.kernel_vol = self.get_kernel_vol()

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return torch.sum(vol)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")

        t2, p2, tp = target**2, pred**2, target * pred
        kernel, kernel_vol = self.kernel.to(pred), self.kernel_vol.to(pred)
        # sum over kernel
        t_sum = separable_filtering(target, kernels=[kernel.to(pred)] * self.ndim)
        p_sum = separable_filtering(pred, kernels=[kernel.to(pred)] * self.ndim)
        t2_sum = separable_filtering(t2, kernels=[kernel.to(pred)] * self.ndim)
        p2_sum = separable_filtering(p2, kernels=[kernel.to(pred)] * self.ndim)
        tp_sum = separable_filtering(tp, kernels=[kernel.to(pred)] * self.ndim)

        # average over kernel
        t_avg = t_sum / kernel_vol
        p_avg = p_sum / kernel_vol

        # normalized cross correlation between t and p
        # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
        # denoted by num / denom
        # assume we sum over N values
        # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
        #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * mean[p] = cross
        # the following is actually squared ncc
        cross = tp_sum - p_avg * t_sum
        t_var = t2_sum - t_avg * t_sum  # std[t] ** 2
        p_var = p2_sum - p_avg * p_sum  # std[p] ** 2
        t_var = torch.max(t_var, torch.zeros_like(t_var))
        p_var = torch.max(p_var, torch.zeros_like(p_var))
        ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)
        # shape = (batch, 1, D, H, W)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(ncc).neg()  # sum over the batch, channel and spatial ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ncc).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class GlobalMutualInformationLoss(_Loss):
    """
    Differentiable global mutual information loss via Parzen windowing method.

    Reference:
        https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1
    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            kernel_type: {``"gaussian"``, ``"b-spline"``}
                ``"gaussian"``: adapted from DeepReg
                Reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1.
                ``"b-spline"``: based on the method of Mattes et al [1,2] and adapted from ITK
                References:
                  [1] "Nonrigid multimodality image registration"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.
                  [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      IEEE Transactions in Medical Imaging. Vol.22, No.1,
                      January 2003. pp.120-128.

            num_bins: number of bins for intensity
            sigma_ratio: a hyper param for gaussian function
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")        
        self.kernel_type = look_up_option(kernel_type, ["gaussian", "b-spline"])
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        self.preterm, self.bin_centers = None, None
        if self.kernel_type == "gaussian":
            bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
            sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        # Parzen window density estimation
        wp, wt = parzen_windowing(
            pred, target, 
            kernel_type=self.kernel_type, num_bins=self.num_bins,
            preterm=self.preterm, bin_centers=self.bin_centers,
        )  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        
        # Compute probabilities: p(pred, target), p(pred), p(target)
        ppt, pp, pt = weights_to_joint_probability(
            wp, wt, smooth_nr=self.smooth_nr, smooth_dr=self.smooth_dr
        )

        # Compute mutual information
        mi = torch.sum(ppt * torch.log2(ppt / pt.bmm(pp)), dim=(1, 2))

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(mi).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return mi.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(mi).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class GlobalNormalisedMutualInformationLoss(_Loss):
    """
    Differentiable normalised global mutual information loss via Parzen windowing method.

    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            kernel_type: {``"gaussian"``, ``"b-spline"``}
                ``"gaussian"``: adapted from DeepReg
                Reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1.
                ``"b-spline"``: based on the method of Mattes et al [1,2] and adapted from ITK
                References:                    
                  [1] "Nonrigid multimodality image registration"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.
                  [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      IEEE Transactions in Medical Imaging. Vol.22, No.1,
                      January 2003. pp.120-128.
                  [3] "An overlap invariant entropy measure of 3D medical image alignment"
                      C. Studholme, D.L.G. Hill, D.J. Hawkes
                      Pattern Recognition. Vol.32, No.1,
                      January 1999. pp.71-86.

            num_bins: number of bins for intensity
            sigma_ratio: a hyper param for gaussian function
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        self.kernel_type = look_up_option(kernel_type, ["gaussian", "b-spline"])
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        self.preterm, self.bin_centers = None, None
        if self.kernel_type == "gaussian":
            bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
            sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        # Parzen window density estimation
        wp, wt = parzen_windowing(
            pred, target, 
            kernel_type=self.kernel_type, num_bins=self.num_bins,
            preterm=self.preterm, bin_centers=self.bin_centers,
        )  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        
        # Compute probabilities: p(pred, target), p(pred), p(target)
        ppt, pp, pt = weights_to_joint_probability(
            wp, wt, smooth_nr=self.smooth_nr, smooth_dr=self.smooth_dr
        )

        # Compute normalised mutual information
        nmi = (torch.sum(pp*pp.log2()) + torch.sum(pt*pt.log2()))/torch.sum(ppt*ppt.log2())

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(nmi).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return nmi.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(nmi).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class GlobalNormalizedCrossCorrelationLoss(_Loss):
    """
    Differentiable global normalised cross correlation loss via Parzen windowing method.

    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            kernel_type: {``"gaussian"``, ``"b-spline"``}
                ``"gaussian"``: adapted from DeepReg
                Reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1.
                ``"b-spline"``: based on the method of Mattes et al [1,2] and adapted from ITK
                References:
                  [1] "Nonrigid multimodality image registration"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.
                  [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      IEEE Transactions in Medical Imaging. Vol.22, No.1,
                      January 2003. pp.120-128.

            num_bins: number of bins for intensity
            sigma_ratio: a hyper param for gaussian function
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        self.kernel_type = look_up_option(kernel_type, ["gaussian", "b-spline"])
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        self.preterm, self.bin_centers = None, None
        if self.kernel_type == "gaussian":
            bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
            sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        # Parzen window density estimation
        wp, wt = parzen_windowing(
            pred, target, 
            kernel_type=self.kernel_type, num_bins=self.num_bins,
            preterm=self.preterm, bin_centers=self.bin_centers,
        )  # (batch, num_sample, num_bin), (batch, 1, num_bin)

        # Compute probabilities: p(pred, target), p(pred), p(target)
        ppt, pp, pt = weights_to_joint_probability(
            wp, wt, smooth_nr=self.smooth_nr, smooth_dr=self.smooth_dr
        )

        # Compute global normalised cross correlation
        # References
        # https://en.wikipedia.org/wiki/Cross-correlation#Normalization
        # https://github.com/spm/spm12/blob/3085dac00ac804adb190a7e82c6ef11866c8af02/spm_coreg.m#L209
        i = torch.arange(1, ppt.shape[1] + 1).to(pp)
        j = torch.arange(1, ppt.shape[2] + 1).to(pp)
        m1 = torch.sum(pt*i[None, :, None])
        m2 = torch.sum(pp*j[None, None, ...])
        i2 = (i - m1)**2; i2 = i2[None, ..., None]
        j2 = (j - m2)**2; j2 = j2[None, None, ...]
        sig1 = torch.sqrt(torch.sum(pt*i2, dim=(1, 2)))
        sig2 = torch.sqrt(torch.sum(pp*j2, dim=(1, 2)))
        i, j = torch.meshgrid(i - m1, j - m2, indexing="ij")
        ncc = torch.sum(torch.sum(ppt*i[None]*j[None]))/(sig1*sig2)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(ncc).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ncc).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
