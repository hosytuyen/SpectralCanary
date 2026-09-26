import numpy as np
from PIL import Image


def create_fft_hybrid(original: Image.Image, canary: Image.Image, cutoff: float, alpha: float) -> Image.Image:
    canary = canary.resize(original.size, Image.Resampling.BICUBIC)
    image = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    donor = np.asarray(canary.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = image.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - height // 2) / height) ** 2 + ((xx - width // 2) / width) ** 2)
    mask = np.exp(-(radius**2) / (2.0 * cutoff**2))[..., None]
    image_fft = np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))
    donor_fft = np.fft.fftshift(np.fft.fft2(donor, axes=(0, 1)), axes=(0, 1))
    hybrid_fft = image_fft + alpha * mask * (donor_fft - image_fft)
    hybrid = np.fft.ifft2(np.fft.ifftshift(hybrid_fft, axes=(0, 1)), axes=(0, 1)).real
    return Image.fromarray(np.clip(hybrid * 255.0, 0, 255).astype(np.uint8))
