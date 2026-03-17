import torch
import numpy as np
import colour

from colour.recovery import RGB_to_sd_Smits1999
from colour.models import cctf_decoding, cctf_encoding, XYZ_to_sRGB

from comfy import model_management

def get_filter_image(image, transmission, filter_factor, auto_exposure, auto_factor, illuminant_name):

    device = model_management.get_torch_device()

    # ComfyUI tensors are (B, H, W, C) in RGB format, float32, range [0.0, 1.0]
    batch_size = image.shape[0]
    output_tensors = []
    
    wl_min = 380
    wl_max = 720
    wl_step = 2

    # Initialize colour science objects once outside the batch loop
    shape = colour.SpectralShape(wl_min, wl_max, wl_step)
    sd_filter = colour.SpectralDistribution(transmission, shape, name="Wratten Filter")
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name].copy().align(shape)

    for i in range(batch_size):
        # Convert to NumPy (on CPU). It is already RGB and [0.0, 1.0]
        img = image[i].detach().cpu().numpy()
        
        # Handle Grayscale
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        
        has_alpha = img.ndim == 3 and img.shape[2] == 4

        if has_alpha:
            alpha = img[:, :, 3:4]
            rgb = img[:, :, :3]
        else:
            alpha = None
            rgb = img
        
        # Ensure we are strictly bounded
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # --- Process only unique RGB triplets for speed ---
        # Scale to uint8 solely for the purpose of deduplication mapping
        flat_rgb8 = (rgb * 255.0 + 0.5).astype(np.uint8).reshape(-1, 3)
        unique_rgb8, inverse = np.unique(flat_rgb8, axis=0, return_inverse=True)

        mapped = np.zeros((unique_rgb8.shape[0], 3), dtype=np.float32)

        for idx, rgb8_value in enumerate(unique_rgb8):
            # Scale back to float [0.0, 1.0] for colour science calculations
            rgb_srgb = rgb8_value.astype(np.float32) / 255.0

            # sRGB -> linear RGB
            rgb_lin = cctf_decoding(rgb_srgb, function="sRGB")

            # Recover approximate spectrum from RGB
            sd = RGB_to_sd_Smits1999(rgb_lin).align(shape)

            # Apply filter transmission
            sd_filtered = colour.SpectralDistribution(
                sd.values * sd_filter.values,
                shape,
                name="Filtered Pixel",
            )

            # Spectrum -> XYZ -> linear sRGB
            XYZ = colour.sd_to_XYZ(
                sd_filtered,
                cmfs=cmfs,
                illuminant=illuminant,
                method="Integration",
            )

            rgb_lin_out = XYZ_to_sRGB(XYZ / 100.0, apply_cctf_encoding=False)

            # Apply exposure compensation in linear space
            if not auto_exposure:
                rgb_lin_out = rgb_lin_out * filter_factor
            else:
                rgb_lin_out = rgb_lin_out * auto_factor

            # Linear sRGB -> display sRGB
            rgb_srgb_out = cctf_encoding(np.clip(rgb_lin_out, 0.0, 1.0), function="sRGB")
            mapped[idx] = np.clip(rgb_srgb_out, 0.0, 1.0)

        # Reconstruct the image from the mapped unique colors
        out_rgb = mapped[inverse].reshape(rgb.shape)

        # --- Write output ---
        if alpha is not None:
            out_img = np.concatenate([out_rgb, alpha], axis=2)
        else:
            out_img = out_rgb

        # out_img is already float32 [0.0, 1.0], ready for ComfyUI
        result_tensor = torch.from_numpy(out_img).float()
        output_tensors.append(result_tensor)

    return (torch.stack(output_tensors),)