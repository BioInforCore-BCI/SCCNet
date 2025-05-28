import os
import numpy as np
import argparse
from PIL import Image
from utils import load_normaliser


def normalize_patches(patch_dir, save_dir, colour_norm=None):
    """
    Apply color normalization to image patches in a specified directory.

    Parameters:
    - patch_dir: Directory containing the patches.
    - save_dir: Directory to save normalized patches.
    - colour_norm: Object for color normalization.
    """
    os.makedirs(save_dir, exist_ok=True)

    # List all image files in the patch directory
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f'Total patches to process: {len(patch_files)}')

    for i, patch_file in enumerate(patch_files):
        try:
            # Load the patch as an RGB image
            patch_path = os.path.join(patch_dir, patch_file)
            patch_PIL = Image.open(patch_path).convert("RGB")

            # Apply color normalization if a standard is provided
            if colour_norm is not None:
                try:
                    patch_PIL = Image.fromarray(colour_norm.transform(np.array(patch_PIL)))
                except Exception as e:
                    print(f"Error in color normalization for {patch_file}: {e}")
                    continue

            # Save the normalized patch
            normalized_path = os.path.join(save_dir, patch_file)
            patch_PIL.save(normalized_path)
        except Exception as e:
            print(f"Error processing patch {patch_file}: {e}")

        # Print progress every 1000 patches
        if (i + 1) % 10000 == 0:
            print(f"Currently processing patch {i + 1}/{len(patch_files)}")

    print('All patches normalization done!')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply color normalization to image patches.')
    parser.add_argument('--patch_dir', type=str, required=True, help='Directory containing image patches.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save normalized patches.')
    parser.add_argument('--colour_standard', type=str, default=None,
                        help='Path to standard image for color normalization.')

    args = parser.parse_args()

    # Load color normalization model if provided
    colour_norm = None
    if args.colour_standard:
        try:
            colour_norm = load_normaliser(args.colour_standard)
        except Exception as e:
            print(f"Error reading standard image for color normalization: {e}")
            exit()

    normalize_patches(args.patch_dir, args.save_dir, colour_norm=colour_norm)