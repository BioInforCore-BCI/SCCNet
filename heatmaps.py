import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from utils import WholeSlideImage, to_percentiles



"""
This script generates heatmaps according to score.csv files for each WSI.
Each score.csv file contains prediction scores for image patches of each WSI.
"""


def create_single_heatmap(wsi_path, score_path, heatmap_save_name, heatmap_config):
    """
    Create a single heatmap for a WSI.

    Parameters:
    - wsi_path (str): Path to the whole slide image.
    - score_path (str): Path to the score file.
    - heatmap_save_name (str): Name to save the heatmap as.
    - heatmap_config (dict): Configuration for the heatmap.

    Returns:
    - None
    """
    try:
        # Load scores and coordinates from the CSV file
        result_df = pd.read_csv(score_path)
        scores = result_df['Score']
        coords = result_df[['coord_x', 'coord_y']].values

        # Extract heatmap configuration
        patch_size = tuple([heatmap_config.get('patch_size', 512)] * 2)
        save_path = heatmap_config.get('heatmap_save_dir', './heatmaps')
        os.makedirs(save_path, exist_ok=True)
        save_ext = heatmap_config.get('save_ext', 'png')
        vis_level = heatmap_config.get('vis_level', 1)
        heatmap_mode = heatmap_config.get('heatmap_mode', 'percentiles').lower()
        high_thresh = heatmap_config.get('thresh_high', 0.7)
        low_thresh = heatmap_config.get('thresh_low', 0.3)
        threshold = heatmap_config.get('threshold', 0.6)
        blank_canvas = heatmap_config.get('blank_canvas', False)
        blur = heatmap_config.get('blur', False)
        alpha = heatmap_config.get('alpha', 0.6)
        overlap = heatmap_config.get('overlap', 0)
        custom_downsample = heatmap_config.get('custom_downsample', 1)
        max_size = heatmap_config.get('max_size', None)
        cmap = plt.get_cmap(heatmap_config.get('cmap', 'viridis'))

        # Set standard threshold for binarization
        if heatmap_mode == 'binarise':
            if threshold < 0:
                threshold = 1.0 / len(scores)
        else:
            threshold = 0
        # Load the WSI
        wsi = WholeSlideImage(wsi_path)
        if vis_level < 0:
            vis_level = wsi.wsi.get_best_level_for_downsample(32)
        downsample = wsi.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]

        region_size = wsi.level_dim[vis_level]
        top_left = (0, 0)
        bot_right = wsi.level_dim[0]
        patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        print(f'\ncreating heatmap for: {heatmap_save_name}')
        print(f'top_left: {top_left}, bot_right: {bot_right}')
        print(f'w: {region_size[0]}, h: {region_size[1]}')
        print(f'scaled patch size: {patch_size}')

        if heatmap_mode == 'percentiles':
            scores = to_percentiles(scores)
            scores /= 100

        overlay = np.zeros(region_size[::-1], dtype=float)
        counter = np.zeros(region_size[::-1], dtype=np.uint16)
        high_count, low_count, threshold_count = 0, 0, 0

        for score, (x, y) in zip(scores, coords):
            effective_score = 0
            if heatmap_mode == 'extreme':
                if score <= low_thresh:
                    low_count += 1
                    effective_score = score
                elif score >= high_thresh:
                    high_count += 1
                    effective_score = score
            elif heatmap_mode == 'binarise':
                if score >= threshold:
                    threshold_count += 1
                    effective_score = 1.0
            else:
                effective_score = score

            overlay[y:y + patch_size[1], x:x + patch_size[0]] += effective_score
            counter[y:y + patch_size[1], x:x + patch_size[0]] += 1

        if heatmap_mode == 'extreme':
            print(f'Extreme mode: high_count={high_count}, low_count={low_count}')
        elif heatmap_mode == 'binarise':
            print(f'Binarize mode: threshold_count={threshold_count}')
        else:
            print('Percentile mode')

        zero_mask = counter == 0
        if heatmap_mode == 'binarise':
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter

        if np.isnan(overlay).any():
            print(f"Warning: NaN values found in overlay for {heatmap_save_name}. Replacing NaNs with zeros.")
            overlay = np.nan_to_num(overlay)

        if blur:
            overlay = cv2.GaussianBlur(overlay, (patch_size[0] * 2 + 1, patch_size[1] * 2 + 1), 0)

        if not blank_canvas:
            img = np.array(wsi.wsi.read_region((0, 0), vis_level, region_size).convert("RGB"))
        else:
            img = np.full((region_size[1], region_size[0], 3), fill_value=255, dtype=np.uint8)

        print(f'\ncomputing heatmap image: {heatmap_save_name}')
        twenty_percent_chunk = max(1, len(coords) // 5)

        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print(f'progress: {idx + 1}/{len(coords)}')
            score = scores[idx]
            coord = coords[idx]
            if heatmap_mode == 'extreme':
                out_condition = not low_thresh <= score <= high_thresh
            elif heatmap_mode == 'binarise':
                out_condition = score >= threshold
            else:
                out_condition = score > threshold

            if out_condition:
                raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
                color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)
                img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = color_block

        if blur:
            img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        if alpha < 1.0:
            img = wsi.block_blending(img, vis_level, (0, 0), wsi.level_dim[0], alpha=alpha, blank_canvas=blank_canvas,
                                     block_size=1024)

        img = Image.fromarray(img)
        if custom_downsample > 1:
            img = img.resize((img.width // custom_downsample, img.height // custom_downsample))

        if max_size is not None and (img.width > max_size or img.height > max_size):
            resize_factor = max_size / max(img.width, img.height)
            img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)))

        img.save(os.path.join(save_path, f'{heatmap_save_name}.{save_ext}'), quality=100)

        if heatmap_config.get('save_orig', False):
            original_path = os.path.join(save_path, 'original')
            os.makedirs(original_path, exist_ok=True)
            original_img = wsi.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=custom_downsample)
            original_img.save(os.path.join(original_path, f'{heatmap_save_name}_original.{save_ext}'), quality=100)

        print(f"{heatmap_save_name} heatmap done")
    except Exception as e:
        print(f"Error creating heatmap for {heatmap_save_name}: {e}")


def create_multi_heatmaps(data_args, heatmap_args):
    """
    Create multiple heatmaps for a list of WSIs.

    Parameters:
    - data_args (dict): Data-related arguments.
    - heatmap_args (dict): Heatmap-related arguments.

    Returns:
    - None
    """
    process_df = pd.read_csv(data_args['process_list'])
    print(f"Starting to create heatmaps for {len(process_df['SampleID'])} WSIs")

    os.makedirs(heatmap_args.get('heatmap_save_dir', './heatmaps'), exist_ok=True)

    for wsi_name in process_df["SampleID"]:
        try:
            if "wsi_path" in process_df.columns:
                wsi_path = os.path.join(data_args['wsi_dir'],
                                        process_df.loc[process_df["SampleID"] == wsi_name, "wsi_path"].values[0])
            else:
                wsi_path = os.path.join(data_args['wsi_dir'], f'{wsi_name}.{data_args.get("wsi_format", "svs")}')

            score_path = os.path.join(data_args['score_dir'], f'{wsi_name}_score_file.csv')
            if os.path.isfile(wsi_path) and os.path.isfile(score_path):
                create_single_heatmap(wsi_path=wsi_path, score_path=score_path, heatmap_save_name=wsi_name,
                                      heatmap_config=heatmap_args)
            else:
                print(f'{wsi_path} or/and {score_path} do(es) not exist')
        except Exception as e:
            print(f"Error processing {wsi_name}: {e}")

    print("All heatmaps done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap generation script')

    # Data arguments
    parser.add_argument('--process_list', type=str, required=True, help='Path to process_list CSV file')
    parser.add_argument('--wsi_dir', type=str, required=True, help='Directory containing WSI files')
    parser.add_argument('--score_dir', type=str, required=True, help='Directory containing score CSV files')
    parser.add_argument('--wsi_format', type=str, default='svs', help='File extension of WSI files (default: svs)')

    # Heatmap arguments
    parser.add_argument('--heatmap_save_dir', type=str, default='./heatmaps', help='Directory to save heatmaps')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for heatmap')
    parser.add_argument('--save_ext', type=str, default='png', help='File extension for saved heatmaps')
    parser.add_argument('--vis_level', type=int, default=1, help='Visualization level')
    parser.add_argument('--heatmap_mode', type=str, default='percentiles', choices=['percentiles', 'binarise', 'extreme'], help='Heatmap mode')
    parser.add_argument('--thresh_high', type=float, default=0.7, help='High threshold for extreme mode')
    parser.add_argument('--thresh_low', type=float, default=0.3, help='Low threshold for extreme mode')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for binarise mode')
    parser.add_argument('--blank_canvas', action='store_true', help='Render heatmap on blank canvas')
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian blur')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha blending value for overlay')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap ratio between patches')
    parser.add_argument('--custom_downsample', type=int, default=1, help='Custom downsampling factor')
    parser.add_argument('--max_size', type=int, default=None, help='Maximum size for saved heatmap image')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap to use for heatmap')
    parser.add_argument('--save_orig', action='store_true', help='Save original WSI image alongside heatmap')

    args = parser.parse_args()

    # Assemble dicts to match the original function signatures
    data_args = {
        'process_list': args.process_list,
        'wsi_dir': args.wsi_dir,
        'score_dir': args.score_dir,
        'wsi_format': args.wsi_format
    }

    heatmap_args = {
        'heatmap_save_dir': args.heatmap_save_dir,
        'patch_size': args.patch_size,
        'save_ext': args.save_ext,
        'vis_level': args.vis_level,
        'heatmap_mode': args.heatmap_mode,
        'thresh_high': args.thresh_high,
        'thresh_low': args.thresh_low,
        'threshold': args.threshold,
        'blank_canvas': args.blank_canvas,
        'blur': args.blur,
        'alpha': args.alpha,
        'overlap': args.overlap,
        'custom_downsample': args.custom_downsample,
        'max_size': args.max_size,
        'cmap': args.cmap,
        'save_orig': args.save_orig
    }

    create_multi_heatmaps(data_args, heatmap_args)
