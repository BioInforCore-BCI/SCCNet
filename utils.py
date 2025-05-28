import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import openslide
import pickle
import yaml
from scipy.optimize import nnls
from scipy.stats import rankdata


# Image Utility Functions
# adapted from https://github.com/KatherLab/preProcessing/


def read_image(path):
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im

def show_colors(C):
    n = C.shape[0]
    for i in range(n):
        color = C[i] / 255 if C[i].max() > 1.0 else C[i]
        plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=color, linewidth=20)
    plt.axis('off')
    plt.axis([0, 1, -1, n])
    plt.show()

def show(image, now=True, fig_size=(10, 10)):
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size is not None:
        plt.rcParams['figure.figsize'] = fig_size
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now:
        plt.show()

def build_stack(tup):
    N = len(tup)
    shape = tup[0].shape
    stack = np.zeros((N, *shape))
    for i in range(N):
        stack[i] = tup[i]
    return stack

def patch_grid(ims, width=5, sub_sample=None, rand=False, save_name=None):
    N0 = ims.shape[0]
    if sub_sample is None:
        N = N0
        stack = ims
    elif not rand:
        N = sub_sample
        stack = ims[:N]
    else:
        N = sub_sample
        idx = np.random.choice(range(N0), sub_sample, replace=False)
        stack = ims[idx]
    height = int(np.ceil(float(N) / width))
    plt.rcParams['figure.figsize'] = (18, (18 / width) * height)
    plt.figure()
    for i in range(N):
        plt.subplot(height, width, i + 1)
        im = stack[i]
        show(im, now=False, fig_size=None)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

def standardize_brightness(I):
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

def remove_zeros(I):
    mask = (I == 0)
    I[mask] = 1
    return I

def RGB_to_OD(I):
    I = remove_zeros(I)
    return -1 * np.log(I / 255)

def OD_to_RGB(OD):
    return (255 * np.exp(-1 * OD)).astype(np.uint8)

def normalize_rows(A):
    return A / np.linalg.norm(A, axis=1)[:, None]

def notwhite_mask(I, thresh=0.8):
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return L < thresh

def sign(x):
    return (x > 0) - (x < 0)

def get_concentrations(I, stain_matrix):
    # Convert image to OD space
    OD = RGB_to_OD(I).reshape((-1, 3))  # shape: (num_pixels, 3)
    n_pixels = OD.shape[0]
    n_stains = stain_matrix.shape[0]

    # Initialize concentrations array
    concentrations = np.zeros((n_pixels, n_stains))

    # Transpose stain_matrix to shape (3, n_stains)
    X = stain_matrix.T

    # Solve NNLS for each pixel
    for i in range(n_pixels):
        y = OD[i, :]
        coeffs, _ = nnls(X, y)
        concentrations[i, :] = coeffs

    return concentrations


# Stain Normalisation Logic


def get_stain_matrix(I, beta=0.15, alpha=1):
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return normalize_rows(HE)

class Normalizer(object):
    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(np.uint8)

    def hematoxylin(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H


# Loader Function


def load_normaliser(sampleImagePath):
    normalizer = Normalizer()
    target = read_image(sampleImagePath)
    normalizer.fit(target)
    return normalizer



# WSI Class


class WholeSlideImage(object):

    def __init__(self, path):

        """
        Purpose: Initializes the WholeSlideImage object by loading a WSI using the OpenSlide library.
        Parameters: path (string) representing the file path to the WSI.
        Operations: Stores basic properties like name, image dimensions, downsample levels, and initializes placeholders for tumor and tissue contours.
        """
        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions

    def visWSI(self, vis_level=0, max_size=None, custom_downsample=1):
        """
        Purpose: Visualizes the WSI with options to overlay contours and annotations.
        Parameters: Various visualization parameters such as visualization level, colors for different annotations, and options for downsampling and resizing.
        Returns: An image object after applying the visualization parameters.
        """

        top_left = (0, 0)
        region_size = self.level_dim[vis_level]

        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

        img = Image.fromarray(img)

        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        return img


    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left  # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                # print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                    1 - alpha, 0, canvas)
        return img


# file loading functions

def save_pkl(filename, save_object):
    writer = open(filename, 'wb')
    pickle.dump(save_object, writer)
    writer.close()


def load_pkl(filename):
    loader = open(filename, 'rb')
    file = pickle.load(loader)
    loader.close()
    return file


def load_config_yaml(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

        for key, value in config_dict.items():
            if isinstance(value, dict):
                print('\n' + key)
                for value_key, value_value in value.items():
                    print(value_key + " : " + str(value_value) + " " + str(type(value_value)))
            else:
                print('\n' + key + " : " + str(value))

        print("configuration loaded")

    return config_dict


def to_percentiles(scores):
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores

