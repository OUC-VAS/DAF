import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.feature import hog
import argparse

# ------------------ args ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./input_image_path', help='Data root directory, which should contain: train/0, train/1, test/0, test/1')
parser.add_argument('--out_root', type=str, default='./out_feature_path', help='Output root directory')
parser.add_argument('--img_size', type=int, default=256, help='resize to img_size x img_size')
parser.add_argument('--M', type=int, default=6, help='number of frequency filters')
parser.add_argument('--scales', type=int, nargs='+', default=[16, 8, 4, 2, 1])
parser.add_argument('--exts', type=str, nargs='+',
                    default=['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'])
parser.add_argument('--skip_exists', action='store_true', help='skip if feature file exists')
args = parser.parse_args()


def sliding_window_feature_reduce(X, window_size=10, stride=10, mode='max'):
    if X.ndim == 1:
        X = X[np.newaxis, :]
    reduced_all = []
    for row in X:
        row_reduced = []
        i = 0
        n_features = row.shape[0]
        while i < n_features:
            window = row[i:i + window_size]
            if len(window) == 0:
                break
            if mode == 'mean':
                val = np.mean(window)
            elif mode == 'max':
                val = np.max(window)
            elif mode == 'min':
                val = np.min(window)
            elif mode == 'std':
                val = np.std(window)
            else:
                raise ValueError("mode must be 'mean', 'max', 'min', or 'std'")
            row_reduced.append(val)
            i += stride
        reduced_all.append(row_reduced)
    result = np.array(reduced_all)
    return result[0] if result.shape[0] == 1 else result


def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) *
          np.cos((j + 0.5) * np.pi * i / size)
          for j in range(size)] for i in range(size)]
    return np.array(m, dtype=np.float32)


def generate_filter(start, end, size):
    return np.array([[0. if i + j > end or i + j < start else 1.
                      for j in range(size)] for i in range(size)], dtype=np.float32)


def apply_filter(x, M, filters, DCT_patch):
    if x.ndim == 3:
        x = x[:, :, 0]
    x_unfold_dct = np.matmul(np.matmul(DCT_patch, x), DCT_patch.T)
    y_list = []
    for i in range(M):
        y = np.abs(x_unfold_dct)
        y = np.log10(y + 1e-15)
        y = y * filters[i]
        y = np.sum(y, axis=0)
        y_list.append(y)
    return y_list


def _ensure_gray_0_255(patch_rgb):
    g = rgb2gray(patch_rgb).astype(np.float32)
    if g.max() <= 1.5:
        g *= 255.0
    return g


def build_patches_16(image_rgb, base_grid=16):
    img = np.asarray(image_rgb)
    H, W = img.shape[:2]
    cell_h, cell_w = H // base_grid, W // base_grid
    Hc, Wc = cell_h * base_grid, cell_w * base_grid
    img = img[:Hc, :Wc, :]
    patches = []
    for i in range(base_grid):
        y0, y1 = i * cell_h, (i + 1) * cell_h
        for j in range(base_grid):
            x0, x1 = j * cell_w, (j + 1) * cell_w
            patches.append(img[y0:y1, x0:x1, :])
    return patches


def _avg_pool_grid(feat_grid: np.ndarray, out_size: int) -> np.ndarray:
    H, W, D = feat_grid.shape
    if out_size == H and out_size == W:
        return feat_grid
    if H % out_size != 0 or W % out_size != 0:
        raise ValueError(f"{H}x{W} can't pool to {out_size}x{out_size}")
    kH, kW = H // out_size, W // out_size
    return feat_grid.reshape(out_size, kH, out_size, kW, D).mean(axis=(1, 3))


def extract_final_feature_one_image(
    image_rgb,
    item_name: str,
    out_dir: str,
    M: int,
    scales=(16, 8, 4, 2, 1),
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    reduce_window=10,
    reduce_stride=10,
    reduce_mode="max",
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{item_name}_features.npy")


    patches16 = build_patches_16(image_rgb, base_grid=16)
    if len(patches16) != 256:
        raise ValueError(f"Expected 256 patches, got {len(patches16)}")

    hog_feats = []
    for patch in patches16:
        gray = rgb2gray(patch)
        feat = hog(
            gray,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True,
        )
        feat = sliding_window_feature_reduce(feat, window_size=reduce_window, stride=reduce_stride, mode=reduce_mode)
        hog_feats.append(feat)

    hog16 = np.asarray(hog_feats, dtype=np.float32).reshape(16, 16, -1)


    ph, pw = patches16[0].shape[:2]
    if ph != pw:
        raise ValueError(f"Patch not square: {patches16[0].shape}")
    window_size = ph

    DCT_patch = DCT_mat(window_size)
    filters = [
        generate_filter(window_size * 2.0 / M * i, window_size * 2.0 / M * (i + 1), window_size)
        for i in range(M)
    ]

    freq_feats = []
    for patch in patches16:
        g = _ensure_gray_0_255(patch)
        g = np.expand_dims(g, axis=2)
        fre = apply_filter(g, M, filters, DCT_patch)
        fre = np.concatenate([mat.ravel() for mat in fre])
        fre = sliding_window_feature_reduce(fre, window_size=reduce_window, stride=reduce_stride, mode=reduce_mode)
        freq_feats.append(fre)

    freq16 = np.asarray(freq_feats, dtype=np.float32).reshape(16, 16, -1)

    # multi-scale pool + concat hog then freq + flatten per scale + concat all scales
    all_scales_flat = []
    for s in scales:
        hog_s = _avg_pool_grid(hog16, s).reshape(s * s, -1)
        freq_s = _avg_pool_grid(freq16, s).reshape(s * s, -1)
        combined = np.concatenate([hog_s, freq_s], axis=1)  # hog first
        all_scales_flat.append(combined.reshape(-1))

    final_feat = np.concatenate(all_scales_flat, axis=0).astype(np.float32)
    np.save(out_path, final_feat)
    return out_path


def list_images(folder, exts):
    exts = tuple(e.lower() for e in exts)
    if not os.path.isdir(folder):
        return []
    return sorted([fn for fn in os.listdir(folder) if fn.lower().endswith(exts)])



if __name__ == '__main__':
    splits = [('train', '0'), ('train', '1'), ('test', '0'), ('test', '1')]
    for split, cls in splits:
        in_dir = os.path.join(args.data_root, split, cls)
        out_dir = os.path.join(args.out_root, split, cls)

        if not os.path.isdir(in_dir):
            print(f"[Skip] Input directory does not exist: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        files = list_images(in_dir, args.exts)
        if len(files) == 0:
            print(f"[No images] {in_dir}")
            continue

        for fn in tqdm(files, desc=f"{split}/{cls}", unit="image"):
            item_name = os.path.splitext(fn)[0]
            out_path = os.path.join(out_dir, f"{item_name}_features.npy")
            if args.skip_exists and os.path.exists(out_path):
                continue

            img_bgr = cv2.imread(os.path.join(in_dir, fn))
            if img_bgr is None:
                print(f"[Failed to read image] {split}/{cls}/{fn}")
                continue

            img_bgr = cv2.resize(img_bgr, (args.img_size, args.img_size))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            extract_final_feature_one_image(
                image_rgb=img_rgb,
                item_name=item_name,
                out_dir=out_dir,
                M=args.M,
                scales=tuple(args.scales),
            )

    print("All completed. Output root directory: ", args.out_root