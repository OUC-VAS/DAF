import numpy as np
import os
from deepforest import CascadeForestClassifier
import time
from datetime import datetime
from sklearn.metrics import accuracy_score

def scan_data_root(data_root):
    real_paths = []
    fake_paths = []
    sample_ids = []

    for label in [0, 1]:
        label_folder = os.path.join(data_root, "train", str(label))
        if not os.path.exists(label_folder):
            continue
        for name in os.listdir(label_folder):
            full_path = os.path.join(label_folder, name)
            if os.path.isfile(full_path) and name.endswith(".npy"):
                if label == 0:
                    real_paths.append(full_path)
                else:
                    fake_paths.append(full_path)
                sample_ids.append(name)

    return real_paths, fake_paths

if __name__ == '__main__':
    ctime = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    data_root = './train_feature_path'
    real_path, fake_path = scan_data_root(data_root)
    N = len(real_path) + len(fake_path)
    weights = np.ones(N, dtype=float)
    tic = time.time()
    print("{} Getting temp model...".format(ctime))
    samples_lenth = int(len(real_path) * 0.9 * 0.1)
    model_temp = CascadeForestClassifier(max_layers=3
                                         , random_state=42)
    real_samples = real_path[:samples_lenth]
    fake_samples = fake_path[:samples_lenth]
    xreal = []
    xfake = []
    for sample in real_samples:
        feature = np.load(sample)
        xreal.append(feature)
    for sample in fake_samples:
        feature = np.load(sample)
        xfake.append(feature)
    X_temp = np.vstack((xreal, xfake))
    yreal = np.zeros(len(real_samples))
    yfake = np.ones(len(fake_samples))
    y_temp = np.concatenate((yreal, yfake))
    _ = model_temp.fit(X_temp, y_temp)

    test_real_dir = os.path.join(data_root, "test/0")
    test_fake_dir = os.path.join(data_root, "test/1")
    real_features = []
    for fname in sorted(os.listdir(test_real_dir)):
        fpath = os.path.join(test_real_dir, fname)
        if fname.endswith(".npy"):
            feat = np.load(fpath)
            real_features.append(feat)
    fake_features = []
    for fname in sorted(os.listdir(test_fake_dir)):
        fpath = os.path.join(test_fake_dir, fname)
        if fname.endswith(".npy"):
            feat = np.load(fpath)
            fake_features.append(feat)

    x_test = np.vstack(real_features + fake_features)
    y_test = np.array([0] * len(real_features) + [1] * len(fake_features))
    ctime = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    print("{} Get temp model!".format(ctime))
    ctime = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    print("{} Start building the model...".format(ctime))
    model = CascadeForestClassifier(random_state=42)
    # training================================================================================================================
    model = model.fit_v(real_path, fake_path, weights, 0.1, 1.5, 3, 3, 10, model_temp)
    y_pred = model.predict(x_test)
    model.save('./save_model_path')
    acc = accuracy_score(y_test, y_pred) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    toc = time.time()
    time = toc - tic
    print("\ntime: {:.3f} s".format(time))



