import numpy as np
import os
from deepforest import CascadeForestClassifier
import time
from datetime import datetime
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    ctime = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    data_root = 'test_feature_path'
    tic = time.time()
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
    model = CascadeForestClassifier(random_state=42)
    model.load('save_model_path')
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    toc = time.time()
    time = toc - tic
    print("\ntime: {:.3f} s".format(time))



