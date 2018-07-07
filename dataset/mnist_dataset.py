import gzip
import numpy as np


def load_mnist_data(image_path, label_path, is_gzip=True, is_norm=True):
    open_ = gzip.open if is_gzip else open
    with open_(image_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0
    with open_(label_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    labels = np.identity(10, dtype=np.int32)[labels]
    return images, labels


class MnistDataset:
    def __init__(self, images, labels):
        if images.shape[0] != labels.shape[0]:
            Exception('データの組み合わせが不正です')
        self._images = images
        self._labels = labels
        self._data_size = images.shape[0]
        self._current_index = 0
        self._perm = np.random.permutation(self._data_size)

    def next_batch(self, batch_size, crop=None):
        if not self.has_next():
            self.reset()
        images = self._images[self._perm[self._current_index:self._current_index+batch_size]]
        labels = self._labels[self._perm[self._current_index:self._current_index+batch_size]]
        self._current_index += batch_size
        if crop is not None:
            _, h, w = images.shape
            x = np.random.randint(0, w - crop)
            y = np.random.randint(0, h - crop)
            pad_x = np.random.randint(0, w - crop)
            pad_y = np.random.randint(0, h - crop)
            images = np.pad(
                images[:, y:y+crop, x:x+crop],
                ((0, 0), (pad_y, (h - crop) - pad_y), (pad_x, (w - crop) - pad_x)),
                'constant'
            )
        return images, labels

    def reset(self):
        self._current_index = 0
        self._perm = np.random.permutation(self._data_size)

    def has_next(self):
        return self._current_index < self._data_size

    def __len__(self):
        return self._data_size


class MnistDatasetManager:
    def __init__(self):
        train_data_path = ('./mnist_data/train-images-idx3-ubyte.gz', './mnist_data/train-labels-idx1-ubyte.gz')
        test_data_path = ('./mnist_data/t10k-images-idx3-ubyte.gz', './mnist_data/t10k-labels-idx1-ubyte.gz')

        train_data = load_mnist_data(*train_data_path)
        test_data = load_mnist_data(*test_data_path)

        self.train = MnistDataset(train_data[0], train_data[1])
        self.test = MnistDataset(test_data[0], test_data[1])

    def reset_all(self):
        self.train.reset()
        self.test.reset()
