import numpy as np
import pandas as pd
from datetime import datetime
from keras.applications.resnet50 import preprocess_input
import skimage as sk
import skimage.transform as transform
import keras
from PIL import Image
import numpy.random as random
data_prefix = '/home/etienne/data/sim/data'
NUM_SAMPLES = 5
RESHAPE = (224, 224)


def canonical(image_array):
    return image_array


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array):
    return image_array[:, ::-1]


available_transformations = {
    'canonical': canonical,
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}


def generate_dataset(data_prefix=data_prefix, to_generate=(0, 6)):
    labels = pd.read_csv(data_prefix+'/label.txt', delimiter=' ')
    labels['prefix'] = labels.apply(lambda frame: int(str(frame['frame'])[:11]), 1)
    labels['date'] = labels.apply(lambda frame : datetime.utcfromtimestamp(int(str(frame['frame'])[:10])).strftime("%H %d %m %Y"), 1)
    groups = labels.date.unique()
    dataset = []
    for group in groups[to_generate[0]:to_generate[1]]:
        frames = labels[labels.date == group].loc[:, ['frame', 'prefix']]
        frames.apply(lambda row: sample(row.prefix, row.frame, 1, frames,  dataset), 1)
        frames.apply(lambda row: sample(row.prefix, row.frame, 0, frames, dataset), 1)
    print(len(dataset))
    return pd.DataFrame(dataset, columns=['frame', 'label']).sample(frac=1).reset_index(drop=True)


def sample(prefix, frame, label, frames, dataset, num_samples=NUM_SAMPLES):
    if label == 1:
        samples = frames[abs(frames.prefix - float(prefix))<20]
        samples = samples[1 < abs(frames.prefix - float(prefix))]
        samples = samples.sample(min(num_samples, samples.shape[0])).iloc[:, 0].values
    else:
        samples = frames[abs(frames.prefix - float(prefix)) > 40]
        samples1 = samples[abs(samples.prefix - float(prefix)) < 80]
        sample1 = samples1.sample(min(num_samples, samples1.shape[0])).iloc[:, 0].values
        samples2 = samples[abs(samples.prefix - float(prefix)) > 80]
        samples2 = samples2[abs(samples2.prefix - float(prefix)) < 150]
        sample2 = samples2.sample(min(num_samples, samples2.shape[0])).iloc[:, 0].values
        samples = list(sample1)+list(sample2)
    for sample in samples:
        if sample != frame:
            # no ordering of the frames for the dataset to be random
            dataset.append([(frame,  sample), label])


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, dataset,  ave=None, std=None, batch_size=32, dim=RESHAPE,
                 n_channels=3,
                 n_classes=2, shuffle=True, show_images=False, resample=False, augment=False, model='resnet'):
        """Initialization.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.dim = dim
        if ave is None:
            self.ave = np.zeros(n_channels)
        else:
            self.ave = ave
        if std is None:
            self.std = np.zeros(n_channels) + 1
        else:
            self.std = std
        self.show_images = show_images
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.resample = resample
        self.on_epoch_end()
        self.augment = augment
        self.model = model

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(self.dataset.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        idx_min = index * self.batch_size
        idx_max = min((index + 1) * self.batch_size, len(self.indexes-1))
        indexes = self.indexes[idx_min:idx_max]

        # Find list of IDs
        img_files_temp = self.dataset.iloc[indexes]
        # if only negative examples in the batch, add a positive
        # if self.augment:
        #     if img_files_temp.label.sum() == 0:
        #         img_files_temp = img_files_temp.append(self.dataset.iloc[self.dataset[self.dataset.label==1].sample(1).index.values[0]])
        #         img_files_temp = img_files_temp.iloc[1:]
        # Generate data
        X, y = self.__data_generation(img_files_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(self.dataset.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)
        if self.resample:
            i = np.random.randint(6)
            self.dataset = generate_dataset(to_generate=(i, i+1))

    def __data_generation(self, batch):
        """Generates data containing batch_size samples."""
        X_img = []
        y = []
        # Generate data
        for frames, label in batch.values:
            # Read image
            x_1 = np.array(Image.open(data_prefix + '/rgb_0/' + str(frames[0]) + '.jpg').resize(RESHAPE))
            x_2 = np.array(Image.open(data_prefix + '/rgb_0/' + str(frames[1]) + '.jpg').resize(RESHAPE))

            # Normalization

            # preprocess in the image_net preprocess (linear combination so doesn't matter which one as long as its consistent with the training
            x_1 = preprocess_input(x_1)
            x_2 = preprocess_input(x_2)
            # Some image augmentation codes
            if self.augment:
                key = random.choice(list(available_transformations))
                if np.random.randint(2):
                    x_1 = available_transformations[key](x_1)
                else:
                    x_2 = available_transformations[key](x_2)

            X_img.append(np.concatenate([x_1, x_2], axis=-1))
            y.append(label)
        if self.n_classes > 1:
            output = keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            output = y
        return np.array(X_img), output
