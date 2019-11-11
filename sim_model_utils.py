import sys
sys.path.append('..')
# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from datagenerator import generate_dataset
from train_edge_predictor import DataGenerator, SimilarityNetwork
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

DROPOUT = 0.3
RESHAPE = (224, 224)
BATCH_SIZE = 32
TOP_HIDDEN = 4
NUM_EMBEDDING= 512
LEARNING_RATE=0.0001
data_prefix = ''

import matplotlib.pyplot as plt
data_prefix = '/home/etienne/data/'


def find_similars(sim):
    labels = pd.read_csv(data_prefix + 'sim/exploration_data/label.txt', delimiter=' ')
    labels['prefix'] = labels.apply(lambda frame: int(str(frame['frame'])[:11]), 1)
    labels['date'] = labels.apply(
        lambda frame: datetime.utcfromtimestamp(int(str(frame['frame'])[:10])).strftime("%H %d %m %Y"), 1)
    groups = labels.date.unique()
    for group in groups[1:2]:
        frames = labels[labels.date == group].loc[:, ['frame', 'prefix']]

    target = load_image(frames.sample(1).frame.values[0])
    frames.loc[:, 'sim'] = frames.apply(lambda row: compare(target, row.frame, sim), axis=1)
    return target, frames


def compare(target, frame, sim):
    input = np.concatenate([target, load_image(frame)], axis=-1).reshape((1,) + (224, 224, 6))

    return sim.model.predict(input)[0][1]


def load_image(image):
    image = data_prefix + '/rgb_0/' + str(image) + '.jpg'
    image = np.array(Image.open(image).resize(RESHAPE))
    return image


def predictions_target(sim, target='1563521837298720204'):
    labels = pd.read_csv(data_prefix + 'sim/exploration_data/poses.txt', delimiter=' ')
    labels = labels.sample(5*128)
    labels['prefix'] = labels.apply(lambda frame: str(frame['frame'])[:11], 1)
    labels['date'] = labels.apply(
        lambda frame: datetime.utcfromtimestamp(int(str(frame['frame'])[:10])).strftime("%H %d %m %Y"), 1)
    labels.loc[:, 'label'] = 0
    if not target:
        target = labels.sample(1).iloc[0, 0]
        print(target)
    labels.loc[:, 'target'] = target
    labels.loc[:, 'couple'] = labels.apply(lambda row: (str(row.frame), str(row.target)), 1)
    gen = DataGenerator(labels.loc[:, ['couple', 'label']], batch_size=128, shuffle=False)
    test = sim.model.predict_generator(gen, steps=labels.shape[0]/128,
                                       verbose=1)
    ones = [pred[1] for pred in test]
    labels.loc[:,'preds'] = ones
    plt.hist(ones)
    return labels


def show_sample(dataset, label_to_show):
    seed = np.random.randint(dataset.shape[0])
    while dataset.iloc[seed, 1] != label_to_show:
        seed = np.random.randint(dataset.shape[0])
    # print(str(dataset.iloc[seed, 0][0])[:11], str(dataset.iloc[seed, 0][1])[:11])
    show_pair(dataset.iloc[seed, 0])


def show_pair(frames):
    print(frames)
    image1 = data_prefix + '/rgb_0/' + str(frames[0]) + '.jpg'
    image2 = data_prefix + '/rgb_0/' + str(frames[1]) + '.jpg'
    image1 = Image.open(image1)
    image1.show()
    image2 = Image.open(image2)
    image2.show()

def get_scores():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot

    input_shape = RESHAPE + (6,)
    val_dataset = generate_dataset(data_prefix='/data/mix/data', to_generate=(0, 1))
    test_dataset = generate_dataset(data_prefix='/data/COM1L3/data', to_generate=(0, 1))
    results = pd.DataFrame([], columns=['type', 'feat', 'dataset', 'accuracy', 'loss', 'auc'])

    print('loaded')
    rocs = []
    for type in ['siamese', 'contrastive']:
        for feat in ['resnet', 'vgg']:
            if type=='siamese' and feat=='resnet':
                path = '/data/siamese/resnet/model.000050.h5'
            elif type=='siamese' and feat=='vgg':
                path = '/data/siamese/vgg/model.000084.h5'
            if type=='contrastive' and feat=='resnet':
                path = '/data/contrative/resnet/model.000059.h5'
            elif type=='contrastive' and feat=='vgg':
                path = '/data/contrastive/vgg/model.000001.h5'

            sim = SimilarityNetwork(input_shape, type=type, feat=feat, load=True,
                                    path=path)

            val_scores = sim.model.evaluate_generator(DataGenerator(val_dataset, n_classes=2-int(type == 'contrastive')), verbose=1)
            print(val_scores)

            print(val_scores)
            val_preds = sim.model.predict_generator(DataGenerator(val_dataset, n_classes=2-int(type == 'contrastive'), shuffle=False), verbose=1)
            test_preds = sim.model.predict_generator(DataGenerator(test_dataset, n_classes=2-int(type == 'contrastive'), shuffle=False), verbose=1)
            if type!='contrastive':
                val_preds = val_preds[:, 1]
                test_preds = test_preds[:, 1]
            val_labels = list(val_dataset.label)
            test_labels = list(test_dataset.label)
            val_auc = roc_auc_score(val_labels, val_preds)
            test_auc = roc_auc_score(test_labels, test_preds)
            val_fpr, val_tpr, _ = roc_curve(val_labels, val_preds)
            test_fpr, test_tpr, _ = roc_curve(test_labels, test_preds)

            pyplot.plot(val_fpr, val_tpr, linestyle='--', label='Validation')
            pyplot.plot(test_fpr, test_tpr, marker='.', label='test')
            # axis labels
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            # show the legend
            pyplot.legend()
            # show the plot
            pyplot.show()
            pyplot.savefig('ROC curve %s %s'%(type, feat))

            results = results.append(pd.DataFrame([[type, feat, 'val', val_scores[0], val_scores[1]]],
                                                  columns=['type', 'feat', 'dataset', 'accuracy', 'loss', val_auc]))
            test_scores = sim.model.evaluate_generator(
                DataGenerator(test_dataset, n_classes=2 - int(type == 'contrastive')), verbose=1)
            results = results.append(pd.DataFrame([[type, feat, 'test', test_scores[0], test_scores[1]]],
                                                  columns=['type', 'feat', 'dataset', 'accuracy', 'loss', test_auc]))
            results.to_csv('test_results')


if __name__ == '__main__':
    input_shape = RESHAPE + (6,)
    # val_dataset = generate_dataset(data_prefix='/home/etienne/data_correct_intention/mix/data', to_generate=(0, 1))
    # test_dataset = generate_dataset(data_prefix='/home/etienne/data_correct_intention/COM1L3/data', to_generate=(0, 1))
    #
    sim = SimilarityNetwork(input_shape, type='contrastive', load=True, path='/home/etienne/data/similarities/contrastive/resnet/model.000059.h5')
    # print('loaded')
    # # sim.load_model('/home/etienne/data/model.000010.h5')
    # print('val contrastive resnet',sim.model.evaluate_generator(DataGenerator(val_dataset, n_classes=1)))
    # print('val contrastive resnet', sim.model.evaluate_generator(DataGenerator(test_dataset, n_classes=1)))
    # sim = SimilarityNetwork(input_shape, model='siamese', load=True, path='/home/etienne/data_correct_intention/similarities/siamese/resnet/model.000050.h5')
    # print('val siamese resnet', sim.model.evaluate_generator(DataGenerator(val_dataset)))
    # print('val siamese resnet', sim.model.evaluate_generator(DataGenerator(test_dataset)))
    # sim = SimilarityNetwork(input_shape, model='siamese', load=True,
    #                         path='/home/etienne/data_correct_intention/similarities/siamese/vgg/model.000084.h5')
    # print('val contrastive vgg', sim.model.evaluate_generator(DataGenerator(val_dataset)))
    # print('val contrastive vgg', sim.model.evaluate_generator(DataGenerator(test_dataset)))
    # sim = SimilarityNetwork(input_shape, model='contrastive', load=True,
    #                         path='/home/etienne/data_correct_intention/similarities/contrastive/vgg/model.000001.h5')
    # print('val contrastive vgg', sim.model.evaluate_generator(DataGenerator(val_dataset, n_classes=1)))
    # print('val contrastive vgg', sim.model.evaluate_generator(DataGenerator(test_dataset, n_classes=1)))