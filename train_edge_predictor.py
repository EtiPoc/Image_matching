import sys
sys.path.append('..')
# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
import numpy as np
from PIL import Image
from keras import regularizers
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.layers import (
        Input,
        Dense,
        Dropout,
        Lambda,
        concatenate,
        Activation,
        BatchNormalization,
        Flatten
        )
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from datagenerator import generate_dataset, DataGenerator
import keras
import matplotlib.pyplot as plt

DROPOUT = 0.3
RESHAPE = (224, 224)
BATCH_SIZE = 32
TOP_HIDDEN = 4
data_prefix = '/home/etienne/data/mix/data'
NUM_EMBEDDING = 512
LEARNING_RATE =0.0001


def plot_history(history):
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')


def _bn_relu_for_dense(input):
    norm = BatchNormalization(axis=1)(input)
    return Activation('relu')(norm)


def FeatModel(model='resnet', load=False):
    if model == 'vgg':
        if load:
            feat_model = VGG16(include_top=False, pooling='avg', weights=None)
        else:
            feat_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
            # feat_model = VGG16(include_top=False, pooling='avg', weights='vgg16-hybrid1365_weights.h5')
            print('loaded vgg places weights')
    else:
        if load:
            feat_model = ResNet50(include_top=False, pooling='avg', weights=None)
        else:
            feat_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')

    inp = feat_model.layers[0].input
    oup = feat_model.layers[-1].output
    return Model(inputs=inp, outputs=oup)


def FCModel(input_length, softmax=True):
    input = Input(shape=(input_length,))
    raw_result =  Activation('relu')(BatchNormalization(axis=1)(input))
    for _ in range(TOP_HIDDEN):
        raw_result = Dense(units=NUM_EMBEDDING, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(raw_result)
        raw_result = _bn_relu_for_dense(raw_result)

    if softmax:
        output = Dense(units=2, activation='softmax', kernel_initializer='he_normal')(raw_result)
    else:
        output = Dense(units=128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(raw_result)

    return Model(inputs=input, outputs=output)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def contrastive(input_shape, model='resnet', load=False):
    feat_model = FeatModel(load=load, model=model)
    input = Input(shape=input_shape)

    feat1 = feat_model(Lambda(lambda x: x[:, :, :, :3] / 255)(input))
    feat2 = feat_model(Lambda(lambda x: x[:, :, :, 3:] / 255)(input))

    fc = FCModel(int(feat1.shape[-1]), softmax=False)

    fc1 = fc(feat1)
    fc2 = fc(feat2)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([fc1, fc2])
    return Model(inputs=input, outputs=distance)


def siamese(input_shape, model='resnet', load=False):
    # model
    feat_model = FeatModel(load=load, model=model)
    input = Input(shape=input_shape)

    feat1 = feat_model(Lambda(lambda x: x[:, :, :, :3])(input))
    feat2 = feat_model(Lambda(lambda x: x[:, :, :, 3:])(input))

    feat = concatenate([feat1, feat2], axis=-1)
    feat = Dropout(DROPOUT)(feat)

    fcmodel = FCModel(int(feat.shape[-1]))
    output = fcmodel(feat)

    return Model(inputs=input, outputs=output)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 1, y_true.dtype)))


def lr_schedule(epoch):
    """
    Learning rate schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    """
    lr = 0.0001
    if epoch > 40:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 5e-2
    elif epoch > 20:
        lr *= 1e-1
    elif epoch > 10:
        lr *= 5e-1
    print ('Learning rate: ', lr)
    return lr


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class SimilarityNetwork:
    def __init__(self, input_shape, type='siamese', feat='resnet', **args):
        self.type = type
        self.feat = feat
        from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
        local_device_protos = device_lib.list_local_devices()
        self.num_gpus = sum([1 for d in local_device_protos if d.device_type == "GPU"])
        self.input_shape = input_shape
        if self.type == 'contrastive':
            loss = contrastive_loss
        else:
            loss='binary_crossentropy'

        if args.get('load', None):
            self.path = data_prefix+'/data/model.000050.h5'
            if args.get('path', None):
                self.path = args.get('path', None)
            if type == 'contrastive':
                print(type)
                self.model = contrastive(input_shape, model=self.feat, load=True)
            else:
                self.model = siamese(input_shape, model=self.feat, load=True)
                loss = 'categorical_crossentropy'

            # self.model._make_predict_function()
            self.model.load_weights(self.path)
            self.model.predict(np.zeros((1, 224, 224, 6)))
            print("=> loaded checkpoint '{}'".format(self.path))

        else:
            if self.type == 'contrastive':
                print('contrastive')
                self.model = contrastive(input_shape, model=self.feat, load=False)
            else:
                self.model = siamese(input_shape, model=self.feat, load=False)

        if self.num_gpus > 1:
            self.model = ModelMGPU(self.model, self.num_gpus)

        adam = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        if self.type == 'contrastive':
            self.model.compile(loss=loss, optimizer=adam, metrics=[accuracy])
        else:
            self.model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

        self.history = []

    def load_model(self, model_path):
        print('loading')
        self.model.load_weights(model_path)
        self.model.predict(np.zeros((1, 224, 224, 6)))
        print ("=> loaded checkpoint '{}'".format(model_path))

    def train(self, datasets=(0, 1)):
        # /home/etienne/data/default_experiment/models/model.{epoch:06d}.h5
        current_model_directory = data_prefix + '/data/similarities/%s/%s'%(self.type, self.feat)
        import os
        if not os.path.exists(current_model_directory):
            os.makedirs(current_model_directory)
        current_model_path = current_model_directory+'/model.{epoch:06d}.h5'
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=1e-7)

        print(current_model_path)
        # tensorboard = keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False)
        checkpoint = keras.callbacks.ModelCheckpoint(current_model_path,
                                                          save_best_only=True, monitor='val_loss', mode='min',
                                                          verbose=1)

        callbacks_list = [lr_reducer, lr_scheduler,
                          # """tensorboard,"""
                          checkpoint]
        dataset = generate_dataset((datasets[0], datasets[1]))

        print(dataset.label.value_counts())
        validation_set = dataset.sample(int(dataset.shape[0] * 0.2), replace=False)
        training_set = dataset.drop(validation_set.index)
        if self.type == 'contrastive':
            n_classes = 1
        else:
            n_classes = 2
        train_datagen = DataGenerator(training_set, batch_size=BATCH_SIZE, augment=True, n_classes=n_classes)
        valid_datagen = DataGenerator(validation_set, batch_size=BATCH_SIZE, n_classes=n_classes)

        history = self.model.fit_generator(train_datagen,
                                            epochs=50,
                                            steps_per_epoch=min(5000, int(training_set.shape[0] / BATCH_SIZE)),
                                           # steps_per_epoch=1,
                                            callbacks=callbacks_list,
                                            validation_steps=min(int(validation_set.shape[0] / BATCH_SIZE), 500),
                                           # validation_steps=1,
                                            validation_data=valid_datagen,
                                            shuffle=True)
        self.history.append(history)

    def predict(self, image1, image2):
        # preprocess in the image_net preprocess (linear combination so doesn't matter which one as long as its consistent with the training
        image1 = preprocess_input(image1)
        image2 = preprocess_input(image2)
        input1 = np.concatenate([image1, image2], axis=-1)
        input1 = input1.reshape((1,) + input1.shape)
        similarity = self.model.predict(input1)[0][1]
        return similarity

    def predict_frames(self, frame1, frame2, directory, show=False):
        image2 = directory + '/rgb_0/' + str(frame1) + '.jpg'
        image2 = np.array(Image.open(image2).resize((224, 224)))
        image1 = directory +'/rgb_0/' + str(frame2) + '.jpg'
        image1 = np.array(Image.open(image1).resize((224, 224)))
        return self.predict(image1, image2)


def main(feat='resnet', type='siamese'):
    sim = SimilarityNetwork(input_shape, type=type, feat=feat)
    sim.train((0, 6))
    plot_history(sim.history[-1])
    return sim


if __name__ == '__main__':
    input_shape = RESHAPE + (6,)
    # sim = SimilarityNetwork(input_shape, type='contrastive', model='vgg', load=False)
    # dataset = generate_dataset((0, 1))
    # train_datagen = DataGenerator(dataset, batch_size=32)
    # print(sim.model.evaluate_generator(train_datagen, steps=20, verbose=1))
    # for i in range(6):
    #     dataset = generate_dataset((i, i+1))
    #     train_datagen = DataGenerator(dataset, batch_size=32)
    #     print(sim.model.evaluate_generator(train_datagen, steps=20, verbose=0))
    main(feat='vgg', type='siamese')
    # main(feat='resnet', type='contrastive')
    # main(feat='vgg', type='contrastive')

