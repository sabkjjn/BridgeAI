from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D, Conv1D,
                                     Conv2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, Reshape,
                                     Flatten, Add, Concatenate, Input, Permute, multiply, GlobalAveragePooling1D)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy import signal as scipysignal
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# 构造带通滤波
fs = 1000 / 2
FLTNUM = scipysignal.firwin(1000 * 2 + 1, np.array([8, 30]) / fs, pass_zero='bandpass')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"使用GPU: {physical_devices}")
else:
    print("未找到GPU，使用CPU")

def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# 提取特定trigger数据段的函数
def extract_specific_segments(eeg_data, trigger_signals, trigger):
    segments = []
    i = 0
    while i < len(trigger_signals):
        if trigger_signals[i] == trigger:
            start_index = i
            end_index = start_index + 4000
            if end_index <= eeg_data.shape[1]:  # 确保段落不会超出数据范围
                segment = eeg_data[:, start_index:end_index]
                segment = scipysignal.filtfilt(FLTNUM, 1, segment, padlen=len(segment)-1)
                segments.append(segment)
        i += 1
    return np.array(segments)

# 拼接函数
def concatenate_segments(segments_list):
    min_len = min(len(segments_list[0]), len(segments_list[1]), len(segments_list[2]))
    segments_list = [seg[:min_len] for seg in segments_list]
    concatenated_segments = np.concatenate(segments_list, axis=2)
    return concatenated_segments

# 准备数据集的函数
def prepare_dataset(filepaths):
    all_segments = []
    all_labels = []
    triggers_list = [11, 21, 31]
    labels = [0, 1, 2]

    for filepath in filepaths:
        data = load_data(filepath)
        eeg_data = data['data'][:-1, :]
        trigger_signals = data['data'][-1, :]
        for triggers, label in zip(triggers_list, labels):
            segments = extract_specific_segments(eeg_data, trigger_signals, triggers)
            all_segments.append(segments)
            all_labels.append(np.full(segments.shape[0], label))

    return np.concatenate(all_segments), np.concatenate(all_labels)
def downsample_data(data, factor=2):
    downsampled_data = data[:, :, ::factor]
    return downsampled_data
# 标准化函数
def standardize_data(X_train, X_test, channels):
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, j, :])
        X_train[:, j, :] = scaler.transform(X_train[:, j, :])
        scaler.fit(X_test[:, j, :])
        X_test[:, j, :] = scaler.transform(X_test[:, j, :])
    return X_train, X_test

# 数据集文件路径
train_filepaths = [
    './TrainingData/s1/train/block1.pkl',
    './TrainingData/s1/train/block2.pkl',
    './TrainingData/s1/train/block3.pkl'
]

test_filepaths = [
    './TrainingData/s1/test/block1.pkl',
    './TrainingData/s1/test/block2.pkl',
    './TrainingData/s1/test/block3.pkl'
]

# 准备训练集和测试集
X_train, y_train = prepare_dataset(train_filepaths)
X_test, y_test = prepare_dataset(test_filepaths)

# X_train = downsample_data(X_train)
# X_test = downsample_data(X_test)
# ch_names = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3',
#             'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5',
#             'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7',
#             'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz',
#             'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6',
#             'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
#
# # 需要提取的22个通道名称
# selected_ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
#                      'C2', 'C4', 'C6', 'CP3', 'CP1', 'CP2', 'CP4', 'Pz', 'POz']
#
# # 查找需要提取通道在原始64通道中的索引
# selected_indices = [ch_names.index(ch) for ch in selected_ch_names]
#
# # 提取需要的通道数据
# X_train = X_train[:, selected_indices, :]
# X_test = X_test[:, selected_indices, :]
# 检查标签分布
print("y_train distribution:", np.bincount(y_train))
print("y_test distribution:", np.bincount(y_test))
# 将y_train和y_test转换为one-hot编码
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)
# 数据归一化
X_train, X_test = standardize_data(X_train, X_test, X_train.shape[1])

# 调整数据形状以符合模型输入
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
# 随机打乱训练集
perm = np.random.permutation(X_train.shape[0])
X_train = X_train[perm]
y_train = y_train[perm]

# X_train = X_train.transpose(0, 2, 3, 1)  # 变为 (样本数, 通道数, 时间点数, 1)
# X_test = X_test.transpose(0, 2, 3, 1)

# 检查训练数据形状
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 模型结构定义
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras.constraints import max_norm
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# class AccuracyHistory(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"Epoch {epoch + 1}: Train accuracy = {logs['accuracy']}, Test accuracy = {logs['val_accuracy']}")
#
# def LC_Block(input_layer, F1=8, kernLength=64, D=2, Chans=22, dropout=0.25, activation='elu', AveragePooling=True):
#     conv_block1 = Conv2D(F1, kernel_size=(1, kernLength), padding='same', data_format='channels_first', use_bias=False)(input_layer)
#     conv_block1 = BatchNormalization(axis=1)(conv_block1)
#
#     conv_block2 = DepthwiseConv2D(kernel_size=(Chans, 1), depth_multiplier=D, data_format='channels_first', use_bias=False, depthwise_constraint=max_norm(1.))(conv_block1)
#     conv_block2 = BatchNormalization(axis=1)(conv_block2)
#     conv_block2 = Activation(activation)(conv_block2)
#     if AveragePooling:
#         conv_block2 = AveragePooling2D(pool_size=(1, kernLength // 8), data_format='channels_first')(conv_block2)
#     else:
#         conv_block2 = MaxPooling2D(pool_size=(1, kernLength // 8), data_format='channels_first')(conv_block2)
#     conv_block2 = Dropout(dropout)(conv_block2)
#
#     conv_block3 = SeparableConv2D(F1*D, kernel_size=(1, kernLength // 4), padding='same', data_format='channels_first', use_bias=False)(conv_block2)
#     conv_block3 = BatchNormalization(axis=1)(conv_block3)
#     conv_block3 = Activation(activation)(conv_block3)
#     if AveragePooling:
#         conv_block3 = AveragePooling2D(pool_size=(1, kernLength // 8), data_format='channels_first')(conv_block3)
#     else:
#         conv_block3 = MaxPooling2D(pool_size=(1, kernLength // 8), data_format='channels_first')(conv_block3)
#     conv_block3 = Dropout(dropout)(conv_block3)
#     conv_block3 = K.squeeze(conv_block3, axis=-2)
#
#     return conv_block3
#
# def SE_Block(input_layer, Seize=2, activation1='relu', activation2='sigmoid', BandSE=True):
#     if BandSE:
#         aver_block = GlobalAveragePooling1D(data_format='channels_first')(input_layer)
#         bands = input_layer.shape[1]
#         aver_block = Reshape((1, bands))(aver_block)
#         se_block = Dense(Seize, activation=activation1, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(aver_block)
#         se_block = Dense(bands, activation=activation2, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(se_block)
#         se_block = Permute((2, 1))(se_block)
#     else:
#         aver_block = GlobalAveragePooling1D(data_format='channels_last')(input_layer)
#         timePoints = input_layer.shape[2]
#         aver_block = Reshape((1, timePoints))(aver_block)
#         se_block = Dense(Seize, activation=activation1, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(aver_block)
#         se_block = Dense(timePoints, activation=activation2, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(se_block)
#
#     se_block = multiply([input_layer, se_block])
#
#     return se_block
#
# def GC_Block(input_layer, dropout=0.3, depth=2, activation='elu', kernel_size=4, TimeConv=True, n_windows=5, step=4, seize=2):
#     F1 = input_layer.shape[1]
#     F2 = input_layer.shape[2]
#     sw_concat = []
#
#     if TimeConv:
#         for j in range(n_windows):
#             st = j * step
#             end = F2 - (n_windows - j - 1) * step
#             sw = input_layer[:, :, st:end]
#             se_block = SE_Block(input_layer=sw, BandSE=False, Seize=seize, activation1='relu')
#             last_block = se_block
#             for i in range(depth):
#                 block_1 = Conv1D(F1, kernel_size=kernel_size, dilation_rate=i+1, padding='causal', kernel_initializer='he_uniform', data_format='channels_first')(last_block)
#                 block_1 = BatchNormalization(axis=1)(block_1)
#                 block_1 = Activation(activation)(block_1)
#                 block_1 = Dropout(dropout)(block_1)
#                 add_block = Add()([block_1, se_block])
#                 last_block = Activation(activation)(add_block)
#             fl_block = Flatten()(last_block)
#             sw_concat.append(fl_block)
#         ca_block = Concatenate()(sw_concat)
#     else:
#         for j in range(n_windows):
#             st = j * step
#             end = F1 - (n_windows - j - 1) * step
#             sw = input_layer[:, st:end, :]
#             se_block = SE_Block(input_layer=sw, BandSE=True, Seize=seize, activation1='relu')
#             last_block = se_block
#             for i in range(depth):
#                 block_1 = Conv1D(F2, kernel_size=kernel_size, dilation_rate=i+1, padding='causal', kernel_initializer='he_uniform', data_format='channels_last')(last_block)
#                 block_1 = BatchNormalization(axis=-1)(block_1)
#                 block_1 = Activation(activation)(block_1)
#                 block_1 = Dropout(dropout)(block_1)
#                 add_block = Add()([block_1, se_block])
#                 last_block = Activation(activation)(add_block)
#             fl_block = Flatten()(last_block)
#             sw_concat.append(fl_block)
#         ca_block = Concatenate()(sw_concat)
#
#     return ca_block
#
# def EEG_DBNet(nb_classes=4, Chans=22, Samples=1125, regRate=0.25, d=4, k=4, n=6, s=1, se=2):
#     inputs = Input(shape=(1, Chans, Samples))
#
#     LC_Block1 = LC_Block(input_layer=inputs, F1=8, kernLength=48, Chans=Chans, dropout=0.3, activation='elu', AveragePooling=True)
#     LC_Block2 = LC_Block(input_layer=inputs, F1=16, kernLength=64, Chans=Chans, dropout=0.3, activation='elu', AveragePooling=False)
#
#     GC_Block1 = GC_Block(input_layer=LC_Block1, TimeConv=True, depth=d, kernel_size=k, n_windows=n, step=s, seize=se, activation='elu')
#     GC_Block2 = GC_Block(input_layer=LC_Block2, TimeConv=False, depth=d, kernel_size=k, n_windows=n, step=s, seize=se, activation='elu')
#
#     conc_block = Concatenate()([GC_Block1, GC_Block2])
#
#     dense_block = Dense(nb_classes, kernel_constraint=max_norm(regRate))(conc_block)
#     softmax = Activation('softmax')(dense_block)
#
#     return Model(inputs=inputs, outputs=softmax)
# # 构建和训练模型
# class AccuracyHistory(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"Epoch {epoch + 1}: Train accuracy = {logs['accuracy']}, Test accuracy = {logs['val_accuracy']}")
#
# # 构建模型
# model = EEG_DBNet(nb_classes=3, Chans=X_train.shape[2], Samples=X_train.shape[3])
#
# # 编译模型
# optimizer = Adam(learning_rate=0.0005)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 定义回调
# early_stopping = EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, min_lr=1e-5)
#
# # train_data, test_data, train_labels, test_labels = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
# # 训练模型
# # history = model.fit(train_data, train_labels, epochs=100, batch_size=8, validation_data=(test_data, test_labels),
# #                     callbacks=[AccuracyHistory(), early_stopping, reduce_lr])
#
# # 训练模型
# history = model.fit(X_train, y_train, epochs=300, batch_size=8, validation_data=(X_test, y_test),
#                     callbacks=[AccuracyHistory(), early_stopping, reduce_lr])
#
# # 可视化混淆矩阵
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test, axis=1)
#
# conf_matrix = confusion_matrix(y_true, y_pred_classes)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
# disp.plot()
# plt.show()

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import SpatialDropout2D
# from tensorflow.keras.regularizers import l1_l2
# from tensorflow.keras.layers import Input, Flatten
# from tensorflow.keras.constraints import max_norm
# def EEGNet(nb_classes, Chans = 64, Samples = 3953,
#              dropoutRate = 0.5, kernLength = 64, F1 = 8,
#              D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
#
#     if dropoutType == 'SpatialDropout2D':
#         dropoutType = SpatialDropout2D
#     elif dropoutType == 'Dropout':
#         dropoutType = Dropout
#     else:
#         raise ValueError('dropoutType must be one of SpatialDropout2D '
#                          'or Dropout, passed as a string.')
#
#     input1 = Input(shape = (Chans, Samples, 1))
#     print("input shape", input1.shape, Chans, Samples, kernLength)
#     ##################################################################
#     block1 = Conv2D(F1, (1, kernLength), padding = 'same',
#                                    input_shape = (Chans, Samples, 1),
#                                    use_bias = False)(input1)
#     block1 = BatchNormalization()(block1)
#     block1 = DepthwiseConv2D((Chans, 1), use_bias = False,
#                                    depth_multiplier = D,
#                                    depthwise_constraint = max_norm(1.))(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Activation('elu')(block1)
#     block1 = AveragePooling2D((1, 4))(block1)
#     block1 = dropoutType(dropoutRate)(block1)
#
#     block2 = SeparableConv2D(F2, (1, 16),
#                                    use_bias = False, padding = 'same')(block1)
#     block2 = BatchNormalization()(block2)
#     block2 = Activation('elu')(block2)
#     block2 = AveragePooling2D((1, 8))(block2)
#     block2 = dropoutType(dropoutRate)(block2)
#     flatten = Flatten(name = 'flatten')(block2)
#
#     dense = Dense(nb_classes, name = 'dense',
#                          kernel_constraint = max_norm(norm_rate))(flatten)
#     softmax = Activation('softmax', name = 'softmax')(dense)
#
#     return Model(inputs=input1, outputs=softmax)
#
#
#
# from tensorflow.keras.callbacks import ModelCheckpoint
# model = EEGNet(nb_classes = 3, Chans = 64, Samples = 4000,
#                dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
#                dropoutType = 'Dropout')
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
# # count number of parameters in the model
# numParams = model.count_params()
# # set a valid path for your system to record model checkpoints
# # checkpointer = ModelCheckpoint(filepath='checkpoint.h5', verbose=1,
# #                                save_best_only=True)
# # class_weights = {0: 1, 1: 1, 2: 1, 3: 1}
# fittedModel = model.fit(X_train, y_train, batch_size=8, epochs=300,
#                         verbose=2, validation_data=(X_test, y_test))
#
# # model.load_weights('checkpoint.h5')
# # probs = model.predict(X_test)
# # preds = probs.argmax(axis=-1)
# # acc = np.mean(preds == y_test.argmax(axis=-1))
# # print("Classification accuracy: %f " % (acc))
# #
# # # plot the accuracy and loss graph
# # plt.plot(fittedModel.history['accuracy'])
# # plt.plot(fittedModel.history['val_accuracy'])
# # plt.plot(fittedModel.history['loss'])
# # plt.plot(fittedModel.history['val_loss'])
# # plt.title('acc & loss')
# # plt.xlabel('epoch')
# # plt.legend(['acc', 'val_acc','loss','val_loss'], loc='upper right')
# # plt.show()

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Dense, Dropout, Activation, Conv2D, BatchNormalization, DepthwiseConv2D, AveragePooling2D, Lambda, GlobalAveragePooling2D, Reshape, multiply, Add, Flatten, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 定义ATCNet模型
def ATCNet(n_classes, in_chans=22, in_samples=1125, n_windows=5, attention='mha',
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):

    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3,2,1))(input_1)

    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    from_logits = False

    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:,:,-1,:])(block1)

    sw_concat = []
    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        if attention is not None:
            if attention == 'se' or attention == 'cbam':
                block2 = Permute((2, 1))(block2)
                block2 = attention_block(block2, attention)
                block2 = Permute((2, 1))(block2)
            else:
                block2 = attention_block(block2, attention)

        block3 = TCN_block_(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        block3 = Lambda(lambda x: x[:,-1,:])(block3)

        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_regularizer=tf.keras.regularizers.L2(dense_weightDecay))(block3))
        elif fuse == 'concat':
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if fuse == 'average':
        if len(sw_concat) > 1:
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:
            sw_concat = sw_concat[0]
    elif fuse == 'concat':
        sw_concat = Dense(n_classes, kernel_regularizer=tf.keras.regularizers.L2(dense_weightDecay))(sw_concat)

    if from_logits:
        out = Activation('linear', name='linear')(sw_concat)
    else:
        out = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=out)

#%% Convolutional (CV) block used in the ATCNet model
def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay=0.009, maxNorm=0.6, dropout=0.25):
    F2= F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0,1,2]),
                    use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    block2 = DepthwiseConv2D((1, in_chans), depth_multiplier=D, data_format='channels_last',
                             depthwise_regularizer=tf.keras.regularizers.L2(weightDecay),
                             depthwise_constraint=max_norm(maxNorm, axis=[0,1,2]),
                             use_bias=False)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((poolSize,1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    block3 = Conv2D(F2, (16, 1), data_format='channels_last',
                    kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0,1,2]),
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((poolSize,1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

#%% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block_(input_layer, input_dimension, depth, kernel_size, filters, dropout,
               weightDecay=0.009, maxNorm=0.6, activation='relu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0,1]),
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                   kernel_constraint=max_norm(maxNorm, axis=[0,1]),
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1, kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                      kernel_constraint=max_norm(maxNorm, axis=[0,1]), padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth-1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                       kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0,1]),
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2**(i+1), activation='linear',
                       kernel_regularizer=tf.keras.regularizers.L2(weightDecay),
                       kernel_constraint=max_norm(maxNorm, axis=[0,1]),
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out

#%% Create and apply the attention model
def attention_block(in_layer, attention_model, ratio=8, residual=False, apply_to_input=True):
    in_sh = in_layer.shape
    in_len = len(in_sh)
    expanded_axis = 2

    if attention_model == 'mha':
        if in_len > 3:
            in_layer = Reshape((in_sh[1],-1))(in_layer)
        out_layer = mha_block(in_layer)
    elif attention_model == 'mhla':
        if in_len > 3:
            in_layer = Reshape((in_sh[1],-1))(in_layer)
        out_layer = mha_block(in_layer, vanilla=False)
    elif attention_model == 'se':
        if in_len < 4:
            in_layer = tf.expand_dims(in_layer, axis=expanded_axis)
        out_layer = se_block(in_layer, ratio, residual, apply_to_input)
    elif attention_model == 'cbam':
        if in_len < 4:
            in_layer = tf.expand_dims(in_layer, axis=expanded_axis)
        out_layer = cbam_block(in_layer, ratio=ratio, residual=residual)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_model))

    if in_len == 3 and len(out_layer.shape) == 4:
        out_layer = tf.squeeze(out_layer, expanded_axis)
    elif in_len == 4 and len(out_layer.shape) == 3:
        out_layer = Reshape((in_sh[1], in_sh[2], in_sh[3]))(out_layer)
    return out_layer

#%% Multi-head self Attention (MHA) block
def mha_block(input_feature, key_dim=8, num_heads=2, dropout=0.5, vanilla=True):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_feature)

    if vanilla:
        x = tf.keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)
    else:
        NUM_PATCHES = input_feature.shape[1]
        diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

        x = MultiHeadAttention_LSA(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(
            x, x, attention_mask=diag_attn_mask)
    x = Dropout(0.3)(x)
    mha_feature = Add()([input_feature, x])

    return mha_feature

#%% Multi head self Attention (MHA) block: Locality Self Attention (LSA)
class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.Variable(tf.math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

#%% Squeeze-and-excitation block
def se_block(input_feature, ratio=8, residual=False, apply_to_input=True):
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if ratio != 0:
        se_feature = Dense(channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')(se_feature)
        assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if tf.keras.backend.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    if apply_to_input:
        se_feature = multiply([input_feature, se_feature])

    if residual:
        se_feature = Add()([se_feature, input_feature])

    return se_feature

#%% Convolutional block attention module
def cbam_block(input_feature, ratio=8, residual=False):
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)

    if residual:
        cbam_feature = Add()([input_feature, cbam_feature])

    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if tf.keras.backend.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if tf.keras.backend.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1,
                          padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if tf.keras.backend.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

# 构建和训练模型
class AccuracyHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Train accuracy = {logs['accuracy']}, Test accuracy = {logs['val_accuracy']}")

# 构建模型
model = ATCNet(n_classes=3, in_chans=X_train.shape[2], in_samples=X_train.shape[3],n_windows=5, attention='mha',
           eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=16, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average')

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 定义回调
early_stopping = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=50, min_lr=1e-5)

# 训练模型
history = model.fit(X_train, y_train, epochs=300, batch_size=8, validation_data=(X_test, y_test),
                    callbacks=[AccuracyHistory(), early_stopping, reduce_lr])

# 可视化混淆矩阵
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.show()