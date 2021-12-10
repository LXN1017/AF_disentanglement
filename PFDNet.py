import time
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.io import loadmat as load
import numpy as np
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Bidirectional
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


tf.__version__
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def feature_normalize(data):
    for i in range(data.shape[0]):
        min_val = np.min(data[i, :])
        max_val = np.max(data[i, :])
        data[i, :] = (data[i, :] - min_val) / (max_val - min_val) - 0.5
    return data


# -----------------------------------------------导入样本和标签----------------------------------------------------
bvp_af = load('Dataset/bvp_af_sum_tip.mat')
label_af = load('Dataset/label_af_tip.mat')
bvp_nor = load('Dataset/bvp_nor_sum_tip.mat')
label_nor = load('Dataset/label_nor_tip.mat')
PPG = np.append(bvp_af['bvp_af_sum_tip'], bvp_nor['bvp_nor_sum_tip'], axis=0)
PPG = feature_normalize(PPG)  # 训练样本
VPPG = PPG + 4 * np.random.rand(PPG.shape[0], PPG.shape[1])
PPG = np.expand_dims(PPG, axis=-1)
PPG = np.expand_dims(PPG, axis=1)
PPG = PPG.astype(np.float32)
VPPG = np.expand_dims(VPPG, axis=-1)
VPPG = np.expand_dims(VPPG, axis=1)
VPPG = VPPG.astype(np.float32)
Label = np.append(label_af['label_af_tip'], label_nor['label_nor_tip'], axis=0)  # 标签
Label = np.array(tf.squeeze(tf.one_hot(Label, 2)))

BATCH_SIZE = 72
BUFFER_SIZE = 1440
datasets = tf.data.Dataset.from_tensor_slices(np.append(PPG, VPPG, axis=1))
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

spa = 1
tem = 600
channel = 1
win_num = 5
win_len = 120
ker_num = 64


def encoder(input):  # 输入一副图像，spa、tem、channel分别表示空间、时间、通道;win_num, win_len分别表示窗数和窗长，ker_num表示卷积核数目
    cnn1 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh')(input)
    cnn2 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh')(cnn1)
    cnn2 = tf.reshape(cnn2, [-1, win_num, spa, win_len, ker_num])
    lstm1 = Bidirectional(tf.keras.layers.ConvLSTM2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh', return_sequences=True))(cnn2)  # 出现128事因为双向LSTM
    lstm1 = tf.reshape(lstm1, [-1, spa, tem, 2*ker_num])
    cnn3 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh')(lstm1)
    return cnn3


def decoder(input):
    cnn1 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh')(input)
    cnn1 = tf.reshape(cnn1, [-1, win_num, spa, win_len, ker_num])
    lstm1 = Bidirectional(tf.keras.layers.ConvLSTM2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh', return_sequences=True))(cnn1)
    lstm1 = tf.reshape(lstm1, [-1, spa, tem, 2*ker_num])
    cnn2 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), padding='same', activation='tanh')(lstm1)
    cnn3 = tf.keras.layers.Convolution2D(filters=1, kernel_size=(1, 3), padding='same', activation='tanh')(cnn2)
    return cnn3


def classifier(pulse):
    cnn1 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), strides=(1, 3), padding='same', activation='tanh')(pulse)
    cnn2 = tf.keras.layers.Convolution2D(filters=ker_num, kernel_size=(1, 3), strides=(1, 3), padding='same', activation='tanh')(cnn1)
    pool1 = tf.keras.layers.Flatten()(cnn2)
    fc1 = tf.keras.layers.Dense(600)(pool1)
    fc2 = tf.keras.layers.Dense(300)(fc1)
    fc3 = tf.keras.layers.Dense(2)(fc2)
    return fc3


def generator_model():
    # --------------------------Input Layer----------------------------------
    input1 = tf.keras.Input(shape=(spa, tem, channel), name='VPPG_map')
    input2 = tf.keras.Input(shape=(spa, tem, channel), name='PPG_map')
    # --------------------------Encoding--------------------------------------
    VPPG_share_en = encoder(input1)
    VPPG_exclu_en = encoder(input1)
    PPG_share_en = encoder(input2)
    PPG_exclu_en = encoder(input2)
    # --------------------------Feature Recombination------------------------
    VPPG_self = VPPG_share_en + VPPG_exclu_en
    VPPG_trans = VPPG_exclu_en + PPG_share_en
    PPG_self = PPG_share_en + PPG_exclu_en
    PPG_trans = PPG_exclu_en + VPPG_share_en
    # ------------------------------Decoding------------------------------------
    VPPG_self_de = decoder(VPPG_self)  # VPPG共享+VPPG私有=VPPG信号
    VPPG_trans_de = decoder(VPPG_trans)  # VPPG私有+PPG共享=VPPG信号
    PPG_self_de = decoder(PPG_self)  # PPG共享+PPG私有=PPG信号
    PPG_trans_de = decoder(PPG_trans)  # PPG私有+VPPG共享=PPG信号
    # -------------------------------Output Layer------------------------------------
    output1 = VPPG_self_de
    output2 = VPPG_trans_de
    output3 = PPG_self_de
    output4 = PPG_trans_de
    # ------------------------------Model---------------------------------------
    model = tf.keras.Model(inputs=[input1, input2], outputs=[output1, output2, output3, output4])
    return model


def discriminator_model():  # 识别输入的图片
    input1 = tf.keras.Input(shape=(spa, tem, channel))
    input2 = tf.keras.Input(shape=(spa, tem, channel))
    input3 = tf.keras.Input(shape=(spa, tem, channel))
    input4 = tf.keras.Input(shape=(spa, tem, channel))
    output1 = classifier(input1)
    output2 = classifier(input2)
    output3 = classifier(input3)
    output4 = classifier(input4)
    model = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=[output1, output2, output3, output4])
    return model


cross_entropy = tf.keras.losses.categorical_crossentropy


def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss


def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)


generator_opt = keras.optimizers.Adam(1e-4)
discriminator_opt = keras.optimizers.Adam(1e-4)

generator = generator_model()
discriminator = discriminator_model()


def train_step(image_batch):
    PPG_len = int(image_batch.shape[1] / 2)
    PPG = image_batch[:, 0:PPG_len]
    VPPG = image_batch[:, PPG_len:2 * PPG_len]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator([VPPG, VPPG, PPG, PPG], training=True)  # real_out: 判别器对于真图像的输出
        re_pulse = generator([VPPG, PPG], training=True)  #
        fake_out = discriminator(re_pulse, training=True)  # fake_out: 判别器对于假图像的输出

        gen_loss = generator_loss(fake_out)  # 希望将假图像分类为1
        disc_loss = discriminator_loss(real_out, fake_out)  # 将真实图像分类为1，虚假图像分类为0，分类的交叉熵
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)  # 考虑生成器路，基于GAN输出和标签的损失函数，计算生成器的可训练参数的梯度带
    gradient_disc = disc_tape.gradient(disc_loss,
                                       discriminator.trainable_variables)  # 考虑判别器路，基于GAN输出和标签的损失函数，计算判别器的可训练参数的梯度带
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))  # 将优化器应用于生成器梯度带，更新一次参数
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))  # 将优化器应用于生成器梯度带，更新一次参数


def generate_plot_image(gen_model, test_noise):
    pre_images = gen_model(test_noise, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.plot(np.squeeze(pre_images[i, :]))
        plt.axis('off')
    plt.show()


EPOCHS = 10  # 训练100次
num_exp_to_generate = 16  # 生成16张图片
seed = tf.random.normal([num_exp_to_generate, tem])  # 16组随机数组，每组含100个随机数，用来生成16张图片。


def train(datasets, epochs):
    for epoch in range(epochs):
        for image_batch in datasets:
            train_step(image_batch)
            print('.', end='')
        if epoch % 50 == 0:
            print(epoch)
            # generate_plot_image(generator, seed)


time_start = time.time()
train(datasets, EPOCHS)
time_end = time.time()
print('totally cost', time_end-time_start)

time_start = time.time()
test = generator.predict([VPPG, PPG])
time_end = time.time()
print('totally cost', time_end-time_start)