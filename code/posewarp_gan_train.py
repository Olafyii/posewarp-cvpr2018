import tensorflow as tf
import os
import numpy as np
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.optimizers import Adam
from keras.models import Model

import cv2

def train(model_name, gpu_id):
    params = param.get_general_params()
    network_dir = params['model_save_dir'] + '/' + model_name

    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)

    train_feed = data_generation.create_feed(params, params['data_dir'], 'train')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    gan_lr = 1e-4
    disc_lr = 1e-4
    disc_loss = 0.1

    generator = networks.network_posewarp(params)
    # generator.load_weights('../models/vgg_100000.h5')
    generator.load_weights('/versa/kangliwei/motion_transfer/posewarp-cvpr2018/models/0228_fulltrain/199000.h5')

    discriminator = networks.discriminator(params)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')

    gan = networks.gan(generator, discriminator, params)
    gan.compile(optimizer=Adam(lr=gan_lr),
                loss=[networks.vgg_loss(vgg_model, response_weights, 12), 'binary_crossentropy'],
                loss_weights=[1.0, disc_loss])

    n_iters = 10000
    batch_size = params['batch_size']

    for step in range(n_iters):

        x, y = next(train_feed)

        gen = generator.predict(x)

        # Train discriminator
        x_tgt_img_disc = np.concatenate((y, gen))
        x_src_pose_disc = np.concatenate((x[1], x[1]))
        x_tgt_pose_disc = np.concatenate((x[2], x[2]))

        L = np.zeros([2 * batch_size])
        L[0:batch_size] = 1

        inputs = [x_tgt_img_disc, x_src_pose_disc, x_tgt_pose_disc]
        d_loss = discriminator.train_on_batch(inputs, L)

        # Train the discriminator a couple of iterations before starting the gan
        if step < 5:
            util.printProgress(step, 0, [0, d_loss])
            step += 1
            continue

        # TRAIN GAN
        L = np.ones([batch_size])
        x, y = next(train_feed)
        g_loss = gan.train_on_batch(x, [y, L])
        util.printProgress(step, 0, [g_loss[1], d_loss])

        if step % params['model_save_interval'] == 0 and step > 0:
            generator.save(network_dir + '/' + str(step) + '.h5')


def test(model_name, gpu_id):
    params = param.get_general_params()
    network_dir = params['model_save_dir'] + '/' + model_name

    # if not os.path.isdir(network_dir):
    #     os.mkdir(network_dir)

    train_feed = data_generation.create_feed(params, params['data_dir'], 'test', do_augment=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    gan_lr = 1e-4
    disc_lr = 1e-4
    disc_loss = 0.1

    generator = networks.network_posewarp(params)
    # generator.load_weights('../models/vgg_100000.h5')
    generator.load_weights('/versa/kangliwei/motion_transfer/posewarp-cvpr2018/models/0301_fullfinetune/9000.h5')

    mask_delta_model = Model(input=generator.input, output=generator.get_layer('mask_delta').output)
    src_mask_model = Model(input=generator.input, output=generator.get_layer('mask_src').output)

    discriminator = networks.discriminator(params)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')

    gan = networks.gan(generator, discriminator, params)
    gan.compile(optimizer=Adam(lr=gan_lr),
                loss=[networks.vgg_loss(vgg_model, response_weights, 12), 'binary_crossentropy'],
                loss_weights=[1.0, disc_loss])

    n_iters = 10000
    batch_size = params['batch_size']

    for step in range(n_iters):

        x, y = next(train_feed)

        gen = generator.predict(x)

        src_mask_delta = mask_delta_model.predict(x)
        print('delta_max', src_mask_delta.max())
        src_mask_delta = src_mask_delta * 255
        src_mask = src_mask_model.predict(x)
        print('mask_max', src_mask.max())
        src_mask = src_mask * 255
        # print('src_mask_delta', type(src_mask_delta), src_mask_delta.shape)
        
        y = (y / 2 + 0.5 ) * 255.0
        gen = (gen / 2 + 0.5 ) * 255.0
        for i in range(gen.shape[0]):  # iterate in batch
            cv2.imwrite('pics/src'+str(i)+'.jpg', x[0][i]*255)
            cv2.imwrite('pics/gen'+str(i)+'.jpg', gen[i])
            cv2.imwrite('pics/y'+str(i)+'.jpg', y[i])
            for j in range(11):
                cv2.imwrite('pics/seg_delta_'+str(i)+'_'+str(j)+'.jpg', src_mask_delta[i][:,:,j])
            for j in range(11):
                cv2.imwrite('pics/seg_'+str(i)+'_'+str(j)+'.jpg', src_mask[i][:,:,j])
        break

        # Train discriminator
        x_tgt_img_disc = np.concatenate((y, gen))
        x_src_pose_disc = np.concatenate((x[1], x[1]))
        x_tgt_pose_disc = np.concatenate((x[2], x[2]))

        L = np.zeros([2 * batch_size])
        L[0:batch_size] = 1

        inputs = [x_tgt_img_disc, x_src_pose_disc, x_tgt_pose_disc]
        d_loss = discriminator.train_on_batch(inputs, L)

        # Train the discriminator a couple of iterations before starting the gan
        if step < 5:
            util.printProgress(step, 0, [0, d_loss])
            step += 1
            continue

        # TRAIN GAN
        L = np.ones([batch_size])
        x, y = next(train_feed)
        g_loss = gan.train_on_batch(x, [y, L])
        util.printProgress(step, 0, [g_loss[1], d_loss])

        if step % params['model_save_interval'] == 0 and step > 0:
            generator.save(network_dir + '/' + str(step) + '.h5')


if __name__ == "__main__":
    if True:
        test(sys.argv[1], sys.argv[2])
    else:
        if len(sys.argv) != 3:
            print("Need model name and gpu id as command line arguments.")
        else:
            train(sys.argv[1], sys.argv[2])
