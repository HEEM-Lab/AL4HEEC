# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:35:51 2023

@author: 18326
"""

# https://keras.io/examples/generative/conditional_gan/

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from bayes_opt import BayesianOptimization
import json

from agat.lib import config_parser

# # tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

# os.envirion["CUDA_VISIBLE_DEVICES"] = '-1'

class ConNormalization(tf.keras.layers.Layer):
  def __init__(self):
      super(ConNormalization, self).__init__()

  def call(self, inputs):
      con_total = tf.reduce_sum(inputs, axis=1, keepdims=True)
      con_norm = tf.math.divide(inputs, con_total)
      return con_norm

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.gen_mae_tracker = keras.metrics.MeanAbsoluteError(name="generator_MAE")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker,
                self.gen_mae_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def save_model(self, generator_dir='generator.keras',
                   discriminator_dir='discriminator.keras'):
        self.generator.save(os.path.join(generator_dir))
        self.discriminator.save(os.path.join(discriminator_dir))

    def train_step(self, data):
        # Unpack the data.
        real_cons, real_labels = data

        real_cons = tf.cast(real_cons, dtype=tf.float32)
        real_labels = tf.reshape(real_labels, (-1,1))
        real_labels = tf.cast(real_labels, dtype=tf.float32)

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_cons)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                        self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, real_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_cons = self.generator(random_vector_labels)

        fake_cons_and_labels = tf.concat([generated_cons, real_labels], -1)
        real_cons_and_labels = tf.concat([real_cons, real_labels], -1)
        combined_cons = tf.concat(
            [fake_cons_and_labels, real_cons_and_labels], axis=0
        )

        # `0` for True, `1` for False
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_cons)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                        self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, real_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_cons = self.generator(random_vector_labels)
            fake_cons_and_labels = tf.concat([fake_cons, real_labels], -1)
            predictions = self.discriminator(fake_cons_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        # Monitor accuracy
        self.gen_mae_tracker.update_state(fake_cons, real_cons)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "g_mae": self.gen_mae_tracker.result(),
        }

def train_cgan(batch_size=4, latent_dim=4, epochs=500,
               dnodes=[7, 6, 5, 4],
               gnodes=[2, 3, 4, 6],
               dataset_fname = 'average_delta_G.csv'):

    generator_dir = 'generator.keras'
    discriminator_dir = 'discriminator.keras'
    train_log_file = 'epoch_g_loss_d_loss_g_mae.txt'
    gen_con_fname = 'generated_cons.txt'


    data = pd.read_csv(dataset_fname)

    x_train = data[['Ni', 'Co', 'Fe', 'Pd', 'Pt']]
    y_train = data['mean_d_G']

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    generator_in_channels = latent_dim + 1
    # discriminator_in_channels = 6

    # Create the discriminator.
    d_sequential_layers = []
    d_sequential_layers.append(keras.layers.InputLayer((6,)))
    for dn in dnodes:
        d_sequential_layers.append(layers.Dense(dn))
        d_sequential_layers.append(layers.LeakyReLU(negative_slope=0.2))
            # d_sequential_layers.append(layers.Softmax())
    d_sequential_layers.append(layers.Dense(1))

    discriminator = keras.Sequential(d_sequential_layers, name="discriminator",)

    # Create the generator.
    g_sequential_layers = []
    g_sequential_layers.append(keras.layers.InputLayer((generator_in_channels,)))
    for gn in gnodes:
        g_sequential_layers.append(layers.Dense(gn))
        d_sequential_layers.append(layers.LeakyReLU(negative_slope=0.2))
            # d_sequential_layers.append(layers.Softmax())
    g_sequential_layers.append(layers.Dense(5, activation='softmax'))
    g_sequential_layers.append(ConNormalization())

    generator = keras.Sequential(g_sequential_layers, name="generator",)

    # Create GAN model
    cond_gan = ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )

    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        loss_fn=keras.losses.MeanSquaredError(),
    )

    history = cond_gan.fit(dataset, epochs=epochs, verbose=0)
    # cond_gan.generator.summary()
    # cond_gan.discriminator.summary()
    g_loss = history.history["g_loss"]
    d_loss = history.history["d_loss"]
    g_mae = history.history["g_mae"]

    np.savetxt(train_log_file, np.array([history.epoch, g_loss,  d_loss, g_mae]).T,
               fmt='%.8f')

    # Use the model
    # Random latent space.
    random_latent = tf.random.normal(shape=(len(x_train), latent_dim))
    input_labels = tf.constant(y_train, dtype=tf.float32)
    input_labels = tf.reshape(input_labels, (-1,1))
    generator_inputs = tf.concat([random_latent, input_labels], axis=1)

    generated_cons = cond_gan.generator(generator_inputs).numpy()

    mae = tf.keras.losses.MeanAbsoluteError()
    x_train = tf.cast(x_train, dtype=tf.float32)
    final_mae = mae(generated_cons, x_train).numpy()

    # print('Final MAE:', final_mae)
    # Save the model
    cond_gan.save_model(generator_dir=generator_dir,
                        discriminator_dir=discriminator_dir)

    # generating concentrations
    generated_cons_with_zero = gan_predict(cond_gan, number_of_outputs=200,
                                           latent_dim=latent_dim).numpy()
    np.savetxt(gen_con_fname, generated_cons_with_zero, fmt='%.8f')

    return -final_mae

def train_cgan_opt(batch_size=4, # latent_dim=4,
               # dnode0=10,dnode1=9,dnode2=8,dnode3=7,dnode4=6,dnode5=5,dnode6=4,
               # gnode0=2,gnode1=3,gnode2=4,gnode3=5,gnode4=6,gnode5=7,gnode6=6,
                dnode3=7,dnode4=6,dnode5=5,dnode6=4,
                gnode0=2,gnode1=3,gnode2=4,gnode6=6,
               ):
    # compile settings
    batch_size = round(batch_size)
    latent_dim = 5

    if os.path.exists('active_learning.json'):
        config_fname = 'active_learning.json'
        sub_dir = '4_generate_new_compositions'
        dataset_sub_dir = '3_high_throughput_prediction'
    elif os.path.exists('cgan_config.json'):
        config_fname = 'cgan_config.json'
        sub_dir = '.'
        dataset_sub_dir = '.'
    else:
        config_fname = {'cgan_Bayesian_step': 0,
                        'working_dir': '.'}
        sub_dir = '.'
        dataset_sub_dir = '.'

    # prepare files and directories.
    cgan_config = config_parser(config_fname)
    if isinstance(config_fname, dict):
        config_fname = 'cgan_config.json'

    if not cgan_config.__contains__('cgan_Bayesian_step'):
        cgan_config['cgan_Bayesian_step'] = 0
    cgan_config['cgan_Bayesian_step'] = str(cgan_config['cgan_Bayesian_step'])

    if not os.path.exists(os.path.join(cgan_config['working_dir'],
                                       sub_dir,
                                       cgan_config['cgan_Bayesian_step'])):
        os.mkdir(os.path.join(cgan_config['working_dir'],
                              sub_dir,
                              cgan_config['cgan_Bayesian_step']))

    generator_dir = os.path.join(cgan_config['working_dir'],
                                 sub_dir,
                                 cgan_config['cgan_Bayesian_step'],
                                 'generator.keras')
    discriminator_dir = os.path.join(cgan_config['working_dir'],
                                     sub_dir,
                                     cgan_config['cgan_Bayesian_step'],
                                     'discriminator.keras')
    train_log_file = os.path.join(cgan_config['working_dir'],
                                  sub_dir,
                                  cgan_config['cgan_Bayesian_step'],
                                  'epoch_g_loss_d_loss_g_mae.txt')
    gen_con_fname = os.path.join(cgan_config['working_dir'],
                                 sub_dir,
                                 cgan_config['cgan_Bayesian_step'],
                                 'generated_cons.txt')
    dataset_fname = os.path.join(cgan_config['working_dir'],
                                 dataset_sub_dir,
                                 'average_delta_G.csv')

    data = pd.read_csv(dataset_fname)

    x_train = data[['Ni', 'Co', 'Fe', 'Pd', 'Pt']]
    y_train = data['mean_d_G']

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    generator_in_channels = latent_dim + 1
    # discriminator_in_channels = 6

    # Create the discriminator.
    d_sequential_layers = []
    d_sequential_layers.append(keras.layers.InputLayer((6,)))
    for dn in [dnode3,dnode4,dnode5,dnode6]:
        if round(dn) != 0:
            d_sequential_layers.append(layers.Dense(round(dn)))
            d_sequential_layers.append(layers.LeakyReLU(negative_slope=0.2))
            # d_sequential_layers.append(layers.Softmax())
    d_sequential_layers.append(layers.Dense(1))

    discriminator = keras.Sequential(d_sequential_layers, name="discriminator",)

    # Create the generator.
    g_sequential_layers = []
    g_sequential_layers.append(keras.layers.InputLayer((generator_in_channels,)))
    for gn in [gnode0,gnode1,gnode2,gnode6]:
        if round(gn) != 0:
            g_sequential_layers.append(layers.Dense(round(gn)))
            d_sequential_layers.append(layers.LeakyReLU(negative_slope=0.2))
            # d_sequential_layers.append(layers.Softmax())
    g_sequential_layers.append(layers.Dense(5,activation='softmax'))
    g_sequential_layers.append(ConNormalization())

    generator = keras.Sequential(g_sequential_layers, name="generator",)

    # Create GAN model
    cond_gan = ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )

    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        # loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        loss_fn=keras.losses.MeanSquaredError(),
    )

    history = cond_gan.fit(dataset, epochs=500,verbose=0)
    # cond_gan.generator.summary()
    # cond_gan.discriminator.summary()
    g_loss = history.history["g_loss"]
    d_loss = history.history["d_loss"]
    g_mae = history.history["g_mae"]

    np.savetxt(train_log_file, np.array([history.epoch, g_loss,  d_loss, g_mae]).T,
               fmt='%.8f')

    # Use the model
    # Random latent space.
    random_latent = tf.random.normal(shape=(len(x_train), latent_dim))
    input_labels = tf.constant(y_train, dtype=tf.float32)
    input_labels = tf.reshape(input_labels, (-1,1))
    generator_inputs = tf.concat([random_latent, input_labels], axis=1)

    generated_cons = cond_gan.generator(generator_inputs).numpy()

    mae = tf.keras.losses.MeanAbsoluteError()
    x_train = tf.cast(x_train, dtype=tf.float32)
    final_mae = mae(generated_cons, x_train).numpy()

    # print('Final MAE:', final_mae)
    # Save the model
    cond_gan.save_model(generator_dir=generator_dir,
                        discriminator_dir=discriminator_dir)

    # generating concentrations
    generated_cons_with_zero = gan_predict(cond_gan, number_of_outputs=200,
                                           latent_dim=latent_dim).numpy()
    np.savetxt(gen_con_fname, generated_cons_with_zero, fmt='%.8f')

    with open(os.path.join(cgan_config['working_dir'],
                           sub_dir,
                           'active_results.txt'), 'a+') as f:
        np.savetxt(f, [[int(cgan_config['cgan_Bayesian_step']), final_mae]],
                   fmt='%.8f')
    cgan_config['cgan_Bayesian_step'] = str(int(cgan_config['cgan_Bayesian_step']) + 1)
    with open(config_fname, 'w') as f:
        json.dump(cgan_config, f, indent=4)

    return -final_mae

def gan_predict(model, condition=0.0, number_of_outputs=20, latent_dim=2): # `model` here is the CGAN model.
    random_latent = tf.random.normal(shape=(number_of_outputs, latent_dim))
    input_labels = tf.fill([number_of_outputs, 1], condition)
    generator_inputs = tf.concat([random_latent, input_labels], axis=1)
    generated_cons = model.generator(generator_inputs)
    return generated_cons

def load_model(generator_dir='generator.keras',
                discriminator_dir='discriminator.keras'):
    pass

def param_opt():
    # batch_size=4,latent_dim=4,
    # dnode0=10,dnode1=9,dnode2=8,dnode3=7,dnode4=6,dnode5=5,dnode6=4,
    # gnode0=2,gnode1=3,gnode2=4,gnode3=5,gnode4=6,gnode5=7,gnode6=6,

    pbounds = {
    'batch_size': (2,8),
    # 'latent_dim': (2,8),
    # 'dnode0': (10,16),
    # 'dnode1': (9,15),
    # 'dnode2': (8,14),
    'dnode3': (7,13),
    'dnode4': (6,12),
    'dnode5': (4,10),
    'dnode6': (2,8),

    'gnode0': (2,6),
    'gnode1': (3,7),
    'gnode2': (4,8),
    # 'gnode3': (5,9),
    # 'gnode4': (6,10),
    # 'gnode5': (7,11),
    'gnode6': (8,12),
    }

    optimizer = BayesianOptimization(
                f=train_cgan_opt,
                # constraint=constraint,
                pbounds=pbounds,
                verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=1,
                )

    optimizer.maximize(
        init_points=3,
        n_iter=50,
        )

    # print(optimizer.max)
    return optimizer.max

if __name__ == '__main__':
    import json
    opt_param = param_opt() # optimize parameters
    with open('opt_param.json', 'w') as fp:
        json.dump(opt_param, fp)
    # opt_param = {'batch_size': 4.716340873267722,
    #  'dnode0': 14.890617834616368,
    #  'dnode1': 13.991083800561789,
    #  'dnode2': 8.452657816243654,
    #  'dnode3': 7.143250516834212,
    #  'dnode4': 8.827845898775012,
    #  'dnode5': 4.515329924055064,
    #  'dnode6': 2.635688220362549,
    #  'gnode0': 5.765707453513908,
    #  'gnode1': 4.490514239930305,
    #  'gnode2': 4.123541572069811,
    #  'gnode3': 7.306853990206889,
    #  'gnode4': 9.133803455084475,
    #  'gnode5': 10.656469912459617,
    #  'gnode6': 11.675494861654641,
    #  'latent_dim': 2.9385947829856827}
    minus_final_mae = train_cgan(**opt_param['params'])
    print(minus_final_mae)

    # plot
    from matplotlib import pyplot  as plt
    import ternary

    x_data = np.loadtxt('generated_cons.txt')
    x_all_plot = [[x[0]+x[1]+x[2], x[3], x[4]] for x in x_data]

    fig,ax = plt.subplots()
    scale = 1.0
    fontsize = 12
    offset = 0.14
    figure, tax = ternary.figure(scale=scale,ax=ax)
    tax.gridlines(multiple=5, color="blue")
    figure.set_size_inches(6, 6)
    tax.right_corner_label("NiCoFe", fontsize=fontsize)
    tax.top_corner_label("Pd", fontsize=fontsize)
    tax.left_corner_label("Pt", fontsize=fontsize)
    tax.left_axis_label("Pt", fontsize=fontsize, offset=offset)
    tax.right_axis_label("Pd", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("NiCoFe", fontsize=fontsize, offset=offset)
    tax.scatter(x_all_plot, marker='s', color='red', label="Initial pred")
    # Remove default Matplotlib Axes
    # tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.boundary()
    # tax.set_title("RGBA Heatmap")
    # plt.show()
    plt.savefig('ternary_cons.png')
