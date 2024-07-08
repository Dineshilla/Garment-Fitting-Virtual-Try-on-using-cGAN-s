# %%
import tensorflow as tf
import numpy as np
import os
import pathlib
from pathlib import Path
import time
import datetime
from IPython.display import clear_output


import matplotlib.pyplot as plt

# %%
def load_image_train(image_file):

    input_image = np.load(image_file)
    input_image = tf.convert_to_tensor(input_image)

    return input_image

# %%
def load_image_test(image_file):

    input_image = np.load(image_file)
    input_image = tf.convert_to_tensor(input_image)
    return input_image

# %% [markdown]
# ### Train Dataset

# %%
PATH = Path(r"E:\Semester-7\Mini Project\Review-2\Data_stage_2")

# The facade training set consist of 400 images
BUFFER_SIZE = 14221
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1

# %%
def load_image_train_wrapper(image_file):
    image_file = image_file.numpy()
    return load_image_train(image_file)

def tf_load_image_train(image_file):
    [full_image,] = tf.py_function(load_image_train_wrapper, [image_file], [tf.float32])
    full_image.set_shape([256, 192, 12])

    input_image = full_image[:, :, :9]  # First 12 channels
    ground_truth = full_image[:, :, 9:] # Last 3 channels

    return input_image, ground_truth

# %%
def load_image_test_wrapper(image_file):
    image_file = image_file.numpy()
    return load_image_test(image_file)

def tf_load_image_test(image_file):
    [full_image,] = tf.py_function(load_image_test_wrapper, [image_file], [tf.float32])
    full_image.set_shape([256, 192, 12])

    input_image = full_image[:, :, :9]  # First 12 channels
    ground_truth = full_image[:, :, 9:] # Last 3 channels

    return input_image, ground_truth

# %%
train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.npy'))
train_dataset = train_dataset.map(tf_load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


# %%
# train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.npy'))
# train_dataset = train_dataset.map(load_image_train,
#                                   num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.batch(BATCH_SIZE)

# %% [markdown]
# ### Test Dataset

# %%
test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.npy'))
test_dataset = test_dataset.map(tf_load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# %%
OUTPUT_CHANNELS = 3

# %%
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

# %%
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# %% [markdown]
# ### Generator

# %%
# def Generator():
#   inputs = tf.keras.layers.Input(shape=[256, 192, 12])

#   down_stack = [
#     downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
#     downsample(128, 4),  # (batch_size, 64, 64, 128)
#     downsample(256, 4),  # (batch_size, 32, 32, 256)
#     downsample(512, 4),  # (batch_size, 16, 16, 512)
#     downsample(512, 4),  # (batch_size, 8, 8, 512)
#     downsample(512, 4),  # (batch_size, 4, 4, 512)
#     downsample(512, 4),  # (batch_size, 2, 2, 512)
#     downsample(512, 4),  # (batch_size, 1, 1, 512)
#   ]

#   up_stack = [
#     upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
#     upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
#     upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
#     upsample(512, 4),  # (batch_size, 16, 16, 1024)
#     upsample(256, 4),  # (batch_size, 32, 32, 512)
#     upsample(128, 4),  # (batch_size, 64, 64, 256)
#     upsample(64, 4),  # (batch_size, 128, 128, 128)
#   ]

#   initializer = tf.random_normal_initializer(0., 0.02)
#   last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
#                                          strides=2,
#                                          padding='same',
#                                          kernel_initializer=initializer,
#                                          activation='tanh')  # (batch_size, 256, 256, 3)

#   x = inputs

#   # Downsampling through the model
#   skips = []
#   for down in down_stack:
#     x = down(x)
#     skips.append(x)

#   skips = reversed(skips[:-1])

#   # Upsampling and establishing the skip connections
#   for up, skip in zip(up_stack, skips):
#     x = up(x)
#     x = tf.keras.layers.Concatenate()([x, skip])

#   x = last(x)

#   return tf.keras.Model(inputs=inputs, outputs=x)

# %%
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    OUTPUT_CHANNELS = 3  # Assuming you want to output RGB images
    inputs = tf.keras.layers.Input(shape=[256, 192, 9])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # 128x96x64
        downsample(128, 4),  # 64x48x128
        downsample(256, 4),  # 32x24x256
        downsample(512, 4),  # 16x12x512
        downsample(512, 4),  # 8x6x512
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # 16x12x512
        upsample(256, 4),  # 32x24x256
        upsample(128, 4),  # 64x48x128
        upsample(64, 4),  # 128x96x64
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # 256x192x3

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# %%
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# %% [markdown]
# ### Generator Loss MAE (L1)

# %%
LAMBDA = 0.5
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# %%
import tensorflow as tf
from tensorflow.keras.applications import VGG19

class Vgg19(tf.keras.Model):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # Load pre-trained VGG19
        vgg_pretrained = VGG19(include_top=False, weights='imagenet')
        if not requires_grad:
            vgg_pretrained.trainable = False
        
        # Define the slices
        self.slice1 = tf.keras.Sequential(vgg_pretrained.layers[:2])
        self.slice2 = tf.keras.Sequential(vgg_pretrained.layers[2:7])
        self.slice3 = tf.keras.Sequential(vgg_pretrained.layers[7:12])
        self.slice4 = tf.keras.Sequential(vgg_pretrained.layers[12:21])
        self.slice5 = tf.keras.Sequential(vgg_pretrained.layers[21:30])

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(tf.keras.losses.Loss):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = tf.keras.losses.MeanAbsoluteError()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

# %%
VGG_LOSS = VGGLoss()

# %%
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  # criterion loss
  context_loss = VGG_LOSS(target, gen_output)

  print(context_loss)

  total_gen_loss = gan_loss + (LAMBDA * l1_loss) + (0.1 * context_loss)

  return total_gen_loss, gan_loss, l1_loss

# %% [markdown]
# ### Trail by changing loss function

# %%
# import tensorflow as tf
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.models import Model
# from tensorflow.keras.losses import BinaryCrossentropy

# def generator_loss(disc_generated_output, y_pred, y_true):
#     """
#     Computes the combined generator loss.
    
#     Parameters:
#         disc_generated_output (tensor): Discriminator output for generated images.
#         y_true (tensor): Ground truth images.
#         y_pred (tensor): Predicted images.
        
#     Returns:
#         tuple: (total generator loss, GAN loss, VGG feature loss, L1 loss)
#     """
    
#     # 1. Adversarial (GAN) loss
#     loss_object = BinaryCrossentropy(from_logits=True)
#     gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
#     # 2. VGG-based perceptual loss
#     vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 192, 3))
#     vgg.trainable = False
#     model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
#     y_true_features = model(y_true)
#     y_pred_features = model(y_pred)
#     vgg_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))
    
#     # 3. L1 loss between ground truth and generated images
#     l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
#     # 4. Total generator loss
#     LAMBDA_L1 = 100  # You can tune this
#     LAMBDA_VGG = 10  # You can tune this
#     total_gen_loss = gan_loss + (LAMBDA_L1 * l1_loss) + (LAMBDA_VGG * vgg_loss)
    
#     return total_gen_loss, gan_loss, vgg_loss #, l1_loss

# # Example usage:
# # total_loss, gan_loss, vgg_loss, l1_loss = combined_generator_loss(disc_output, y_true, y_pred)


# %% [markdown]
# ### Discriminator

# %%
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 192, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 192, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

# %%
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# %% [markdown]
# ### Discriminator Loss

# %%
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

# %%
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# %%
checkpoint_dir = './training_checkpoints_stage_2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# %%
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0][:,:,:3], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

# %% [markdown]
# ### Training Steps

# %%
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# %%
@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image[:,:,:,:3], target], training=True)
    disc_generated_output = discriminator([input_image[:,:,:,:3], gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

# %%
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if int(step) % 1000 == 0:
      clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if int(step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if int(step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

# %%
fit(train_dataset, test_dataset, steps=10000)

# %%
for inp, tar in test_dataset.take(20):
  generate_images(generator, inp, tar)


