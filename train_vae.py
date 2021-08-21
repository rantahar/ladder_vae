import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
import time
import sys
import os

learning_rate = 0.001
elbo_ramp = 10
BATCH_SIZE = 64
epochs = 10
save_every = 100
log_every = 10

IMG_SIZE = 64
latent_dim = 64
elbo_weight = 3e-5 # ~ 1.0/#images
min_features = 16
max_features = 256

MODEL_PATH = f'vae_2_{IMG_SIZE}_{min_features}_{max_features}_{latent_dim}_{elbo_weight}'
sample_path = f'samples_2_{IMG_SIZE}_{min_features}_{max_features}_{latent_dim}__{elbo_weight}'
DATA_PATH = '../data/celeba'


def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def flip(image):
    tf.image.random_flip_left_right(image)
    return image


def dataset_from_folder(path, IMG_SIZE, BATCH_SIZE):

    dataset = image_dataset_from_directory(
        path, shuffle=True, batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE)
    )

    dataset = dataset.map(normalize).map(flip)
    return dataset


dataset = dataset_from_folder(DATA_PATH, IMG_SIZE, BATCH_SIZE)
n_batches = tf.data.experimental.cardinality(dataset)
init = RandomNormal(stddev=0.02)


def conv_block(x, size, width=4):
    x = layers.SeparableConv2D(size, width, padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def downscale(x, size):
    shape = tf.shape(x)
    new_shape = shape[1:3] // 2
    x = tf.image.resize(x, new_shape, method="bilinear")
    x = layers.Conv2D(size, 1, padding='same', kernel_initializer=init)(x)
    x = conv_block(x, size)
    return x


def downscale_block(x, size):
    x = layers.SeparableConv2D(size, 4, 2, padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def upscale(x):
    shape = tf.shape(x)
    new_shape = 2 * shape[1:3]
    x = tf.image.resize(x, new_shape, method="bilinear")
    return x


def upscale_block(x, size):
    x = upscale(x)
    x = conv_block(x, size)
    x = conv_block(x, size)
    return x


# At the end of an encoder, flatten and map to a latent space vector
def variational_encoder_head(x, latent_dim):
    x = layers.Flatten()(x)
    mu = layers.Dense(latent_dim)(x)
    log_sigma = layers.Dense(latent_dim)(x)
    return mu, log_sigma


def encoding(input, dimension):
    mean = layers.SeparableConv2D(
            dimension, 1, padding='same', kernel_initializer=init
        )(input)
    log_sigma = layers.SeparableConv2D(
            dimension, 1, padding='same', kernel_initializer=init
        )(input)
    return mean, log_sigma


def to_rgb(x, n_colors=3):
    x = layers.SeparableConv2D(
            n_colors, (4, 4), activation='tanh',
            padding='same', kernel_initializer=init
        )(x)
    return x


def make_encoder(shape, latent_dim):
    input = tf.keras.Input(shape=shape)
    means = []
    logvars = []

    size = shape[1]
    features = min_features
    x = conv_block(input, features)  # From RGB

    while size > 8:
        if features < max_features:
            features *= 2
        size //= 2
        x = downscale_block(x, features)
        mean, logvar = encoding(x, latent_dim)
        means.append(mean)
        logvars.append(logvar)

    x = downscale_block(x, features)

    mean, logvar = variational_encoder_head(x, latent_dim)
    means.append(mean)
    logvars.append(logvar)
    return Model(inputs=input, outputs=[means, logvars])


def make_decoder(latent_dim, shape):
    size = 4
    s = IMG_SIZE//size
    features = min(s*min_features, max_features)
    n_nodes = features * size * size

    z = tf.keras.Input(shape=(latent_dim))
    zs = [z]
    x = layers.Dense(n_nodes)(z)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((size, size, features))(x)

    x = upscale_block(x, features)
    size *= 2
    s //= 2

    while size < IMG_SIZE:
        z = tf.keras.Input(shape=(size, size, latent_dim))
        zs.append(z)
        x_hat = conv_block(z, features, 1)
        x = x + x_hat

        size *= 2
        s //= 2
        features = min(s*min_features, max_features)
        x = upscale_block(x, features)

    zs.reverse()
    image = to_rgb(x)
    return Model(inputs=zs, outputs=image)


if os.path.exists(MODEL_PATH):
    encoder = tf.keras.models.load_model(MODEL_PATH + '/encoder')
    decoder = tf.keras.models.load_model(MODEL_PATH + '/decoder')
    s = int(sys.argv[1])
else:
    encoder = make_encoder((IMG_SIZE, IMG_SIZE, 3), latent_dim)
    decoder = make_decoder(latent_dim, (IMG_SIZE, IMG_SIZE, 3))
    s = 0

start_step = s

encoder.summary()
decoder.summary()


optimizer = tf.keras.optimizers.Adam(lr=learning_rate)


def draw_latent(num):
    size = IMG_SIZE
    zs = []
    while size > 8:
        size //= 2
        z = tf.random.normal([num, size, size, latent_dim])
        zs.append(z)

    z = tf.random.normal([num, latent_dim])
    zs.append(z)
    return zs


test_noise = draw_latent(16)
test_images = next(iter(dataset.take(1)))


def encode(x):
    means, logvars = encoder(x)
    return means, logvars


def reparameterize(means, logvars):
    zs = []
    for mean, logvar in zip(means, logvars):
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(logvar * .5) + mean
        zs.append(z)
    return zs


def reproduction_loss(x, y):
    return tf.math.reduce_mean((x - y)**2, axis=[3])


def calc_elb_loss(zs, means, logvars):
    loss = 0
    for z, mean, logvar in zip(zs, means, logvars):
        losses = logvar - tf.exp(logvar) - mean**2 + 1
        loss += -0.5*(tf.reduce_sum(losses)/BATCH_SIZE)
    return loss


@tf.function
def train(images, batch_size, elbo_factor):
    with tf.GradientTape() as tape:
        means, logvars = encode(images)
        zs = reparameterize(means, logvars)
        decodings = decoder(zs)

        encoding_loss = reproduction_loss(decodings, images)
        encoding_loss = tf.math.reduce_mean(encoding_loss)

        elb_loss = calc_elb_loss(zs, means, logvars)

        loss = encoding_loss + elbo_factor*elb_loss

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return encoding_loss, elb_loss


def save_models(s):
    encoder.save(MODEL_PATH+"/encoder")
    decoder.save(MODEL_PATH+"/decoder")

    means, logvars = encoder(test_images)
    zs = reparameterize(means, logvars)
    generated = decoder(zs)

    fig, arr = plt.subplots(4, 4)

    for i in range(6):
        im = (test_images[i] + 1) / 2
        arr[i % 3, 2*(i//3)].imshow(im)
        im = (generated[i] + 1) / 2
        arr[i % 3, 2*(i//3)+1].imshow(im)

    generated = decoder(test_noise)

    for i in range(4):
        im = (generated[i] + 1) / 2
        arr[3, i].imshow(im)
    for ax in fig.axes:
        ax.axis("off")
    fig.tight_layout()
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    for ax in fig.axes:
        ax.axis("off")

    fig.savefig(f'{sample_path}/sample_{s}.png')
    plt.close(fig)


def ramp(s, regulator=10):
    n = regulator*s/elbo_ramp - regulator
    n = tf.clip_by_value(n, -40, 40)
    w = tf.exp(n) / (1 + tf.exp(n))
    return w

# train the discriminator and decoder
start_time = time.time()
# manually enumerate epochs
for i in range(epochs):
    save_models(s)
    print("saved")

    for element in dataset:
        s += 1
        j = s % n_batches

        this_batch_size = element.shape[0]

        elbo_factor = elbo_weight * ramp(s)
        ef_tensor = tf.convert_to_tensor(elbo_factor, dtype=tf.float32)
        enc_loss, elb_loss = train(element, this_batch_size, ef_tensor)
        time_per_step = (time.time() - start_time)/(s - start_step)

        if s % save_every == save_every-1:
            save_models(s)
            print("saved")

        if s % log_every == log_every-1:
            print(' %d, %d/%d, e=%.5f, elb=%.5f (factor %.3g), time per step=%.3f' %
                  (s, j, n_batches, enc_loss, elb_loss, elbo_factor, time_per_step),
                  flush=True)

print("DONE, saving...")
save_models(s)
print("saved")
