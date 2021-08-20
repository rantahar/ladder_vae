import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
import sys

MODEL_PATH = sys.argv[1]
print(MODEL_PATH)
DATA_PATH = '../data/celeba'

IMG_SIZE = 64


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


dataset = dataset_from_folder(DATA_PATH, IMG_SIZE, 6)
images = next(iter(dataset.take(1)))


encoder = tf.keras.models.load_model(MODEL_PATH + '/encoder')
decoder = tf.keras.models.load_model(MODEL_PATH + '/decoder')


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


fig, arr = plt.subplots(3, 4)

means, logvars = encoder(images)
zs = reparameterize(means, logvars)
generated = decoder(zs)

print(generated.shape)

for i in range(6):
    im = (images[i] + 1) / 2
    arr[i % 3, 2*(i//3)].imshow(im)
    im = (generated[i] + 1) / 2
    arr[i % 3, 2*(i//3)+1].imshow(im)

for ax in fig.axes:
    ax.axis("off")

fig.tight_layout()
plt.show()
