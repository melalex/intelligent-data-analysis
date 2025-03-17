import numpy as np


def guassian_noise(image):
    r, c = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, (r, c))
    gaussian = gaussian.reshape(r, c)
    noisy = image + gaussian
    return noisy


def salt_and_pepper_noise(image):
    ratio = 0.9
    amount = 0.1
    noisy = np.copy(image)

    salt_count = np.ceil(amount * image.size * ratio)
    coords = [np.random.randint(0, i - 1, int(salt_count)) for i in image.shape]
    noisy[coords] = 1
    pepper_count = np.ceil(amount * image.size * (1.0 - ratio))
    coords = [np.random.randint(0, i - 1, int(pepper_count)) for i in image.shape]
    noisy[coords] = 0
    return noisy


def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy


def speckle_noise(image):
    r, c = image.shape
    speckle = np.random.randn(r, c)
    speckle = speckle.reshape(r, c)
    noisy = image + image * speckle
    return noisy


def add_noise(image):
    p = np.random.random()
    if p <= 0.33:
        noisy = guassian_noise(image)
    elif p <= 0.66:
        noisy = poisson_noise(image)
    else:
        noisy = speckle_noise(image)

    return noisy
