# dataloader/preprocessors.py

import numpy as np
import random

def default_preprocess(sample):
    """
    A default preprocessing function that returns the sample unchanged.

    This function serves as a placeholder or a no-op preprocessing step.

    Args:
        sample (Any): The sample to be preprocessed.

    Returns:
        Any: The sample unchanged.
    """
    return sample

def normalize(sample):
    """
    Normalizes the sample by dividing it by 255.0.

    This function is typically used to normalize image data to a range of [0, 1].

    Args:
        sample (numpy.ndarray): The sample to be normalized.

    Returns:
        numpy.ndarray: The normalized sample.
    """
    return (sample / 255.0)

def augment_image(image):
    """
    Augments an image by applying random transformations.

    The transformations include random rotation and horizontal flip.

    Args:
        image (numpy.ndarray): The image to be augmented.

    Returns:
        numpy.ndarray: The augmented image.
    """    
    # Example: Random rotation and flip
    if random.choice([True, False]):
        image = np.rot90(image)
    if random.choice([True, False]):
        image = np.fliplr(image)
    return image

def augment_text(text):
    """
    Augments a text sample by swapping two random words.

    This function is an example of text augmentation where words in the text are randomly swapped.

    Args:
        text (str): The text sample to be augmented.

    Returns:
        str: The augmented text sample.
    """
    # Example: Swap words
    words = text.split()
    if len(words) > 1:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

