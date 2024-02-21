import numpy as np

def augment_voice_embedding(voice_embedding, noise_level=0.02, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    """
    Apply augmentation to voice embeddings by adding noise, scaling, and shifting.
    
    Parameters:
    - voice_embedding: numpy array, the original voice embedding.
    - noise_level: float, standard deviation of Gaussian noise to add.
    - scale_range: tuple, the range of scaling factors.
    - shift_range: tuple, the range of values to shift the embedding.
    
    Returns:
    - Augmented voice embedding.
    """
    # Add random noise
    noise = np.random.normal(0, noise_level, voice_embedding.shape)
    augmented_embedding = voice_embedding + noise
    
    # Apply scaling
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    augmented_embedding *= scale_factor
    
    # Apply shifting
    shift_value = np.random.uniform(shift_range[0], shift_range[1])
    augmented_embedding += shift_value
    
    return augmented_embedding

import numpy as np

def augment_face_embedding(face_embedding, noise_level=0.02, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
    """
    Apply augmentation to face embeddings by adding noise, scaling, and shifting.
    
    Parameters:
    - face_embedding: numpy array, the original face embedding.
    - noise_level: float, standard deviation of Gaussian noise to add.
    - scale_range: tuple, the range of scaling factors.
    - shift_range: tuple, the range of values to shift the embedding.
    
    Returns:
    - Augmented face embedding.
    """
    # Add random noise
    noise = np.random.normal(0, noise_level, face_embedding.shape)
    augmented_embedding = face_embedding + noise
    
    # Apply scaling
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    augmented_embedding *= scale_factor
    
    # Apply shifting
    shift_value = np.random.uniform(shift_range[0], shift_range[1])
    augmented_embedding += shift_value
    
    return augmented_embedding
