import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from augmentation import augment_voice_embedding, augment_face_embedding

class TripletDataset(Dataset):
    def __init__(self, triplet_path='triplets.pickle', indices=None, augmentation=None):
        with open(triplet_path, 'rb') as f:
            triplets = pickle.load(f)
        self.augmentation = augmentation
        self.triplets = [triplets[i] for i in indices] if indices is not None else triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        audio_emb, positive_image_emb, negative_image_emb = self.triplets[idx]
        if self.augmentation is not None:
            # Apply augmentation to embeddings
            audio_emb = augment_voice_embedding(audio_emb)
            positive_image_emb = augment_face_embedding(positive_image_emb)
            negative_image_emb = augment_face_embedding(negative_image_emb)
            
            # Convert back to torch tensors
            audio_emb = torch.from_numpy(audio_emb).float()
            positive_image_emb = torch.from_numpy(positive_image_emb).float()
            negative_image_emb = torch.from_numpy(negative_image_emb).float()
        
        if random.random() < 0.5:
            return (audio_emb, positive_image_emb, negative_image_emb, torch.tensor([1]).float())
        else:
            return (audio_emb, negative_image_emb, positive_image_emb, torch.tensor([0]).float())

def get_test_loader(triplet_path, batch_size=32, shuffle=False, num_workers=0):
    dataset = TripletDataset(triplet_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_loaders(triplet_path, train_split=0.7, val_split=0.15, batch_size=32, shuffle=True, augmentation=None):
    # Load the entire dataset
    with open(triplet_path, 'rb') as f:
        triplets = pickle.load(f)

    total_size = len(triplets)
    indices = list(range(total_size))
    if shuffle:
        np.random.shuffle(indices)

    # Split indices
    train_end = int(train_split * total_size)
    val_end = train_end + int(val_split * total_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create datasets
    train_dataset = TripletDataset(triplet_path, train_indices, augmentation=augmentation)
    val_dataset = TripletDataset(triplet_path, val_indices)
    test_dataset = TripletDataset(triplet_path, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader

# Example usage:
if __name__ == '__main__':
    loader = get_loaders()
    for data in loader:
        audio_emb, first_image_emb, second_image_emb, label = data
        # Your training loop here
