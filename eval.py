import torch
import torch.nn as nn
from dataset_loader import get_loaders, get_test_loader
from model import SpeakerIdentificationModel

from dataset_construction import main as create_triplets
import argparse
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_model(model_path, input_size, hidden_sizes):
    model = SpeakerIdentificationModel(input_size, hidden_sizes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def calculate_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            audio_emb, pos_image_emb, neg_image_emb, labels = [d.to(device) for d in data]
            outputs = model(audio_emb, pos_image_emb, neg_image_emb)
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def eval(model_path, test_loader, input_size, hidden_sizes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained model
    model = load_model(model_path, input_size, hidden_sizes).to(device)
    
    # Calculate accuracy on the test set
    accuracy = calculate_accuracy(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate speaker identification model')
    parser.add_argument('--audio_embedding_path', type=str, default='data/audio_embeddings.pickle',
                        help='Path to the audio embeddings file')
    parser.add_argument('--image_embedding_path', type=str, default='data/image_embeddings.pickle',
                        help='Path to the image embeddings file')
    parser.add_argument('--output_path', type=str, default='data/triplets.pickle',
                        help='Path to save the triplets file')
    parser.add_argument('--model_path', type=str, default='models/model.pth',
                        help='Path to the pre-trained model')
    parser.add_argument('--input_size', type=int, default=1216,
                        help='Input size for the model')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 128, 64],
                        help='Hidden sizes for the model')

    args = parser.parse_args()
    
    print('Creating dataset from embeddings...')
    create_triplets(args.image_embedding_path, args.audio_embedding_path, args.output_path)

    print('Loading dataset...')
    test_loader = get_test_loader(args.output_path, batch_size=32, shuffle=False)
    #test_loader = torch.load('output/test_loader.pth')
    
    print('Evaluating...')
    eval(args.model_path, test_loader, args.input_size, args.hidden_sizes)
