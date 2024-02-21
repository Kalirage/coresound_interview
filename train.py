import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_loaders
from model import SpeakerIdentificationModel
import matplotlib.pyplot as plt
import numpy as np

def calculate_accuracy(outputs, labels):
    predicted = outputs.round().unsqueeze(1)
    correct = (predicted == labels).float().sum()
    return correct / labels.size(0)

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=50, patience=10, output_path='model.pth'):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for data in train_loader:
            audio_emb, pos_image_emb, neg_image_emb, labels = [d.to(device) for d in data]
            outputs = model(audio_emb, pos_image_emb, neg_image_emb).squeeze()
            loss = criterion(outputs, labels.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += calculate_accuracy(outputs, labels).item()
            train_total += 1

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                audio_emb, pos_image_emb, neg_image_emb, labels = [d.to(device) for d in data]
                outputs = model(audio_emb, pos_image_emb, neg_image_emb).squeeze()
                loss = criterion(outputs, labels.squeeze())

                val_loss += loss.item()
                val_correct += calculate_accuracy(outputs, labels).item()
                val_total += 1

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        avg_train_acc = train_correct / train_total
        avg_val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)  # Save the best model
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    test_correct = test_total = 0          
    with torch.no_grad():
        for data in test_loader:
            audio_emb, pos_image_emb, neg_image_emb, labels = [d.to(device) for d in data]
            outputs = model(audio_emb, pos_image_emb, neg_image_emb).squeeze()
            loss = criterion(outputs, labels.squeeze())

            test_correct += calculate_accuracy(outputs, labels).item()
            test_total += 1
            
    test_acc = test_correct / test_total
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    
    # Add test accuracy to the plot
    
    plt.plot(len(history['train_acc'])-1, test_acc, 'ro', label=f'Test Accuracy: {test_acc:.4f}')
    
    plt.title('Accuracy')
    plt.legend()

    plt.show()

    # print accuracies
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Best Validation Loss: {best_val_loss:.4f}')
    
if __name__ == '__main__':
    
    # Hyperparameters
    input_size = 1216
    hidden_sizes = [512, 128, 64]
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 100
    patience=25
    apply_augmentation = False
    
    print('Training model...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    model = SpeakerIdentificationModel(input_size, hidden_sizes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader, test_loader = get_loaders('triplets.pickle', batch_size=batch_size, augmentation=apply_augmentation, train_split=0.8, val_split=0.15, shuffle=True)
    print(model)

    train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs=num_epochs, patience=patience, output_path='best_model.pth')
    # saving test_loader for eval.py
    torch.save(test_loader, 'output/test_loader.pth')