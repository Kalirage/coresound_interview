import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerIdentificationModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(SpeakerIdentificationModel, self).__init__()
        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, voice_emb, face1_emb, face2_emb):
        # Concatenate the embeddings
        x = torch.cat((voice_emb, face1_emb, face2_emb), dim=1)
        # Pass through the fully connected layers
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        # Output layer
        x = self.output(x)
        # Use sigmoid for binary classification
        x = torch.sigmoid(x)
        return x