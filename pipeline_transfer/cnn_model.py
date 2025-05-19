import torch
import torch.nn.functional as F
class Net(torch.nn.Module):
    def __init__(self, extra_predictors_dim=60):  # datum_diff_dim is now 128
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=60, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(p=0.05)  # Dropout with a probability of 0.1
        # Compute the size of the input to the first fully connected layer
        self.fc1 = torch.nn.Linear(32 * 2 * 2 + extra_predictors_dim, 128)  # + datum_diff_dim for the datum_diff feature
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 1)  # Assuming single target output; adjust if multiple targets

    def forward(self, x, extra_predictors):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Ensure datum_diff is already of shape (batch_size, 128)
        # Concatenate datum_diff to the flattened output along the feature dimension
        x = torch.cat((x, extra_predictors), dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x

class Net_5x5(torch.nn.Module):
    def __init__(self, extra_predictors_dim=63):
        super(Net_5x5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=60, out_channels=64, kernel_size=3, stride=1, padding=1)  # (5, 5)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  # (5, 5)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        self.dropout = torch.nn.Dropout(p=0.05)  # Dropout layer

        # After two conv layers and pooling: output size will be (32, 1, 1)
        self.fc1 = torch.nn.Linear(32 * 1 * 1 + extra_predictors_dim, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 1)  # Output layer

    def forward(self, x, extra_predictors):
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 64, 5, 5) -> (batch_size, 64, 2, 2)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 32, 2, 2) -> (batch_size, 32, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 32 * 1 * 1)
        x = torch.cat((x, extra_predictors), dim=1)  # Concatenate with extra predictors
        x = self.dropout(F.relu(self.fc1(x)))  # (batch_size, 128)
        x = F.relu(self.fc2(x))  # (batch_size, 32)
        x = self.fc3(x).squeeze(1)  # (batch_size, 1)
        return x


class Net_3x3(torch.nn.Module):
    def __init__(self, extra_predictors_dim=63):
        super(Net_3x3, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=60, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(p=0.05)
        # Fully connected layer for input (3, 3, 60) -> (1, 1, 32) after conv + pooling
        self.fc1 = torch.nn.Linear(32 * 2 * 2 + extra_predictors_dim, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x, extra_predictors):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten to (batch_size, feature_size)
        x = torch.cat((x, extra_predictors), dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x
    