import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nn import NeuralNetwork

# Custom dataset class
# Creating a small toy dataset
X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
Y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6],
    ]
)
Y_test = torch.Tensor([0, 1])


# Defining a custom Dataset class
class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y

    # Instructions for retrieving exactly one data record and the corresponding label
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, Y_train)
test_ds = ToyDataset(X_test, Y_test)

print(len(train_ds))
# 5

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
)
test_loader = DataLoader(
    dataset=test_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch: {idx+1}:", x, y)

torch.manual_seed(123)
model = NeuralNetwork(
    num_inputs=2,
    num_outputs=2,
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.2f}"
        )

model.eval()
