from torch import nn
import torch

class RegressionNet(nn.Module):
     def __init__(self, input_size = 7, output_size = 24):
          super(RegressionNet, self).__init__()
          self.linear_relu = nn.Sequential(
               nn.Linear(input_size, 32),
               nn.ReLU(),
               nn.Linear(32, 64),
               nn.ReLU(),
               nn.Linear(64, output_size)
          )
          self._init_weights()
          
     def _init_weights(self):
          for m in self.modules():
               if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias,0)

     def forward(self, x):
          return self.linear_relu(x)
     
     def train_loop(self, dataloader, optimizer, loss_fn, device, batch_size):
      size = len(dataloader.dataset)
      #setting training mode for model
      self.train()

      for batch, data in enumerate(dataloader):
            X, y = data[0].to(device), data[1].to(device)
            pred = self(X)
            print(pred.shape, y.shape)
            loss = loss_fn(pred, y)


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 20 == 0:
                  loss, current = loss.item(), batch*batch_size + len(X)
                  print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

      print("Finished local training")

     def test_loop(self, dataloader, loss_fn, device, tolerance=5.0):
          self.eval()
          size = len(dataloader.dataset)
          num_batches = len(dataloader)
          total_loss = 0.0
          total_correct = 0

          with torch.no_grad():
               for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = self(X)
                    total_loss += loss_fn(pred, y).item()

                    # Calcolo accuratezza con tolleranza
                    correct_matrix = (torch.abs(pred - y) <= tolerance).type(torch.float)
                    # Un campione è "corretto" se *tutti i 24 valori* lo sono
                    sample_correct = correct_matrix.all(dim=1)
                    total_correct += sample_correct.sum().item()

          avg_loss = total_loss / num_batches
          accuracy = total_correct / size
          print(f"Test Error: \n Accuracy: {100*accuracy:>0.1f}%, Avg loss: {avg_loss:>8f} \n")
          return 100 * accuracy, avg_loss