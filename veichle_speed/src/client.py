import data_handler as dh
from NeuralNet import RegressionNet as net

import torch
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import TensorDataset

import nvflare.client as fl


#model traing parameters and device set
DEVICE = "mps" if torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32 
learning_rate = 1e-3
epochs = 5 

def _train_loop(dataloader, model, optimizer, loss_fn):
      size = len(dataloader.dataset)
      #setting training mode for model
      model.train()

      for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(X, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                  loss, current = loss.item(), batch*batch_size + len(X)
                  print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def _test_loop(dataloader, model, loss_fn):
      #setting evaluation mode for model
      model.eval()
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      test_loss, correct = 0, 0

      with torch.no_grad():
            for X, y in dataloader:
                  pred = model(X)
                  test_loss += loss_fn(pred, y).item()
                  correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def client():
     """
     Client function to load data and perform operations.
     
     Returns:
     None
     """
     file_path = "veichle_speed/data/METRO966_averageSpeed_desc.csv"
     result = dh.load_data(file_path)
     
     if result is None:
           print("Failed to load data.")
           return
     
     X, y = result
     print("Data loaded successfully.")

     # Convert to PyTorch tensors
     X_tensor = torch.tensor(X, dtype=torch.float32)
     
     y_tensor = torch.tensor(y, dtype=torch.float32)
     
     print(f"Tensors, loaded succesfully: {X_tensor.shape}, {y_tensor.shape}")

     #taking input and output sizes
     input_size = X_tensor.shape[1]
     output_size = len(y[0])

     #model initialization
     model = net(input_size, 24)

     #flare initialize
     fl.init()

     #data loading 
     train_dataset = TensorDataset(X_tensor, y_tensor)
     train_dataloader = Dataloader(train_dataset, batch_size, shuffle=True)
     

     loss_fn = torch.nn.MSELoss()
     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




client()  # Call the client function to execute the data loading and processing