import veichle_speed.app.custom.utils.data_handler as dh
import os 

from veichle_speed.app.custom.models.NeuralNet import RegressionNet as net
from veichle_speed.app.custom.utils.parser import parse_arguments as parse

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import nvflare.client as fl
from nvflare.client.tracking import SummaryWriter



def train_loop(dataloader, model, optimizer, loss_fn, device, batch_size, summary_writer = None):
      size = len(dataloader.dataset)
      #setting training mode for model
      model.train()

      for batch, data in enumerate(dataloader):
            X, y = data[0].to(device), data[1].to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 20 == 0:
                  loss, current = loss.item(), batch*batch_size + len(X)
                  print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

      print("Finished local training")

def test_loop(dataloader, model, loss_fn):
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
      return 100*correct, test_loss


def client():
     """
     Client function to load data and perform operations.
     
     Returns:
     None
     """
     
     args = parse()

     print(f"[DEBUG] Current working dir: {os.getcwd()}")
     print(f"[DEBUG] Data file path received: {args.data_file}")
     print(f"[DEBUG] File exists? {os.path.exists(args.data_file)}")

     result = dh.load_data(args.data_file)
     
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
     model = net(input_size, output_size)

     #flare initialize
     fl.init()
     #data loading 
     train_dataset = TensorDataset(X_tensor, y_tensor)
     train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
     
     #summary = SummaryWriter()
     while fl.is_running():
           recived_model = fl.receive()

           #taking care of NULL model in the first iteration and logging number of rounds
           if recived_model == None:
                 recived_model = model(input_size, output_size)
                 print("Round di addestramento federato Iniziale")
           if recived_model.current_round is not None:
                 print(f"Round di addestramento federato N: {recived_model.current_round}")

           #setting up the local model based on recived model, function loss and optimization criteria 
           model.load_state_dict(recived_model.params)
           model.to(args.device)
           loss_fn = torch.nn.MSELoss()
           optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

           steps = args.epochs * len(train_dataloader)
           for epoch in range(args.epochs):
                 train_loop(train_dataloader, model, optimizer, loss_fn, args.device, args.batch_size)
            
           output_model = fl.FLModel(
                 params=model.cpu().state_dict(),
                 #metrics={"accuracy": accuracy},
                 meta={"NUM_STEPS_CURRENT_ROUND": steps}
           )

           fl.send(output_model)

client()
