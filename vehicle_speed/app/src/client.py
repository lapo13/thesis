import vehicle_speed.app.custom.utils.data_handler as dh

from vehicle_speed.app.custom.models.NeuralNet import RegressionNet as net
from vehicle_speed.app.custom.utils.parser import parse_arguments as parse
from vehicle_speed.app.custom.utils.server import load_data_from_url

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import nvflare.client as fl
from nvflare.client.tracking import SummaryWriter

def client():
     """
     Client function to load data and perform operations.
     
     Returns:
     None
     """
     
     args = parse()

     print(args)

     data = load_data_from_url(args.data_url)
     #print(f"recived: {data} from url: {args.data_url}")

     result = dh.load_data(data)
     if result is None:
           print("Failed to load data.")
           return
     print("Data loaded successfully.")
     
     #create different tensors for the 2 clients with low common values from the same dataset
     X_train, X_test, y_train, y_test = dh.create_highly_differentiated_data(result, args.client_id)
     
     print(f"Tensors, loaded succesfully: {X_train.shape}, {len(y_train[0])}, {X_test.shape}, {len(y_test[0])}")

     #taking input and output sizes
     input_size = X_train.shape[1]
     output_size = len(y_train[0])


     #model initialization
     model = net(input_size, output_size)

     #flare initialize
     fl.init()

     #data loading 
     train_dataset = TensorDataset(X_train, y_train)
     train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
     test_dataset = TensorDataset(X_test, y_test)
     test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True)
     
     #summary = SummaryWriter()
     while fl.is_running():
           recived_model = fl.receive()

           #taking care of NULL model in the first iteration and logging number of rounds
           if recived_model == None:
                 return
           if recived_model.current_round is not None:
                 print(f"Round di addestramento federato N: {recived_model.current_round}")

           #setting up the local model based on recived model, function loss and optimization criteria 
           model.load_state_dict(recived_model.params)
           model.to(args.device)
           loss_fn = torch.nn.MSELoss()
           optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

           steps = args.epochs * len(train_dataloader)
           for epoch in range(args.epochs):
                 model.train_loop(train_dataloader, optimizer, loss_fn, args.device, args.batch_size)
                 accuracy, avgloss = model.test_loop(test_dataloader, loss_fn, args.device)
            
           output_model = fl.FLModel(
                 params=model.cpu().state_dict(),
                 metrics={"accuracy": accuracy}, # type: ignore
                 meta={"NUM_STEPS_CURRENT_ROUND": steps}
           )

           fl.send(output_model)

client()
