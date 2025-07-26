import vehicle_speed.app.custom.utils.data_handler as dh

from vehicle_speed.app.custom.models.NeuralNet import RegressionNet as net
from vehicle_speed.app.custom.utils.parser import parse_arguments as parse
from vehicle_speed.app.custom.utils.server import import_data_from_url

import torch

import nvflare.client as fl
from nvflare.client.tracking import SummaryWriter

def client():
     """
     Client function to load data and perform operations.
     
     Returns:
     None
     """
     
     args = parse()
     #print(args)

     data = import_data_from_url(args.data_url)
     #print(f"recived: {data} from url: {args.data_url}")

     result = dh.load_data(data, args.client_id, args.batch_size)
     if result is None:
           print("Failed to load data.")
           return
     print("Data loaded successfully.")
     
     #creating dataloader for client
     train_dataloader, test_dataloader = result

     #model initialization
     model = net()

     #flare initialize
     fl.init()
     
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
