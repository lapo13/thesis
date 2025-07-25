import torch
import os
from vehicle_speed.app.custom.utils.parser import build_script_args
from vehicle_speed.app.custom.utils.server import start_data_server

from vehicle_speed.app.custom.models.NeuralNet import RegressionNet as Net

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner


# #model traing parameters, device set and datapath
DATA_SERVER = "http://localhost:8000/METRO966_averageSpeed_desc.csv"
DEVICE = "mps" if torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10

#job setup
n_clients = 5
n_rounds = 7
httpd = start_data_server(os.path.abspath("vehicle_speed/app/data"), port=8000)


job = BaseFedJob(
    name="avgSpeed_FedAVG",
    initial_model=Net()
)

controller = FedAvg(num_clients=n_clients, num_rounds=n_rounds)
job.to(controller, "server")


for i in range(n_clients):
    script_args = build_script_args(
        data_file=DATA_SERVER,
        client_id=i+1,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE, 
        device = DEVICE
    )
    
    runner = ScriptRunner(
        script=os.path.abspath(
            os.path.join("vehicle_speed", "app", "src", "client.py")),
        script_args=script_args
    )
    job.to(runner, f"site-{i+1}")

if __name__ == "__main__":
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
    
