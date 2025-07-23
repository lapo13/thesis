from app.src.NeuralNet import RegressionNet as Net


from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

n_clients = 2

job = BaseFedJob(
    name="avgSpeed_FedAVG",
    initial_model=Net(),
)

controller = FedAvg(
    num_clients=n_clients,
    num_rounds=2,
)
job.to(controller, "server")

for i in range(n_clients):
    runner = ScriptRunner(
        script="veichle_speed/app/src/client.py", script_args="" 
    )
    job.to(runner, f"site-{i+1}")

job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")