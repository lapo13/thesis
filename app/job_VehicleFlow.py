from pathlib import Path
from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv
import pandas as pd
import pickle, ast
#import os

if __name__ == "__main__":
    # Configurazione dataset
    dataset_path = "/Users/lapotinacci/thesis/metric_datasets/vehicleFlow_dataset.xlsx"
    df = pd.read_excel(dataset_path)
    df['TTT'] = df['TTT'].apply(ast.literal_eval)
    df = df[df["TTT"].apply(lambda x: len(x) == 24)].sample(frac=1).reset_index(drop=True)
    df = df[df["type_of_TTT"] == "daily"]
    df.drop(columns=["type_of_TTT"], inplace=True)

    # Filtra le righe con address
    df_filtered = df[df["serviceUri"].notna()]
    sensors = df_filtered["serviceUri"].unique()
    print(f"Sensori trovati: {sensors}")
    n_clients = len(sensors)
    num_rounds = 15

    # Cartella dove salvare i dataset dei client
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)

    # Salva i sotto-dataset per ogni client
    for i, sensor in enumerate(sensors):
        data = sensors[i]
        with open(dataset_dir / f"site-{i+1}.pkl", "wb") as f:
            pickle.dump(data, f)

    # Path dello script client
    script_dir = Path(__file__).parent
    train_script = str(script_dir / "src" / "client_VehicleFlow.py")

    # Configura la ricetta FedAvg
    recipe = FedAvgRecipe(
        name="tf_fedavg",
        num_rounds=num_rounds,
        initial_model=None,
        min_clients=n_clients,
        train_script=train_script
    )

    # Configura l'ambiente di simulazione
    env = SimEnv(num_clients=n_clients)

    # Avvia il training
    print("\nAvvio training...\n")
    run = recipe.execute(env=env)

    # Stampa risultati
    print()
    print("Result can be found in:", run.get_result())
    print("Job Status is:", run.get_status())
    print()