from pathlib import Path
from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

if __name__ == "__main__":
    # Configurazione della simulazione
    n_clients = 3     
    num_rounds = 25
    
    # Get absolute path to client script
    script_dir = Path(__file__).parent
    train_script = str(script_dir / "src" / "client.py")
    
    # FedAvg è l'algoritmo standard di FL: aggrega i pesi dei modelli facendo la media

    recipe = FedAvgRecipe(
        name="tf_fedavg",         
        num_rounds=num_rounds,            
        initial_model = None,       
        min_clients=n_clients,            
        train_script=train_script 
    )
    
    # Aggiunge il tracking dell'esperimento tramite TensorBoard

    add_experiment_tracking(recipe, tracking_type="tensorboard")
    
    # Crea un ambiente di simulazione locale con n_clients client virtuali
    # Simula un ambiente distribuito su una singola macchina per testing/sviluppo
    env = SimEnv(num_clients=n_clients)
    
    # Esegue l'esperimento di federated learning
    # Questa chiamata blocca fino al completamento di tutti i round
    run = recipe.execute(env=env)
    
    # Stampa i risultati dell'esperimento
    print()
    print("Result can be found in :", run.get_result())  # Percorso dove sono salvati i risultati
    print("Job Status is:", run.get_status())            # Status finale (SUCCESS, FAILED, ecc.)
    print()


