import numpy as np
import json

from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from app.custom.utils.data_handler import DataHandler
from app.custom.models.NeuralNet import TFModel as Net


def load_config(config_path: str = "/Users/lapotinacci/thesis/vehicle_speed/app/config") -> dict:
    """
    Load configuration from JSON file based on site name.
    
    Args:
        config_path: Path to configuration directory (default: "config/")
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If site name is not recognized
    """
    # Ottieni informazioni sul sito corrente da NVFLARE
    sys_info = flare.system_info()
    site_name = sys_info["site_name"]
    
    # Costruisci il percorso assoluto relativo allo script
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    
    config_file = None
    
    # Seleziona il file di configurazione in base al nome del sito
    match site_name:
        case "site-1": 
            config_file = config_path + "/arpat_config.json"
            print(f"Loading configuration for site-1, path: {config_file}")
        case "site-2":
            config_file = config_path + "/metro_config.json"
            print(f"Loading configuration for site-2, path: {config_file}")
        case "site-3":
            config_file = config_path + "/tos_config.json"
            print(f"Loading configuration for site-3, path: {config_file}")
        case _:  # Caso default se il sito non è riconosciuto
            raise ValueError(f"Unknown site name: {site_name}")
    
    # Verifica che il file esista
    if config_file is None:
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not Exists: {config_file}")
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Carica e restituisci la configurazione
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Successfully loaded config from {config_file}")
    return config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcola tutte le metriche di valutazione.
    
    Args:
        y_true: Valori veri
        y_pred: Valori predetti
        
    Returns:
        dict: Dizionario con tutte le metriche
    """
    # RMSE
    rmse = root_mean_squared_error(y_true, y_pred, multioutput='uniform_average')
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
    
    # R-squared
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    
    # MAPE (evita divisione per zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else 0.0
    
    # Explained Variance
    explained_var = float(1 - np.var(y_true - y_pred) / np.var(y_true)) if np.var(y_true) != 0 else 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'explained_variance': explained_var
    }


def client():
    """Funzione principale del client per NVFlare con configurazione JSON."""
    try:
        
        flare.init()
        writer = SummaryWriter()

        config = load_config()

        if not config:
            raise ValueError("Configuration could not be loaded.")
        
        # Extract configuration
        data_path = config['data_path']
        file_pattern_var = config.get('file_pattern_var', 'data_with_embeddings')
        columns = config['columns']
        categorical_vars = config.get('categorical_vars', [])
        target_column = config.get('target_column', 'TTT')
        output_dim = config['output_dim']
        
        # Initialize data handler
        data_handler = DataHandler(
            data_path=data_path
        )
        
        # Load and preprocess data
        data_handler.load_data(
            columns=columns,
            categorical_vars=categorical_vars,
            file_pattern_var=file_pattern_var,
            target_column=target_column
        )
        
        # Get training data
        X_train, y_train = data_handler.get_train_data()
        X_val, y_val = data_handler.get_test_data()

        # Initialize model
        input_dim = X_train.shape[1]
        model = Net()
        model.build(input_dim=input_dim, output_dim=output_dim)
        if model is None:
            raise ValueError("Model building failed.")
        
        # training steps
        batch_size = config.get('batch_size', 32)
        num_steps = len(X_train) // batch_size
    
        # Initialize client side FL
        while flare.is_running():
            # Receive global model
            input_model = flare.receive()
           
            if input_model is None or input_model.params is None:
                    # First round: use local initialization
                    current_round = 0
                    print(f"Round {current_round} - Using local model initialization")
            else:
                # Subsequent rounds: update weights from global model
                current_round = input_model.current_round
                print(f"Round {current_round} - Received global model, updating weights")
                    
                for layer_name, weights in input_model.params.items():
                    layer = model.get_layer(layer_name)
                    if layer is None:
                        raise ValueError(f"Layer {layer_name} not found in the model.")
                    layer.set_weights(weights)

            # Log system info
            sys_info = flare.system_info()
            print(f"system info is: {sys_info}")
            
            #valuating the model before training
            if current_round is not 0:
                y_pred_initial = model.predict(X_val)
                initial_metrics = calculate_metrics(y_val, y_pred_initial)
                print(f"Round {current_round} - Initial Validation Metrics: {initial_metrics}") 
                for metric_name, metric_value in initial_metrics.items():
                    writer.add_scalar(f'validation/initial_{metric_name}', metric_value, current_round)

            # Train local model
            model.train(
                X_train=X_train,
                y_train=y_train,
                epochs=config.get('epochs', 50),
                learning_rate=config.get('learning_rate', 1e-3),
                batch_size=config.get('batch_size', 32)
            )

            # Evaluate model after training
            y_pred = model.predict(X_val)
            final_metrics = calculate_metrics(y_val, y_pred)
            print(f"Final Validation Metrics: {final_metrics}")
            for metric_name, metric_value in final_metrics.items():
                writer.add_scalar(f'validation/final_{metric_name}', metric_value, current_round)
            
            # System info after training
            sys_info = flare.system_info()
            print(f"system info is: {sys_info}", flush=True)
            print(f"finished round: {current_round}", flush=True)

            #new model after training
            output_model = flare.FLModel(
                params={layer.name: layer.get_weights() for layer in (model.model).layers},  # Estrae tutti i pesi # type: ignore
                params_type="FULL",  
                metrics=final_metrics,  # Include metriche di performance
                current_round=current_round,  # Mantiene il numero del round corrente
                meta ={
                    "NUM_STEPS_CURRENT_ROUND": num_steps,
                    "Aggregation_weight": len(X_train),
                }  # Metadati aggiuntivi necessari come parametri dell'algoritmo di averaging 
            )

            # Invia il modello aggiornato al server per l'aggregazione con altri client
            flare.send(output_model)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
            



if __name__ == "__main__":
    client()