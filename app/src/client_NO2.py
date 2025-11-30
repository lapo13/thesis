from pathlib import Path
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
from app.custom.utils.NeuralNet import TFModel as Net
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle, os
from typing import Dict, Any

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average"))
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))*100) if np.any(mask) else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape
    }

ordered_params = [
    # 1. NO2_GR
    {
        "learning_rate": 0.0017, "architecture": [128], "activation": "relu",
        "batch_size": 64, "dropout": [0], "optimizer": "adamw",
        "weight_decay": 0.0158, "early_stop_patience": 100, "epochs": 1000
    },

    # 2. NO2_RO
    {
        "learning_rate": 0.0062, "architecture": [128], "activation": "relu",
        "batch_size": 128, "dropout": [0], "optimizer": "adamw",
        "weight_decay": 0.0397, "early_stop_patience": 200, "epochs": 1000
    },

    # 3. NO2_sett
    {
        "learning_rate": 0.0025, "architecture": [128], "activation": "relu",
        "batch_size": 64, "dropout": [0], "optimizer": "adamw",
        "weight_decay": 0.0437, "early_stop_patience": 350, "epochs": 1000
    },

    # 4. NO2_viar
    {
        "learning_rate": 0.012, "architecture": [128], "activation": "relu",
        "batch_size": 256, "dropout": [0], "optimizer": "adamw",
        "weight_decay": 0.0185, "early_stop_patience": 100, "epochs": 1000
    },

    # 5. NO2_GRam
    {
        "learning_rate": 0.007, "architecture": [128], "activation": "relu",
        "batch_size": 64, "dropout": [0], "optimizer": "adamw",
        "weight_decay": 0.0575, "early_stop_patience": 100, "epochs": 1000
    },

    # 6. NO2_ACR
    {
        "learning_rate": 0.0101, "architecture": [128], "activation": "relu",
        "batch_size": 256, "dropout": [0], "optimizer": "adamw",
        "weight_decay": 0.0107, "early_stop_patience": 100, "epochs": 1000
    },

    # 7. NO2_LAP
    {
        "learning_rate": 0.01, "architecture": [128], "activation": "relu",
        "batch_size": 128, "dropout": [0.0], "optimizer": "adamw",
        "weight_decay": 0.2072, "early_stop_patience": 100, "epochs": 1000
    }
]



def get_site_params(site_url: str) -> Dict[str, Any]:
    """
    Restituisce il dizionario dei parametri associato all'URL fornito.
    
    L'associazione si basa sull'ordine originale dei link forniti.

    Args:
        site_url: L'URL completo del sito, es. 'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_FI-GRAMSCI'.

    Returns:
        Il dizionario dei parametri associato (Dict).

    Raises:
        ValueError: Se l'URL non viene trovato nella lista di riferimento.
    """
    
    # Mappa degli URL di riferimento con i dizionari dei parametri associati.
    # Ho usato un dizionario che mappa l'identificatore finale del sito (es. 'FI-GRAMSCI')
    # al suo dizionario di parametri, per una ricerca più efficiente.

    # Lista degli URL di riferimento nell'ordine corretto
    reference_urls = [
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_GR-MAREMMA',
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_PO-ROMA',
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_FI-SETTIGNANO',
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_LU-VIAREGGIO',
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_FI-GRAMSCI',
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_AR-ACROPOLI',
    'http://www.disit.org/km4city/resource/iot/orionUNIFI/DISIT/ARPAT_QA_LI-LAPIRA'
    ]



    # Dizionari dei parametri (già riordinati secondo il tuo schema)
    try:
        # Troviamo l'indice dell'URL fornito nella lista di riferimento
        index = reference_urls.index(site_url)
        
        # Restituiamo il dizionario dei parametri corrispondente a quell'indice
        return ordered_params[index]
        
    except ValueError:
        # Se site_url non è presente in reference_urls, .index() solleva un ValueError
        raise ValueError(f"URL non trovato. Nessun parametro associato a: '{site_url}'")

get_after_QA = lambda s: s.split("QA_", 1)[1]

def client():
    try:
        flare.init()
        writer = SummaryWriter()

        # Recupera client_id e costruisci il path del dataset
        site_id = flare.system_info()["site_name"]  # es: client_0
        datasets_dir = Path("/Users/lapotinacci/thesis/Federated_Sys/datasets")
        dataset_file = datasets_dir / f"{site_id}.pkl" #in realtà nel dataset file c'è solo il suri
        suri = pickle.load(open(dataset_file, "rb"))
        sensor = get_after_QA(suri)

        path = '/Users/lapotinacci/thesis/Federated_Sys/train_test/'
        files = os.listdir(path)
        files = [f for f in files if f.endswith(f"_{sensor}.pkl")]
        files.sort()

        X_train = pickle.load(open(os.path.join(path, files[1]), "rb"))
        y_train = pickle.load(open(os.path.join(path, files[3]), "rb"))
        X_val = pickle.load(open(os.path.join(path, files[0]), "rb"))
        y_val = pickle.load(open(os.path.join(path, files[2]), "rb"))

        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)


        site_params = get_site_params(suri)
        # Inizializza modello
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        model = Net(params=site_params if isinstance(site_params, dict) else None)
        model.build(input_dim=input_dim, output_dim=output_dim)

        #raise ValueError(f"{site_id} initialized with params: {site_params}")

        batch_size = site_params.get("batch_size", 32) if isinstance(site_params, dict) else 32
        num_steps = len(X_train) / batch_size
        print(f"sire: {site_id}")
        print(X_train.shape, y_train.shape)
        print(X_val.shape, y_val.shape)

        while flare.is_running():
            input_model = flare.receive()
            current_round = 0

            if input_model and input_model.params:
                current_round = input_model.current_round
                # aggiorna i pesi
                for layer_name, weights in input_model.params.items():
                    layer = model.get_layer(layer_name)
                    if layer:
                        layer.set_weights(weights)

                    if current_round > 0 and site_id == "site-1":  # type: ignore
                        global_save_path = Path(f"/Users/lapotinacci/thesis/Federated_Sys/app/custom/models/fed_model_Arpat.keras")
                        global_save_path.parent.mkdir(parents=True, exist_ok=True)
                        model.model.save(global_save_path) # type: ignore
                
                #calcolo metriche pre-training
                y_pred = model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                for name, val in metrics.items():
                    writer.add_scalar(f"validation/initial_{name}", val, current_round)

            if current_round >= 1: # type: ignore
                model.train(X_train=X_train, y_train=y_train, val_split=0.2, early_stop=False)
            else:
                model.train(X_train=X_train, y_train=y_train, val_split=0.2)

            # calcolo metriche post-training
            y_pred = model.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred)
            for name, val in metrics.items():
                writer.add_scalar(f"validation/final_{name}", val, current_round)

            

            # invio modello aggiornato
            output_model = flare.FLModel(
                params={layer.name: layer.get_weights() for layer in model.model.layers}, # type: ignore
                params_type="FULL",
                metrics=metrics,
                current_round=current_round,
                meta={"NUM_STEPS_CURRENT_ROUND": float(num_steps)}
            )

            writer.flush()
            flare.send(output_model)

    except Exception as e:
        print(f"Client error: {e}", flush=True)


if __name__ == "__main__":
    client()
