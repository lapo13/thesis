import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, required=True)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr ", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Argomenti sconosciuti ignorati: {unknown_args}")
    
    return args

def build_script_args(data_file, client_id, device, epochs=10, lr=0.01, batch_size=32):
    return f"--data_url '{data_file}' --client_id {client_id} --epochs {epochs} --lr {lr} --batch_size {batch_size} --device '{device}'"
