import pandas as pd
import ast
import torch
from torch.utils.data import random_split, TensorDataset, DataLoader

def load_data(data, id, batch_size):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    use_columns = ['mean', 'median', 'max', 'min', 'var', 'avg_variation', 'linear_trend']  # Adjust this if you want to specify columns to load
    try:
        data = pd.read_csv(data)
        #data = data[data['type_of_TTT'] == 'daily']
        data['TTT'] = data['TTT'].apply(ast.literal_eval)

        filtered_data = _filtering(data, use_columns)

        #create different tensors for the 2 clients with low common values from the same dataset
        X_train, X_test, y_train, y_test = _create_highly_differentiated_data(filtered_data, id)
        print(f"Tensors, loaded succesfully: {X_train.shape}, {len(y_train[0])}, {X_test.shape}, {len(y_test[0])}")

        #data loading 
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(test_dataset, batch_size, shuffle=True)
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def _filtering(df, columns):
    """
    Filter the TTT values taking only the 24 hours series

    Parameters:
    df: DataFrame

    Returns:
    df[lunghezze == max_len]: Filtered Dataset based on TTT lenght
    """

    lunghezze = df['TTT'].apply(len)
    max_len = lunghezze.max()

    df = df[lunghezze == max_len]
    X = df[columns].values  # Filter to only include specified columns
    y = df['TTT'].values
    y =[row for row in y]
    return X,y


def _create_highly_differentiated_data(data, client_id, train_ratio=0.5):
    """
    Crea tesnori per l'addestramento dei due client
    con sovrapposizione 20-30% tra client
    """

    # Assicura che X e y siano tensori PyTorch
    X, y = data
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    print(f"Input shapes: X={X.shape}, y={y.shape}")
    
    # Seed per riproducibilità
    base_seeds = [12345, 67890, 13579, 24680, 97531]
    client_seed = base_seeds[client_id - 1] if client_id <= len(base_seeds) else client_id * 87654
    
    print(f"Client {client_id}: seed = {client_seed}")
    
    # Imposta il seed per PyTorch
    torch.manual_seed(client_seed)
    
    # Crea dataset PyTorch
    dataset = TensorDataset(X, y)
    
    # Calcola dimensioni
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # Split usando PyTorch (mantiene consistenza)
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(client_seed)
    )
    
    # Estrai tensori dai dataset
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    '''
    print(f"Client {client_id}: "
          f"X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_val={X_val.shape}, y_val={y_val.shape}")
    '''
    
    
    return X_train, X_val, y_train, y_val