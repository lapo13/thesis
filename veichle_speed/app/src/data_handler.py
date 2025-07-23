import pandas as pd
import ast
#from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    use_columns = ['mean', 'median', 'max', 'min', 'var', 'avg_variation', 'linear_trend']  # Adjust this if you want to specify columns to load
    try:
        data = pd.read_csv(file_path)
        #data = data[data['type_of_TTT'] == 'daily']
        data['TTT'] = data['TTT'].apply(ast.literal_eval)
        data = _list_filtering(data)
        
        X = data[use_columns].values  # Filter to only include specified columns
        y = data['TTT'].values
        y =[row for row in y]

        print(f"Data loaded with shape: {X.shape} ")
        return X, y
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def _list_filtering(df):
    """
    Filter the TTT values taking only the 24 hours series

    Parameters:
    df: DataFrame

    Returns:
    df[lunghezze == max_len]: Filtered Dataset based on TTT lenght
    """
    lunghezze = df['TTT'].apply(len)
    max_len = lunghezze.max()
    return df[lunghezze == max_len]
