import os, re, ast, logging
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from typing import List, Optional, Dict, Any


class DataHandler:
    """    
    Attributes:
        data_path (str): Path to the directory containing data files
        data (pd.DataFrame): Loaded and processed dataset
        X, X_tr, X_ts (np.ndarray): Features for full, train, and test sets
        y, y_tr, y_ts (np.ndarray): Targets for full, train, and test sets
    """
    
    def __init__(self, data_path: str):
        """        
        Args:
            data_path (str): Path to the directory containing data files
            
        Raises:
            FileNotFoundError: If data_path does not exist
        """
        self.data_path = os.path.abspath(data_path)
        self.data = None
        self.X, self.X_tr, self.X_ts = None, None, None
        self.y, self.y_tr, self.y_ts = None, None, None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that data_path exists and is a directory."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError(f"Data path is not a directory: {self.data_path}")
    
    def _is_useful_series(self, series: List[float], 
                         zero_ratio_thr: float = 0.75, 
                         var_thr: float = 3e-1, 
                         unique_thr: int = 3) -> bool:
        """
        Args:
            series (List[float]): Time series data
            zero_ratio_thr (float): Maximum allowed ratio of zero values
            var_thr (float): Minimum required variance
            unique_thr (int): Minimum required unique values
            
        Returns:
            bool: True if series meets quality criteria
        """
        try:
            arr = np.array(series, dtype=float)
            
            # Check variance
            if np.var(arr) <= var_thr:
                return False
                
            # Check zero ratio
            zero_ratio = (arr == 0).mean()
            if zero_ratio >= zero_ratio_thr:
                return False
                
            # Check unique values
            if len(np.unique(arr)) <= unique_thr:
                return False
                
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error processing series: {e}")
            return False
    
    def _filter_series(self, df: pd.DataFrame, target_column: str = "TTT", lenght: int = 24) -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the column containing time series
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Filter by series length
        series_lengths = df[target_column].apply(len)
        
        if lenght == 0:
            raise ValueError("No valid series found in the dataset")
            
        df_filtered = df[series_lengths == lenght]
        self.logger.info(f"Filtered to {len(df_filtered)} series of length {lenght}")
        
        # Filter by series quality
        mask_useful = df_filtered[target_column].apply(self._is_useful_series)
        df_useful = df_filtered[mask_useful]
        self.logger.info(f"Further filtered to {len(df_useful)} useful series")
        
        return df_useful
    
    def load_data(self, 
                  columns: List[str], 
                  categorical_vars: List[str], 
                  file_pattern_var: str,
                  target_column: str = "TTT",
                  split: str = "EMBEDDING") -> None:
        """Load and preprocess data."""
        
        # ✅ FIX: Parse columns if they arrive as a string
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(',')]
        
        # ✅ FIX: Parse categorical_vars if string
        if isinstance(categorical_vars, str):
            categorical_vars = [c.strip() for c in categorical_vars.split(',') if c.strip()]
        
        # Validate inputs
        if not columns:
            raise ValueError("Columns list cannot be empty")
        if target_column not in columns:
            raise ValueError(f"Target column '{target_column}' must be in columns list")
        
        try:
            # Find matching files
            files = [f for f in os.listdir(self.data_path) 
                    if os.path.isfile(os.path.join(self.data_path, f))]
            
            pattern = rf".*{re.escape(file_pattern_var)}"
            matching_files = [f for f in files if re.search(pattern, f)]
            
            if not matching_files:
                raise FileNotFoundError(f"No files found matching pattern: {file_pattern_var}")
            
            self.logger.info(f"Found {len(matching_files)} matching files: {matching_files}")
            
            # Load and concatenate data
            data_frames = []
            for file in matching_files:
                file_path = os.path.join(self.data_path, file)
                df = pd.read_excel(file_path, usecols=columns)
                data_frames.append(df)

            self.smart_interpolation(data_frames)
            
            combined_df = pd.concat(data_frames, ignore_index=True)
            self.logger.info(f"Loaded {len(combined_df)} total records")
            
            # Parse TTT column
            if target_column in combined_df.columns:
                combined_df[target_column] = combined_df[target_column].apply(ast.literal_eval)
            else:
                raise ValueError(f"Target column '{target_column}' not found in loaded data")
            
            # Filter series
            filtered_df = self._filter_series(combined_df, target_column)
            
            # Remove columns with NaN values
            cols_with_nan = filtered_df.columns[filtered_df.isna().any()]
            if len(cols_with_nan) > 0:
                self.logger.info(f"Removing columns with NaN: {list(cols_with_nan)}")
                df_clean = filtered_df.drop(columns=cols_with_nan)
            else:
                df_clean = filtered_df
            
            # ✅ CRITICAL: Select columns BEFORE encoding time features
            available_columns = [col for col in columns if col in df_clean.columns]
            missing_columns = set(columns) - set(available_columns)
            
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
            
            df_selected = df_clean[available_columns]
            
            # ✅ Store temporarily WITHOUT dropping time columns yet
            self.data = df_selected
            
            # ✅ NOW encode time features (uses interval_start/end)
            self.encode_time_features()
            
            # ✅ NOW we can drop interval columns if they were temporary
            time_cols_to_drop = ['interval_start', 'interval_end']
            self.data = self.data.drop(columns=[c for c in time_cols_to_drop if c in self.data.columns], errors='ignore')
            
            # One-hot encode categorical variables
            existing_categorical = [var for var in categorical_vars if var in self.data.columns]
            missing_categorical = set(categorical_vars) - set(existing_categorical)
            
            if missing_categorical:
                self.logger.warning(f"Missing categorical variables: {missing_categorical}")
            
            if existing_categorical:
                self.data = pd.get_dummies(self.data, columns=existing_categorical)
                self.logger.info(f"Applied one-hot encoding to {len(existing_categorical)} categorical variables")
            
            # Split embeddings/series into columns
            self.data = self._spezza_serie_in_colonne(colonna_stringa=split, prefix='split_')
            
            # ✅ NOW split the data into train/test
            self.split_data(target_column=target_column)
            
            self.logger.info(f"Final dataset shape: {self.data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def encode_time_features(self, 
                           start_col: str = "interval_start", 
                           end_col: str = "interval_end") -> None:
        """
        Encode temporal features using cyclic encoding.
        
        Args:
            start_col (str): Name of start time column
            end_col (str): Name of end time column
            
        Raises:
            ValueError: If data is not loaded or required columns are missing
        """
        if self.data is None:
            raise ValueError("Data must be loaded before encoding time features")
        
        print(self.data.columns)
        
        required_cols = [start_col, end_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        try:
            # Convert to datetime
            self.data[start_col] = pd.to_datetime(self.data[start_col], utc=True)
            self.data[end_col] = pd.to_datetime(self.data[end_col], utc=True)

            self.data['month'] = self.data[start_col].dt.month
            self.data['day'] = self.data[start_col].dt.dayofweek
            
            self.data['month_cos'] = np.cos(2 * np.pi * self.data['month']/len(self.data['month'].unique()))
            self.data['month_sin'] = np.sin(2 * np.pi * self.data['month']/len(self.data['month'].unique()))
            self.data['day_sin'] = np.sin(2 * np.pi * self.data['day']/len(self.data['day'].unique()))
            self.data['day_cos'] = np.cos(2 * np.pi * self.data['day']/len(self.data['day'].unique()))
            # Encode start time features

            self.data["hour_start"] = self.data[start_col].dt.hour
            angle_day_start = 2 * np.pi * self.data["hour_start"] / 24
            self.data["start_time_sin"] = np.sin(angle_day_start)
            self.data["start_time_cos"] = np.cos(angle_day_start)
            
            # Encode end time features
            self.data["hour_end"] = self.data[end_col].dt.hour
            angle_day_end = 2 * np.pi * self.data["hour_end"] / 24
            self.data["end_time_sin"] = np.sin(angle_day_end)
            self.data["end_time_cos"] = np.cos(angle_day_end)
            
            # Calculate duration
            self.data["duration_minutes"] = (
                self.data[end_col] - self.data[start_col]
            ).dt.total_seconds() / 60
            
            # Encode day of week
            dayofweek = self.data[start_col].dt.dayofweek
            angle_week = 2 * np.pi * dayofweek / 7
            self.data["dow_sin"] = np.sin(angle_week)
            self.data["dow_cos"] = np.cos(angle_week)
            
            # Encode day of year
            dayofyear = self.data[start_col].dt.dayofyear
            angle_year = 2 * np.pi * dayofyear / 365
            self.data["doy_sin"] = np.sin(angle_year)
            self.data["doy_cos"] = np.cos(angle_year)
            
            # Remove original and helper columns
            columns_to_drop = [start_col, end_col, "hour_start", "hour_end", "day", "month"]
            self.data.drop(columns=columns_to_drop, inplace=True)
            
            self.logger.info("Successfully encoded time features")
            
        except Exception as e:
            self.logger.error(f"Error encoding time features: {e}")
            raise
    
    def _series_to_2d_array(self, target_column: str = "TTT", output_dim: int = 24) -> np.ndarray:
        """
        Convert series column to 2D numpy array.
        
        Args:
            target_column (str): Name of the column containing series data
            output_dim (int): Expected dimension of each series
            
        Returns:
            np.ndarray: 2D array of shape (n_samples, output_dim)
            
        Raises:
            ValueError: If series have inconsistent dimensions
        """
        if self.data is None or target_column not in self.data.columns:
            raise ValueError(f"Data not loaded or target column '{target_column}' not found")
        
        arr = np.stack(self.data[target_column].tolist())
        
        if arr.shape[1] != output_dim:
            raise ValueError(
                f"Expected each series to have {output_dim} values, "
                f"but found {arr.shape[1]}"
            )
        
        return arr.astype(np.float32)
    
    def split_data(self, 
               test_size: float = 0.2, 
               random_state: int = 42,
               target_column: str = "TTT",
               output_dim: int = 24,
               stratification_bins: Optional[List[float]] = None) -> None:
        """Split data into training and test sets."""
        if self.data is None:
            raise ValueError("Data must be loaded before splitting")
        
        if stratification_bins is None:
            stratification_bins = [0.0, 0.4, 0.8, 1.0]
        
        try:
            # Prepare features and target
            self.X = self.data.drop(columns=[target_column]).to_numpy().astype(np.float32)
            self.y = self._series_to_2d_array(target_column, output_dim)
            
            # Calculate stratification variable
            std_scores = np.std(self.y, axis=1)
            std_scores_log = np.log1p(std_scores)
            std_scores_normalized = (
                (std_scores_log - np.min(std_scores_log)) / 
                (np.max(std_scores_log) - np.min(std_scores_log))
            )
            
            std_bins = np.digitize(std_scores_normalized, stratification_bins)
            
            # Perform stratified split
            self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=std_bins
            )
            
            self.logger.info(
                f"Data split: train={len(self.X_tr)}, test={len(self.X_ts)}, "
                f"features={self.X_tr.shape[1]}"
            )
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise
        
    
    def smart_interpolation(self, df_list) -> None:
        for i, df in enumerate(df_list):
            print(f"Processing dataset {i+1}...")
            
            for column in df.columns:
                missing_count = df[column].isna().sum()
                if missing_count > 0:
                    print(f"  - Column '{column}': {missing_count} missing values")
                    
                    # Colonne numeriche
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # Se pochi missing values, interpolazione
                        if missing_count / len(df) < 0.1:  # meno del 10%
                            df[column].interpolate(method='linear', inplace=True)
                        else:
                            # Se molti missing, usa la media
                            df[column].fillna(df[column].mean(), inplace=True)
                    
                    # Colonne stringa/categoriche
                    elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                        # Usa la moda (valore più frequente)
                        mode_values = df[column].mode()
                        if not mode_values.empty:
                            df[column].fillna(mode_values[0], inplace=True)
                        else:
                            df[column].fillna('Unknown', inplace=True)
                    
                    # Colonne datetime
                    elif pd.api.types.is_datetime64_any_dtype(df[column]):
                        df[column].interpolate(method='linear', inplace=True)
            
            print(f"Completed dataset {i+1}")
        
    def _spezza_serie_in_colonne(self, colonna_stringa, prefix='col_'):
        if self.data is None:
            raise ValueError("I dati non sono stati caricati") #la gestione degli errori potrebbe essere migliorata, un
        #oggetto custom che viene lanciato quando la funzione viene chiamata in modo errato (prima di caricare i dati) sarebbe piu' elegante

    # Verifica che la colonna esista
        if colonna_stringa not in self.data.columns:
            raise ValueError(f"La colonna '{colonna_stringa}' non esiste nel DataFrame")
        
        # Crea una copia del DataFrame
        result_df = self.data.copy()
        
        # Rimuove la colonna originale
        colonna_originale = result_df.pop(colonna_stringa)
        
        # Lista per raccogliere tutti gli array convertiti
        arrays_convertiti = []
        
        # Converte ogni stringa in array numerico
        for stringa in colonna_originale:
            # Gestione di valori NaN o stringhe vuote
            if pd.isna(stringa) or stringa == '':
                arrays_convertiti.append([])
                continue
                
            # Rimuove le parentesi quadre se presenti
            stringa_pulita = str(stringa).strip('[]')
            
            # Divide per spazi multipli e filtra stringhe vuote
            valori = [x for x in stringa_pulita.split() if x]
            
            # Converte in float
            try:
                array_numerico = [float(x) for x in valori]
                arrays_convertiti.append(array_numerico)
            except ValueError as e:
                print(f"Errore nella conversione di '{stringa}': {e}")
                arrays_convertiti.append([])
        
        # Trova la lunghezza massima per determinare il numero di colonne
        lunghezze = [len(arr) for arr in arrays_convertiti]
        
        if not lunghezze:
            raise ValueError("Nessun dato valido trovato nella colonna")
        
        lunghezza_max = max(lunghezze)
        lunghezza_min = min(lunghezze)
        
        # Verifica che tutti gli array abbiano la stessa lunghezza
        if lunghezza_max != lunghezza_min:
            print(f"Attenzione: array di lunghezze diverse (min: {lunghezza_min}, max: {lunghezza_max})")
            print("I valori mancanti verranno riempiti con NaN")
            
            # Riempie con NaN gli array più corti
            arrays_completi = []
            for arr in arrays_convertiti:
                if len(arr) < lunghezza_max:
                    arr_completo = arr + [np.nan] * (lunghezza_max - len(arr))
                    arrays_completi.append(arr_completo)
                else:
                    arrays_completi.append(arr)
        else:
            arrays_completi = arrays_convertiti
        
        # Crea il DataFrame con le nuove colonne
        nuove_colonne = pd.DataFrame(
            arrays_completi,
            index=result_df.index,
            columns=[f"{prefix}{i}" for i in range(lunghezza_max)]
        )
        
        # Combina con il DataFrame originale
        result_df = pd.concat([result_df, nuove_colonne], axis=1)
        
        return result_df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dict[str, Any]: Dictionary containing data summary
        """
        if self.data is None:
            return {"status": "No data loaded"}
        
        summary = {
            "status": "Data loaded",
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
        }
        
        if all(x is not None for x in [self.X_tr, self.X_ts, self.y_tr, self.y_ts]):
            summary.update({
                "train_shape": self.X_tr.shape, # type: ignore
                "test_shape": self.X_ts.shape, # type: ignore
                "target_train_shape": self.y_tr.shape, # type: ignore
                "target_test_shape": self.y_ts.shape # type: ignore
            })
        
        return summary
    
    def has_split_data(self) -> bool:
        """Check if data has been split into train/test sets."""
        return all([
            self.X_tr is not None,
            self.X_ts is not None,
            self.y_tr is not None,
            self.y_ts is not None
        ])
    
    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get training data with validation."""
        if self.X_tr is None or self.y_tr is None:
            raise ValueError("Training data not available. Call split_data() first")
        return self.X_tr, self.y_tr
    
    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get test data with validation."""
        if self.X_ts is None or self.y_ts is None:
            raise ValueError("Test data not available. Call split_data() first")
        return self.X_ts, self.y_ts
    
    def load_and_preprocess(
        self,
        columns: List[str],
        categorical_vars: List[str],
        target_column: str,
        start_col: Optional[str] = None,  # Nome della colonna
        end_col: Optional[str] = None,    # Nome della colonna
        test_size: float = 0.2,
        random_state: int = 42
    ):
        if self.data is None:
            raise ValueError("Data must be loaded before preprocessing")
        # Se specificate, converti le colonne datetime
        if start_col and start_col in self.data.columns:
            self.data[start_col] = pd.to_datetime(self.data[start_col])
        
        if end_col and end_col in self.data.columns:
            self.data[end_col] = pd.to_datetime(self.data[end_col])
        
'''
to_keep = ['type_of_TTT', 'min', 'var', 'median', 'interval_end', 
            'is_festive', 'mean', 'max', 'interval_start', 'linear_trend',
            'is_weekend', 'EMBEDDING','month', 'avg_variation', 
            'TTT','day'
            ]

handler = DataHandler(data_path="../data")  # Example instantiation
print(handler.data_path)
handler.load_data(
    columns=to_keep,
    categorical_vars=["is_festive", "is_weekend", "type_of_TTT"],
    file_pattern_var="ARPAT_with_embeddings",
    split="EMBEDDING"
)  # Example data loading
handler.split_data()  # Example data splitting

print(handler.get_data_summary())  # Example usage'''