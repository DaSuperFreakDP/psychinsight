import pandas as pd
import numpy as np
import streamlit as st

class DataManager:
    def __init__(self):
        self.data = None
        self.column_descriptions = {}
    
    def load_data(self, uploaded_file):
        """Load data from uploaded file (CSV or Excel)"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        self.data = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Unable to decode CSV file with supported encodings")
            
            elif file_extension in ['xlsx', 'xls']:
                self.data = pd.read_excel(uploaded_file)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data cleaning
            self.data = self._clean_data(self.data)
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _clean_data(self, data):
        """Perform basic data cleaning"""
        # Remove completely empty rows and columns
        data = data.dropna(how='all', axis=0)
        data = data.dropna(how='all', axis=1)
        
        # Strip whitespace from string columns
        string_columns = data.select_dtypes(include=['object']).columns
        for col in string_columns:
            data[col] = data[col].astype(str).str.strip()
            # Replace 'nan' strings back to NaN
            data[col] = data[col].replace('nan', np.nan)
        
        # Convert numeric columns that might be stored as strings
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    # Remove any non-numeric characters except decimal point and minus
                    cleaned_series = data[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If more than 50% of values can be converted to numeric, treat as numeric
                    if pd.notna(numeric_series).sum() / len(numeric_series) > 0.5:
                        data[col] = numeric_series
                except:
                    pass
        
        return data
    
    def set_column_descriptions(self, descriptions):
        """Store column descriptions"""
        self.column_descriptions = descriptions
    
    def get_column_descriptions(self):
        """Get column descriptions"""
        return self.column_descriptions
    
    def get_data_summary(self):
        """Get a summary of the loaded data"""
        if self.data is None:
            return None
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': dict(self.data.dtypes),
            'missing_values': dict(self.data.isnull().sum()),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
        }
        
        return summary
    
    def filter_data(self, filters):
        """Apply filters to the data"""
        if self.data is None:
            return None
        
        filtered_data = self.data.copy()
        
        for column, filter_config in filters.items():
            if column not in filtered_data.columns:
                continue
            
            filter_type = filter_config.get('type')
            
            if filter_type == 'range' and 'min' in filter_config and 'max' in filter_config:
                filtered_data = filtered_data[
                    (filtered_data[column] >= filter_config['min']) &
                    (filtered_data[column] <= filter_config['max'])
                ]
            
            elif filter_type == 'values' and 'values' in filter_config:
                filtered_data = filtered_data[filtered_data[column].isin(filter_config['values'])]
        
        return filtered_data
    
    def get_column_stats(self, column):
        """Get statistics for a specific column"""
        if self.data is None or column not in self.data.columns:
            return None
        
        col_data = self.data[column]
        
        if col_data.dtype in [np.number]:
            stats = {
                'type': 'numeric',
                'count': col_data.count(),
                'missing': col_data.isnull().sum(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            }
        else:
            stats = {
                'type': 'categorical',
                'count': col_data.count(),
                'missing': col_data.isnull().sum(),
                'unique': col_data.nunique(),
                'top': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                'freq': col_data.value_counts().iloc[0] if not col_data.value_counts().empty else 0,
                'value_counts': dict(col_data.value_counts().head(10))
            }
        
        return stats
