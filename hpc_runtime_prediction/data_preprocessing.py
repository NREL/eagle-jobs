import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional

def convert_timedelta_columns(df, columns):
    """
    Convert a Timedelta column to integer number of seconds.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the Timedelta columns will be converted.
    - columns (List[str]): A list of the columns to be converted.
    """
    for column in columns:
        df[column] = df[column] // pd.Timedelta(seconds=1)

def convert_datetime_columns(df, columns):
    """
    Convert a Datetime column to integer number of seconds since UNIX epoch time.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the Datetime columns will be converted.
    - columns (List[str]): A list of the columns to be converted.
    """
    for column in columns:
        df[column] = (df[column] - np.datetime64('1970-01-01T00:00:00')) // np.timedelta64(1, 's')

def convert_string_to_int_columns(df, columns):
    """
    Convert strings to integers in pandas DataFrame columns.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the columns will be converted.
    - columns (List[str]): A list of the columns to be converted.
    """
    for column in columns:
        df[column] = df[column].apply(int)

def hour_to_seconds(hour) -> int:
    """
    Convert hours to seconds.
    
    Parameters:
    - hour: Number of hours.
    
    Returns:
    - int: Integer number of seconds.
    """
    if pd.isnull(hour):
        return hour
    else:
        return(int(hour * 3600))

def convert_hours_to_seconds_columns(df, columns):
    """
    Convert a column of hours (Float) to integer # of seconds.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the columns will be converted.
    - columns (List[str]): A list of the columns to be converted.
    """
    for column in columns:
        df[column] = df[column].apply(hour_to_seconds)

def convert_req_mem(req_mem: str, nodes_req: int, cores_req: int) -> float: 
    """
    Convert a requested memory string to number of Megabytes.
    
    Various foormatting rules are applies:
    c - Megabytes per core
    Mc - Megabytes per core
    Gc - Gigabytes per core
    Tc - Terabytes per core
    n - Megabytes per node
    Mn - Megabytes per node
    Gn - Gigabytes per node
    Tn - Terabytes per node
    M - Megabytes
    G - Gigabytes
    T - Terabytes
    
    If no label is provided, the default is Megabytes.
    
    Parameters:
    - req_mem (str): The requested memory.
    - nodes_req (int): The number of nodes requested.
    - cores_req (int): The number of cores requested.
    
    Returns:
    - float: The amount of memory requested (in megabytes)
    """
    if req_mem.endswith('Mc'): 
        return float(req_mem[:-2]) * cores_req 
    elif req_mem.endswith('Gc'): 
        return float(req_mem[:-2]) * 1024 * cores_req 
    elif req_mem.endswith('c'): 
        return float(req_mem[:-1]) * cores_req 
    elif req_mem.endswith('Mn'): 
        return float(req_mem[:-2]) * nodes_req 
    elif req_mem.endswith('Gn'): 
        return float(req_mem[:-2]) * 1024 * nodes_req 
    elif req_mem.endswith('n'): 
        return float(req_mem[:-1]) * nodes_req 
    elif req_mem.endswith('M'): 
        return float(req_mem[:-1])
    elif req_mem.endswith('G'): 
        return float(req_mem[:-1]) * 1024
    elif req_mem.endswith('T'): 
        return float(req_mem[:-1]) * 1024 * 1024
    elif req_mem.endswith('?'): 
        return float(req_mem[:-1])
    else: 
        return float(req_mem)

def convert_req_mem_string_to_float(df):
    """
    Normalize the requested memory column to number of Megabytes.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the column will be converted.
    """
    df['mem_req'] = df.apply(lambda x: convert_req_mem(x['req_mem'], x['nodes_req'], x['processors_req']), axis=1)

def label_encode_columns(df, columns):
    """
    Label encode a categorical column in a DataFrame and save the encoding as a list in a new column.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the categorical columns will be encoded.
    - columns (List[str]): A list of the columns to be encoded.
    """
    for column in columns:
        le = LabelEncoder()
        le = le.fit(df[column])
        df[column] = le.transform(df[column])

def one_hot_encode_columns(df, columns):
    """
    One-hot encode a categorical column in a DataFrame and save the encoding as a list in a new column.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the categorical columns will be encoded.
    - columns (List[str]): A list of the columns to be encoded.
    """
    for column in columns:
        one_hot = pd.get_dummies(df[column]).to_numpy()
        df[column + '_onehot'] = one_hot.tolist()
        
# Lookup Dictionary to convert Partition acronyms to full titles
partition_lookup = {
    'standard': 'standard', 
    'bigmem-8600': 'big memory 8600', 
    'bigmem': 'big memory', 
    'dav': 'data analytics visualization', 
    'ddn': 'ddn', 
    'gpu': 'graphics processing unit', 
    'short': 'short', 
    'long': 'long', 
    'bigscratch': 'big scratch', 
    'debug': 'debug', 
    'csc': 'computational science center', 
    'mono': 'mono', 
    'haswell': 'haswell', 
    'standard,': 'standard', 
    'weto': 'wind energy technologies office', 
    'weto,standard': 'wind energy technologies office standard', 
    'weto,short': 'wind energy technologies office short', 
    'standard,weto': 'standard wind energy technologies office', 
    'short,weto': 'short wind energy technologies office', 
    'amo': 'advanced manufacturing office', 
    'long,weto': 'long wind energy technologies office', 
    'standard,amo': 'standard advanced manufacturing office', 
    'short,amo': 'short advanced manufacturing office', 
    'long,amo': 'long advanced manufacturing office', 
    'gpul': 'graphics processing unit long', 
    'csc-stdby': 'computational science center standby', 
    'short-stdby': 'short standby', 
    'gpul-stdby': 'graphics processing unit long standby', 
    'standard-stdby': 'standard standby', 
    'debug-stdby': 'debug standby', 
    'gpu-stdby': 'graphics processing unit standby', 
    'long-stdby': 'long standby', 
    'bigscratch-stdby': 'big scratch standby', 
    'bigmem-stdby': 'big memory standby', 
    'weto-stdby,short-stdby': 'wind energy technologies office standby short standby', 
    'short,debug': 'short debug', 
    'long-stdby,weto-stdby': 'long standby wind energy technologies office standby', 
    'amo-stdby': 'advanced manufacturing office standby', 
    'amo-stdby,long-stdby': 'advanced manufacturing office standby long standby', 
    'weto-stdby': 'wind energy technologies office standby', 
    'amo,standard': 'advanced manufacturing office standard', 
    'short,amo,standard': 'short advanced manufacturing office standard'
}

# Lookup Dictionary to convert QOS acronyms to full titles
qos_lookup = {
    'Unknown': 'unknown',
    'normal': 'normal',
    'high': 'high',
    'debug': 'debug',
    'penalty': 'penalty',
    'standby': 'standby',
    'buy-in': 'buy in',
    'weto': 'wind energy technologies office',
    'p_gpu_half': 'p graphics processing unit half'
}

        
def encode_other(column, instance, top_n):
    if instance in top_n:
        return instance
    else:
        return column + '_other'
        
def onehot_with_other(df, columns, n_values):
    encoded_columns = dict()
    for i, column in enumerate(columns):
        encoded_columns[column] = list()
        n = n_values[i]
        top_n = df[column].value_counts().nlargest(n).index.tolist()

        for i, instance in enumerate(top_n):
            df[column+'_'+str(i)] = 0
            encoded_columns[column].append(column+'_'+str(i))
        df[column+'_other'] = 0
        encoded_columns[column].append(column+'_other')

        df[column+'_with_other'] = df[column].apply(lambda x: encode_other(column, x, top_n))

        for i, instance in enumerate(top_n):
            df.loc[df[column+'_with_other'] == instance, column+'_'+str(i)] = 1
        df.loc[df[column+'_with_other'] == column+'_other', column+'_other'] = 1

        df.drop(column+'_with_other', axis=1, inplace=True)
    
    return encoded_columns