import pandas as pd
from datetime import datetime

def initialize_version_columns(df):
    """
    Initialize version tracking columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to initialize version tracking for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with version tracking columns added
    """
    df_copy = df.copy()
    
    # Add version tracking columns if they don't exist
    if 'version' not in df_copy.columns:
        df_copy['version'] = 1
    if 'created_at' not in df_copy.columns:
        df_copy['created_at'] = datetime.now()
    if 'last_updated' not in df_copy.columns:
        df_copy['last_updated'] = datetime.now()
        
    return df_copy

def update_version(df, changed_indices=None):
    """
    Update version and timestamp for modified rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to update versions for
    changed_indices : list-like or None, optional
        Indices of rows that were modified. If None, updates all rows.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated version numbers and timestamps
    """
    df_copy = df.copy()
    current_time = datetime.now()
    
    # Initialize version columns if they don't exist
    df_copy = initialize_version_columns(df_copy)
    
    # Update version and timestamp for changed rows
    if changed_indices is not None:
        df_copy.loc[changed_indices, 'version'] += 1
        df_copy.loc[changed_indices, 'last_updated'] = current_time
    else:
        df_copy['version'] += 1
        df_copy['last_updated'] = current_time
        
    return df_copy

def track_changes(original_df, modified_df):
    """
    Compare two DataFrames and track version changes for modified rows.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        The original DataFrame before modifications
    modified_df : pandas.DataFrame
        The modified DataFrame after changes
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated version numbers and timestamps for changed rows
    """
    # Initialize version columns if they don't exist
    if 'version' not in original_df.columns:
        original_df = initialize_version_columns(original_df)
    
    # Find changed rows by comparing all columns except version tracking columns
    version_cols = ['version', 'created_at', 'last_updated']
    data_cols = [col for col in original_df.columns if col not in version_cols]
    
    # Find indices where data has changed
    changed_mask = (original_df[data_cols] != modified_df[data_cols]).any(axis=1)
    changed_indices = changed_mask[changed_mask].index
    
    # Update versions for changed rows
    result_df = update_version(modified_df, changed_indices)
    
    return result_df