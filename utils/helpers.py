import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from functools import lru_cache
from tqdm import tqdm
import psycopg2
import redshift_connector
from dateutil.relativedelta import relativedelta

def get_ffrct_by(df, 
                    group_by='salesforce_id',
                    start_month=None, 
                    start_year=None, 
                    end_month=None, 
                    end_year=None):
    """
    Calculate the number of unique FFRct cases (hf_id) for a given grouping and time period.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The main dataframe containing case data
        
    group_by : str, default='salesforce_id'
        Column to group by. Options:
        - 'salesforce_id' : Group by Salesforce ID (includes sfdc_account_name)
        - 'site_slug' : Group by Site Slug (includes site_name)
        - 'institution_name' : Group by Institution Name
        
    start_month : int, optional (1-12)
        Starting month (inclusive). If None, no start date filter applied.
        
    start_year : int, optional
        Starting year. Required if start_month is provided.
        
    end_month : int, optional (1-12)
        Ending month (inclusive). If None, no end date filter applied.
        
    end_year : int, optional
        Ending year. Required if end_month is provided.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - If group_by='salesforce_id': ['salesforce_id', 'sfdc_account_name', 'ffrct_count']
        - If group_by='site_slug': ['site_slug', 'site_name', 'ffrct_count']
        - If group_by='institution_name': ['institution_name', 'ffrct_count']
        Sorted by ffrct_count in descending order
    
    Example Usage:
    --------------
    # Get FFRct count by Salesforce ID for Jan 2024 - Mar 2024
    result = get_ffrct_by(df, 
                          group_by='salesforce_id',
                          start_month=1, start_year=2024,
                          end_month=3, end_year=2024)
    # Returns: salesforce_id, sfdc_account_name, ffrct_count
    
    # Get FFRct count by Site Slug for all of 2024
    result = get_ffrct_by(df,
                          group_by='site_slug', 
                          start_month=1, start_year=2024,
                          end_month=12, end_year=2024)
    # Returns: site_slug, site_name, ffrct_count
    
    # Get FFRct count by Institution with no date filter
    result = get_ffrct_by(df, group_by='institution_name')
    # Returns: institution_name, ffrct_count
    """
    
    valid_groups = ['salesforce_id', 'site_slug', 'institution_name']
    if group_by not in valid_groups:
        raise ValueError(f"group_by must be one of {valid_groups}")
    
    if start_month is not None and start_year is None:
        raise ValueError("start_year is required when start_month is provided")
    if end_month is not None and end_year is None:
        raise ValueError("end_year is required when end_month is provided")
    
    ffrct_filter = (
        (df['total_commercial'] == True) & 
        (df['latest_submission'] == True) & 
        (df['case_state'].isin(['COMPLETED', 'RCAG_HOLDING', 'RETURNED'])) & 
        (df['Case-Specific Product Offerings'].isin(['CCTA/FFRct Only', 'Billable: Plaque + FFRct', 'Non-Billable: Plaque + FFrct'])) & 
        (df['revenue_generating'] == True)
    )
    
    filtered_df = df[ffrct_filter].copy()
    
    if start_month is not None and start_year is not None:
        start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
        filtered_df = filtered_df[filtered_df['billing_timestamp_local'] >= start_date]
    
    if end_month is not None and end_year is not None:
        if end_month == 12:
            end_date = pd.Timestamp(year=end_year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
        filtered_df = filtered_df[filtered_df['billing_timestamp_local'] <= end_date]
    
    if group_by == 'salesforce_id':
        result = filtered_df.groupby([group_by, 'sfdc_account_name'])['hf_id'].nunique().reset_index()
        result.columns = ['salesforce_id', 'sfdc_account_name', 'ffrct_count']
    elif group_by == 'site_slug':
        result = filtered_df.groupby([group_by, 'site_name'])['hf_id'].nunique().reset_index()
        result.columns = ['site_slug', 'site_name', 'ffrct_count']
    else:  # institution_name
        result = filtered_df.groupby(group_by)['hf_id'].nunique().reset_index()
        result.columns = ['institution_name', 'ffrct_count']
    
    result = result.sort_values('ffrct_count', ascending=False).reset_index(drop=True)
    
    return result

def get_ccta_by(df, 
                group_by='salesforce_id',
                start_month=None, 
                start_year=None, 
                end_month=None, 
                end_year=None):
    """
    Calculate the number of unique CCTA cases (hf_id) for a given grouping and time period.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The main dataframe containing case data
        
    group_by : str, default='salesforce_id'
        Column to group by. Options:
        - 'salesforce_id' : Group by Salesforce ID (includes sfdc_account_name)
        - 'site_slug' : Group by Site Slug (includes site_name)
        - 'institution_name' : Group by Institution Name
        
    start_month : int, optional (1-12)
        Starting month (inclusive). If None, no start date filter applied.
        
    start_year : int, optional
        Starting year. Required if start_month is provided.
        
    end_month : int, optional (1-12)
        Ending month (inclusive). If None, no end date filter applied.
        
    end_year : int, optional
        Ending year. Required if end_month is provided.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - If group_by='salesforce_id': ['salesforce_id', 'sfdc_account_name', 'ccta_count']
        - If group_by='site_slug': ['site_slug', 'site_name', 'ccta_count']
        - If group_by='institution_name': ['institution_name', 'ccta_count']
        Sorted by ccta_count in descending order
    
    Example Usage:
    --------------
    # Get CCTA count by Salesforce ID for Jan 2024 - Mar 2024
    result = get_ccta_by(df, 
                         group_by='salesforce_id',
                         start_month=1, start_year=2024,
                         end_month=3, end_year=2024)
    # Returns: salesforce_id, sfdc_account_name, ccta_count
    
    # Get CCTA count by Site Slug for all of 2024
    result = get_ccta_by(df,
                         group_by='site_slug', 
                         start_month=1, start_year=2024,
                         end_month=12, end_year=2024)
    # Returns: site_slug, site_name, ccta_count
    
    # Get CCTA count by Institution with no date filter
    result = get_ccta_by(df, group_by='institution_name')
    # Returns: institution_name, ccta_count
    """
    
    valid_groups = ['salesforce_id', 'site_slug', 'institution_name']
    if group_by not in valid_groups:
        raise ValueError(f"group_by must be one of {valid_groups}")
    
    if start_month is not None and start_year is None:
        raise ValueError("start_year is required when start_month is provided")
    if end_month is not None and end_year is None:
        raise ValueError("end_year is required when end_month is provided")
    
    ccta_filter = (
        (df['total_commercial'] == True) & 
        (df['latest_submission'] == True)
    )
    
    filtered_df = df[ccta_filter].copy()
    
    if start_month is not None and start_year is not None:
        start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
        filtered_df = filtered_df[filtered_df['created_at_local'] >= start_date]
    
    if end_month is not None and end_year is not None:
        if end_month == 12:
            end_date = pd.Timestamp(year=end_year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
        filtered_df = filtered_df[filtered_df['created_at_local'] <= end_date]
    
    if group_by == 'salesforce_id':
        result = filtered_df.groupby([group_by, 'sfdc_account_name'])['hf_id'].nunique().reset_index()
        result.columns = ['salesforce_id', 'sfdc_account_name', 'ccta_count']
    elif group_by == 'site_slug':
        result = filtered_df.groupby([group_by, 'site_name'])['hf_id'].nunique().reset_index()
        result.columns = ['site_slug', 'site_name', 'ccta_count']
    else:  # institution_name
        result = filtered_df.groupby(group_by)['hf_id'].nunique().reset_index()
        result.columns = ['institution_name', 'ccta_count']
    
    result = result.sort_values('ccta_count', ascending=False).reset_index(drop=True)
    
    return result