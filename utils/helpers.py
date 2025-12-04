import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from functools import lru_cache
from tqdm import tqdm
import psycopg2
import redshift_connector
from dateutil.relativedelta import relativedelta

def fetch_dataframe_in_chunks(cursor, query, chunk_size=100000):
    cursor.execute(query)
    
    chunks = []
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
            
        chunk_df = pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
        chunks.append(chunk_df)
        print(f"Fetched chunk with {len(chunk_df)} rows")
    
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=[])

def fetch_dataframe(cursor, query):
    """Execute a query and fetch results as a DataFrame using psycopg2."""
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    return pd.DataFrame(rows, columns=columns)

def change_date_format(df):
    """Vectorized date format conversion."""
    datetime_columns = [
        "sfdc_ccta_fpa_end_date", "sfdc_ccta_fpa_start_date",
        "sfdc_hf_one_start_date", "hf_one_end_date",
        "hf_direct_end_date", "hf_direct_start_date",
        "plaque_claims_submitting_start_date__c", "plaque_claims_submitting_end_date__c",
        "sfdc_plaque_commercial_start_date", "sfdc_plaque_commercial_end_date",
        "plaque_pace_program_start_date__c", "plaque_pace_program_end_date__c",
        "plaque_ordered_at_local", "created_at_local"
    ]
    
    cols_to_convert = [col for col in datetime_columns if col in df.columns]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_datetime, errors='coerce')
    
    return df

def create_plaque_billing_category(df):
    df_result = df.copy()
    df_result['Case-Specific Plaque Billing Category'] = 'Other'
    
    plaque_ordered = 'plaque_ordered_at_local'
    
    mask_no_plaque = df_result[plaque_ordered].isna()
    df_result.loc[mask_no_plaque, 'Case-Specific Plaque Billing Category'] = 'No Plaque Ordered'
    
    # Condition 2: FPA EA Eval (90 day Eval)
    if 'plaque_fpa_ea_eval_start_date__c' in df_result.columns and \
       'plaque_fpa_ea_eval_end_date__c' in df_result.columns and \
       'plaque_early_adopt_type__c' in df_result.columns:
        
        mask_ea_eval = (
            (df_result[plaque_ordered] >= df_result['plaque_fpa_ea_eval_start_date__c']) &
            (df_result[plaque_ordered] <= df_result['plaque_fpa_ea_eval_end_date__c']) &
            (df_result['plaque_early_adopt_type__c'] == '90 day Eval')
        )
        df_result.loc[mask_ea_eval, 'Case-Specific Plaque Billing Category'] = 'Non-Billable: EA Eval'
    
    # Condition 3 & 4: Commercial Plaque Program
    if 'sfdc_plaque_commercial_start_date' in df_result.columns and \
       'sfdc_plaque_commercial_end_date' in df_result.columns and \
       'sfdc_plaque_program' in df_result.columns:
        
        # With end date
        mask_commercial_with_end = (
            (df_result[plaque_ordered] >= df_result['sfdc_plaque_commercial_start_date']) &
            (df_result['sfdc_plaque_commercial_end_date'].notna()) &
            (df_result[plaque_ordered] < df_result['sfdc_plaque_commercial_end_date'])
        )
        df_result.loc[mask_commercial_with_end, 'Case-Specific Plaque Billing Category'] = \
            'Commercial: ' + df_result.loc[mask_commercial_with_end, 'sfdc_plaque_program'].astype(str)
        
        # Without end date (ongoing)
        mask_commercial_no_end = (
            (df_result[plaque_ordered] >= df_result['sfdc_plaque_commercial_start_date']) &
            (df_result['sfdc_plaque_commercial_end_date'].isna())
        )
        df_result.loc[mask_commercial_no_end, 'Case-Specific Plaque Billing Category'] = \
            'Commercial: ' + df_result.loc[mask_commercial_no_end, 'sfdc_plaque_program'].astype(str)
    
    # Condition 5 & 6: Non-Billable PACE Program
    if 'plaque_pace_program_start_date__c' in df_result.columns and \
       'plaque_pace_program_end_date__c' in df_result.columns:
        
        # With end date
        mask_pace_with_end = (
            (df_result[plaque_ordered] >= df_result['plaque_pace_program_start_date__c']) &
            (df_result['plaque_pace_program_end_date__c'].notna()) &
            (df_result[plaque_ordered] <= df_result['plaque_pace_program_end_date__c'])
        )
        df_result.loc[mask_pace_with_end, 'Case-Specific Plaque Billing Category'] = 'Non-Billable: PACE'
        
        # Without end date (ongoing)
        mask_pace_no_end = (
            (df_result[plaque_ordered] >= df_result['plaque_pace_program_start_date__c']) &
            (df_result['plaque_pace_program_end_date__c'].isna())
        )
        df_result.loc[mask_pace_no_end, 'Case-Specific Plaque Billing Category'] = 'Non-Billable: PACE'
    
    # Condition 7 & 8: Registry Programs
    if 'plaque_registry_start_date__c' in df_result.columns and \
       'plaque_registry_end_date__c' in df_result.columns and \
       'plaque_registry_program__c' in df_result.columns:
        
        # With end date
        mask_registry_with_end = (
            (df_result[plaque_ordered] >= df_result['plaque_registry_start_date__c']) &
            (df_result['plaque_registry_end_date__c'].notna()) &
            (df_result[plaque_ordered] <= df_result['plaque_registry_end_date__c'])
        )
        df_result.loc[mask_registry_with_end, 'Case-Specific Plaque Billing Category'] = \
            'Non-Billable: ' + df_result.loc[mask_registry_with_end, 'plaque_registry_program__c'].astype(str)
        
        # Without end date (ongoing)
        mask_registry_no_end = (
            (df_result[plaque_ordered] >= df_result['plaque_registry_start_date__c']) &
            (df_result['plaque_registry_end_date__c'].isna())
        )
        df_result.loc[mask_registry_no_end, 'Case-Specific Plaque Billing Category'] = \
            'Non-Billable: ' + df_result.loc[mask_registry_no_end, 'plaque_registry_program__c'].astype(str)
    
    return df_result

def create_product_offerings(df):
    if 'Case-Specific Plaque Billing Category' not in df.columns:
        raise ValueError("'Case-Specific Plaque Billing Category' column must exist first")
    
    df_result = df.copy()
    
    df_result['Case-Specific Product Offerings'] = 'Fix'
    
    plaque_ordered = 'plaque_ordered_at_local'
    billing_timestamp = 'billing_timestamp_local'
    plaque_category = 'Case-Specific Plaque Billing Category'
    
    # Condition 1: No Plaque Ordered
    mask_no_plaque = df_result[plaque_ordered].isna()
    df_result.loc[mask_no_plaque, 'Case-Specific Product Offerings'] = 'CCTA/FFRct Only'
    
    # Condition 2: Commercial with Billing (Plaque + FFRct)
    mask_commercial_with_billing = (
        df_result[plaque_category].str.contains('Commercial', na=False) &
        df_result[billing_timestamp].notna()
    )
    df_result.loc[mask_commercial_with_billing, 'Case-Specific Product Offerings'] = 'Billable: Plaque + FFRct'
    
    # Condition 3: Commercial without Billing (Plaque-Only)
    mask_commercial_no_billing = (
        df_result[plaque_category].str.contains('Commercial', na=False) &
        df_result[billing_timestamp].isna()
    )
    df_result.loc[mask_commercial_no_billing, 'Case-Specific Product Offerings'] = 'Billable: Plaque-Only'
    
    # Condition 4: Non-Billable or Other with Billing (Plaque + FFRct)
    mask_nonbillable_with_billing = (
        (df_result[plaque_category].str.contains('Non-Billable', na=False) | 
         df_result[plaque_category].str.contains('Other', na=False)) &
        df_result[billing_timestamp].notna()
    )
    df_result.loc[mask_nonbillable_with_billing, 'Case-Specific Product Offerings'] = 'Non-Billable: Plaque + FFrct'
    
    # Condition 5: Non-Billable or Other without Billing (Plaque-Only)
    mask_nonbillable_no_billing = (
        (df_result[plaque_category].str.contains('Non-Billable', na=False) | 
         df_result[plaque_category].str.contains('Other', na=False)) &
        df_result[billing_timestamp].isna()
    )
    df_result.loc[mask_nonbillable_no_billing, 'Case-Specific Product Offerings'] = 'Non-Billable: Plaque-Only'
    
    return df_result

def get_df(cursor,columns):
    print('Pulling in Case Data...')
    query = f"SELECT {', '.join(columns)} FROM sg_analytics_schema.case_submissions_sga where regulatory_region IN ('US', 'NORTH_AMERICA');"
    df_redshift = fetch_dataframe_in_chunks(cursor, query)
    print('Pulling SFDC Account Data...')
    sfdc_account = fetch_dataframe(cursor, "SELECT * FROM sg_analytics_schema.sfdc_accounts;")
    print('Pulling in Funnel Data...')
    funnel = fetch_dataframe(cursor, "SELECT * FROM sg_analytics_schema.opportunity_funnel_sga_reports;")

    print('Merging...')
    df_redshift = df_redshift.merge(
        sfdc_account[[
            "id", "plaque_claims_submitting_start_date__c",
            "plaque_claims_submitting_end_date__c", "plaque_pace_program_start_date__c",
            "plaque_pace_program_end_date__c"
        ]],
        left_on="salesforce_id",
        right_on="id",
        how="left"
    )

    print('Changing date formats...')
    df = change_date_format(df_redshift)

    print('Adding custom fields...')
    df['total_commercial'] = (df['stack'] == 'prod01') & (df['case_type'] == 'COMMERCIAL')
    df['latest_submission'] = (df['uuid'] == df['canonical_data_id'])
    df["is_perform_ffrct_enabled"] = df["order_is_active"].notna()
    df['is_ordered'] = (df["is_perform_ffrct_enabled"] & df['ordered_at'].notna()) | (~df["is_perform_ffrct_enabled"])
    df['revenue_generating'] = (
        df['total_commercial'] & 
        (df['stack'] == "prod01") & 
        (df['billing_type'] != "FREE") & 
        (df['case_state'] == "COMPLETED") & 
        (df['is_ordered'])
    )
    df['billing_timestamp_local'] = np.where(
        df["is_perform_ffrct_enabled"],
        df["ordered_at_local"],
        df["terminal_state_timestamp_local"]
    )

    billing_dt = pd.to_datetime(df["billing_timestamp_local"])
    created_dt = pd.to_datetime(df["created_at_local"])

    df["month_billing_timestamp_local"] = billing_dt.dt.month
    df["year_billing_timestamp_local"] = billing_dt.dt.year
    df["month_created_timestamp_local"] = created_dt.dt.month
    df["year_created_timestamp_local"] = created_dt.dt.year

    df = create_plaque_billing_category(df)
    df = create_product_offerings(df)

    return df

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
    
    # Base filter
    ffrct_filter = (
        (df['total_commercial'] == True) & 
        (df['latest_submission'] == True) & 
        (df['case_state'].isin(['COMPLETED', 'RCAG_HOLDING', 'RETURNED'])) & 
        (df['Case-Specific Product Offerings'].isin(['CCTA/FFRct Only', 'Billable: Plaque + FFRct', 'Non-Billable: Plaque + FFrct'])) & 
        (df['revenue_generating'] == True)
    )
    
    # Add date filters if provided
    if start_year is not None:
        if start_month is not None:
            # Filter by year and month
            ffrct_filter = ffrct_filter & (
                (df['year_billing_timestamp_local'] > start_year) |
                ((df['year_billing_timestamp_local'] == start_year) & 
                 (df['month_billing_timestamp_local'] >= start_month))
            )
        else:
            # Filter by year only
            ffrct_filter = ffrct_filter & (df['year_billing_timestamp_local'] >= start_year)
    
    if end_year is not None:
        if end_month is not None:
            # Filter by year and month
            ffrct_filter = ffrct_filter & (
                (df['year_billing_timestamp_local'] < end_year) |
                ((df['year_billing_timestamp_local'] == end_year) & 
                 (df['month_billing_timestamp_local'] <= end_month))
            )
        else:
            # Filter by year only
            ffrct_filter = ffrct_filter & (df['year_billing_timestamp_local'] <= end_year)
    
    filtered_df = df[ffrct_filter]
    
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
    
    # Base filter
    ccta_filter = (
        (df['total_commercial'] == True) & 
        (df['latest_submission'] == True) &
        (df['case_state'].isin(['COMPLETED', 'RCAG_HOLDING', 'RETURNED']))
    )
    
    # Add date filters if provided
    if start_year is not None:
        if start_month is not None:
            # Filter by year and month
            ccta_filter = ccta_filter & (
                (df['year_created_at_local'] > start_year) |
                ((df['year_created_at_local'] == start_year) & 
                 (df['month_created_at_local'] >= start_month))
            )
        else:
            # Filter by year only
            ccta_filter = ccta_filter & (df['year_created_at_local'] >= start_year)
    
    if end_year is not None:
        if end_month is not None:
            # Filter by year and month
            ccta_filter = ccta_filter & (
                (df['year_created_at_local'] < end_year) |
                ((df['year_created_at_local'] == end_year) & 
                 (df['month_created_at_local'] <= end_month))
            )
        else:
            # Filter by year only
            ccta_filter = ccta_filter & (df['year_created_at_local'] <= end_year)
    
    filtered_df = df[ccta_filter]
    
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