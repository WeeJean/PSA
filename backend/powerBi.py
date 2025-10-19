import os
import requests
import pandas as pd
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
# --- Configuration (Assumes Environment Variables are Set) ---
# NOTE: Replace os.getenv calls with your actual string values if not using environment variables.
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")  # Power BI Group ID
REPORT_ID = os.getenv("REPORT_ID")

# The name of the table you want to extract data from.
# You MUST change 'data' to the actual table name in your Power BI dataset.
# Common table names are 'Sales', 'FactInternetSales', or the name of your imported Excel sheet.
TARGET_TABLE_NAME = 'data' 

# --- Constants ---
AUTHORITY_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
RESOURCE_URL = "https://analysis.windows.net/powerbi/api/.default"
POWERBI_API_BASE = "https://api.powerbi.com/v1.0/myorg"

def get_access_token() -> Optional[str]:
    """Acquires an access token using Service Principal credentials."""
    print("1. Attempting to acquire access token...")
    
    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]):
        print("Error: Missing one or more required environment variables for authentication.")
        return None

    payload = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': RESOURCE_URL
    }

    try:
        response = requests.post(AUTHORITY_URL, data=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        token_data = response.json()
        print("Token acquired successfully.")
        return token_data.get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"Error during token acquisition: {e}")
        return None

def get_dataset_id(access_token: str) -> Optional[str]:
    """Fetches the Dataset ID associated with the Report ID."""
    print("2. Retrieving Dataset ID from Report metadata...")

    report_info_url = f"{POWERBI_API_BASE}/groups/{WORKSPACE_ID}/reports/{REPORT_ID}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(report_info_url, headers=headers)
        response.raise_for_status()
        report_data = response.json()
        dataset_id = report_data.get('datasetId')
        
        if dataset_id:
            print(f"   Dataset ID found: {dataset_id}")
            return dataset_id
        else:
            print("Error: 'datasetId' not found in the report metadata. Check REPORT_ID and permissions.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching report info: {e}")
        print("   Ensure the WORKSPACE_ID, REPORT_ID, and Service Principal permissions are correct.")
        return None

def get_report_data(access_token: str, dataset_id: str) -> Optional[pd.DataFrame]:
    """
    Executes a DAX query against the dataset and returns the result as a DataFrame.
    """
    print(f"3. Executing DAX query against Dataset ID: {dataset_id}")
    
    execute_query_url = f"{POWERBI_API_BASE}/groups/{WORKSPACE_ID}/datasets/{dataset_id}/executeQueries"
    
    # The DAX query to retrieve all rows from the specified table.
    # Note: Power BI REST API has limitations on query size and complexity. 
    # For very large datasets, you might need to implement pagination or filtering.
    dax_query = f"EVALUATE '{TARGET_TABLE_NAME}'"
    
    payload = {
        "queries": [
            {
                "query": dax_query
            }
        ],
        "serializerSettings": {
            "includeNulls": True 
        }
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(execute_query_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        query_result = response.json()
        
        # Parse the JSON result into a pandas DataFrame
        tables: List[Dict[str, Any]] = query_result.get('results', [{}])[0].get('tables', [{}])
        
        if not tables:
            print("Warning: Query returned no tables. Structure was unexpected.")
            return pd.DataFrame()

        # Safely get the first table element
        table_data = tables[0]

        rows = table_data.get('rows', [])
        
        if not rows:
            print("Warning: Query returned no data or the table structure was unexpected.")
            return pd.DataFrame()
        
        # Use .get('columns', []) to safely handle cases where the 'columns' key is missing 
        # or empty, preventing the 'columns' KeyError.
        columns = [col['name'] for col in table_data.get('columns', [])]
        
        # FALLBACK: If 'columns' array was empty or missing, infer column names from the 
        # keys of the first row dictionary.
        if not columns and rows:
            print("Note: Inferring column names from the first row keys.")
            columns = list(rows[0].keys())
            
        if not columns:
            print("Warning: Could not determine column names from response structure.")
            return pd.DataFrame()

        # Each row is a dictionary where keys are column names
        df = pd.DataFrame(rows, columns=columns)
            
        print(f"   Successfully retrieved {len(df)} rows.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error executing DAX query: {e}")
        print(f"   Response status: {response.status_code if 'response' in locals() else 'N/A'}")
        print("   Ensure the TARGET_TABLE_NAME is correct and the Service Principal has Dataset Read permissions.")
        # Print the error message from Power BI if available
        if 'response' in locals() and response.text:
            print(f"   API Error Details: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        return None

def main():
    """Main function to run the data extraction process."""
    
    # 0. Initial check for required inputs
    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID, WORKSPACE_ID, REPORT_ID]):
        print("\n--- ERROR ---")
        print("Please ensure the following environment variables are set or their values are hardcoded:")
        print("CLIENT_ID, CLIENT_SECRET, TENANT_ID, WORKSPACE_ID (Group ID), REPORT_ID.")
        print("Exiting...")
        return
    
    print("\n--- Power BI Data Extractor Initiated ---")
    print(f"Targeting Workspace: {WORKSPACE_ID}, Report: {REPORT_ID}, Table: '{TARGET_TABLE_NAME}'")

    # Step 1: Get Access Token
    token = get_access_token()
    if not token:
        print("Process aborted.")
        return

    # Step 2: Get Dataset ID
    dataset_id = get_dataset_id(token)
    if not dataset_id:
        print("Process aborted.")
        return

    # Step 3: Get Report Data
    df = get_report_data(token, dataset_id)

    if df is not None and not df.empty:
        print("\n--- Data Extraction Complete ---")
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns into a pandas DataFrame.")
      
        # Example of what you can do with the DataFrame
        print("\nFirst 5 rows of the data:")
        print(df.head())
        
        # To save the data to a CSV file:
        output_filename = f"{TARGET_TABLE_NAME}_data.csv"
        df.to_csv(output_filename, index=False)
        print(f"\nData saved successfully to {output_filename}")
        
    elif df is not None:
        print("\n--- Data Extraction Complete (Empty) ---")
        print("The process finished but the DataFrame is empty.")
    else:
        print("\n--- Data Extraction Failed ---")

if __name__ == "__main__":
    main()