#!/usr/bin/env python3
"""
Create a sample DataFrame from ClickHouse database and display first 10 rows.
"""

import clickhouse_connect
import pandas as pd
from src.config import settings

def get_sample_dataframe(table_name='ig_post', limit=10):
    """
    Query ClickHouse and return a pandas DataFrame.
    
    Args:
        table_name: Name of the table to query
        limit: Number of rows to fetch
        
    Returns:
        pandas DataFrame
    """
    # Connect to ClickHouse
    client = clickhouse_connect.create_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_db,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        secure=settings.clickhouse_secure
    )
    
    print(f"🔗 Connected to ClickHouse: {settings.clickhouse_host}")
    print(f"📊 Querying table: {table_name}")
    print(f"📏 Limit: {limit} rows\n")
    
    # Query the table
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    
    return df

def print_dataframe_info(df, table_name):
    """Print DataFrame information."""
    print("=" * 80)
    print(f"DATAFRAME: {table_name.upper()}")
    print("=" * 80)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")
    
    print("\n" + "=" * 80)
    print("FIRST 10 ROWS:")
    print("=" * 80)
    
    # Display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print("\n", df.head(10))
    
    print("\n" + "=" * 80)
    print("DATA TYPES:")
    print("=" * 80)
    print(df.dtypes)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (for numeric columns):")
    print("=" * 80)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found")

if __name__ == "__main__":
    import sys
    
    # Default table
    table_name = 'ig_post'
    
    # Allow table name as argument
    if len(sys.argv) > 1:
        table_name = sys.argv[1]
    
    try:
        # Get DataFrame
        df = get_sample_dataframe(table_name, limit=10)
        
        # Print information
        print_dataframe_info(df, table_name)
        
        print("\n" + "=" * 80)
        print("✅ Successfully created DataFrame!")
        print("=" * 80)
        print(f"\nYou can now use the DataFrame 'df' with {len(df)} rows and {len(df.columns)} columns.")
        print("\nExample usage:")
        print("  df.head()")
        print("  df.columns")
        print("  df.describe()")
        print(f"  df['{df.columns[0] if len(df.columns) > 0 else 'column_name'}']")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

