#!/usr/bin/env python3
"""
Create a sample pandas DataFrame from ClickHouse and display it.
Returns a DataFrame with the first 10 rows.
"""

import clickhouse_connect

def create_sample_dataframe(table_name='ig_post', limit=10):
    """
    Create a pandas DataFrame from ClickHouse table.
    
    Args:
        table_name: Name of the table
        limit: Number of rows to fetch
        
    Returns:
        pandas DataFrame
    """
    # Connect to ClickHouse
    client = clickhouse_connect.create_client(
        host='clickhouse.bragmant.noooo.art',
        port=443,
        username='dev2',
        password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
        database='crawler',
        secure=True
    )
    
    print(f"🔗 Connected to ClickHouse")
    print(f"📊 Querying: {table_name} (limit: {limit})\n")
    
    # Query data
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    # Try to create DataFrame with pandas
    try:
        import pandas as pd
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        return df
    except ImportError:
        print("⚠️  pandas not available. Please install: pip install pandas")
        return None

def print_dataframe(df, table_name='ig_post'):
    """Print DataFrame information and first 10 rows."""
    if df is None:
        print("❌ No DataFrame available")
        return
    
    print("=" * 100)
    print(f"PANDAS DATAFRAME: {table_name.upper()}")
    print("=" * 100)
    
    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n📋 Columns ({len(df.columns)}):")
    print("-" * 100)
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        print(f"  {i:2}. {col:40} ({dtype})")
    
    print("\n" + "=" * 100)
    print("FIRST 10 ROWS:")
    print("=" * 100)
    
    # Configure pandas display options
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 50)
    
    print("\n", df.head(10))
    
    # Show data types
    print("\n" + "=" * 100)
    print("DATA TYPES:")
    print("=" * 100)
    print(df.dtypes.to_string())
    
    # Show summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS (numeric columns):")
        print("=" * 100)
        print(df[numeric_cols].describe().to_string())
    
    # Show memory usage
    print("\n" + "=" * 100)
    print("MEMORY INFO:")
    print("=" * 100)
    memory_kb = df.memory_usage(deep=True).sum() / 1024
    print(f"Total memory: {memory_kb:.2f} KB")

if __name__ == "__main__":
    import sys
    
    table_name = sys.argv[1] if len(sys.argv) > 1 else 'ig_post'
    
    try:
        # Create DataFrame
        df = create_sample_dataframe(table_name, limit=10)
        
        if df is not None:
            # Print information
            print_dataframe(df, table_name)
            
            print("\n" + "=" * 100)
            print("✅ DataFrame created successfully!")
            print("=" * 100)
            print(f"\nYou can now use the DataFrame:")
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {len(df.columns)}")
            print(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print("\nExample operations:")
            print("  df.head()")
            print("  df.columns.tolist()")
            print("  df.describe()")
            print("  df['like_count']")
            print("  df[df['is_video'] == True]")
        else:
            print("❌ Could not create DataFrame")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

