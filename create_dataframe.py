#!/usr/bin/env python3
"""
Create a pandas DataFrame from ClickHouse database.
This script queries the database and creates a DataFrame with the first 10 rows.
"""

import clickhouse_connect

# Connection details
HOST = 'clickhouse.bragmant.noooo.art'
PORT = 443
USER = 'dev2'
PASSWORD = '730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102'
DATABASE = 'crawler'
SECURE = True

def create_dataframe(table_name='ig_post', limit=10):
    """
    Create a pandas DataFrame from ClickHouse table.
    
    Returns:
        pandas.DataFrame or None if pandas is not available
    """
    # Connect
    client = clickhouse_connect.create_client(
        host=HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        database=DATABASE,
        secure=SECURE
    )
    
    print(f"✅ Connected to ClickHouse")
    print(f"📊 Table: {table_name}")
    print(f"📏 Rows: {limit}\n")
    
    # Query
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    # Create DataFrame
    try:
        import pandas as pd
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        print(f"✅ Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error creating DataFrame: {e}")
        print("\nReturning raw data instead...")
        return {
            'columns': result.column_names,
            'rows': result.result_rows,
            'shape': (len(result.result_rows), len(result.column_names))
        }

def print_dataframe(df, table_name='ig_post'):
    """Print DataFrame nicely."""
    import pandas as pd
    
    print("\n" + "=" * 100)
    print(f"DATAFRAME: {table_name.upper()}")
    print("=" * 100)
    
    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # List columns
    print(f"\n📋 Columns ({len(df.columns)}):")
    print("-" * 100)
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        print(f"  {i:2}. {col:45} {dtype}")
    
    # Display first 10 rows with better formatting
    print("\n" + "=" * 100)
    print("FIRST 10 ROWS:")
    print("=" * 100)
    
    # Set display options
    pd.set_option('display.max_columns', 20)  # Show first 20 columns
    pd.set_option('display.max_rows', 12)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 40)
    
    print("\n", df)
    
    # Key columns summary
    key_cols = ['post_id', 'owner_ig_username', 'like_count', 'comment_count', 
                'is_video', 'post_type', 'caption']
    available_key_cols = [col for col in key_cols if col in df.columns]
    
    if available_key_cols:
        print("\n" + "=" * 100)
        print("KEY COLUMNS SUMMARY:")
        print("=" * 100)
        print(df[available_key_cols])
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print("\n" + "=" * 100)
        print("NUMERIC COLUMNS SUMMARY:")
        print("=" * 100)
        print(df[numeric_cols].describe())

if __name__ == "__main__":
    import sys
    
    table_name = sys.argv[1] if len(sys.argv) > 1 else 'ig_post'
    
    # Create DataFrame
    df = create_dataframe(table_name, limit=10)
    
    if df is not None:
        # Check if it's a pandas DataFrame
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                print_dataframe(df, table_name)
                print("\n" + "=" * 100)
                print("✅ DataFrame ready to use!")
                print("=" * 100)
                print(f"\nThe DataFrame 'df' has {len(df)} rows and {len(df.columns)} columns.")
                print("\nYou can now use it in your code or save it:")
                print("  df.to_csv('sample_data.csv')")
                print("  df.to_json('sample_data.json')")
            else:
                # Raw data format
                print("\n📊 Data retrieved:")
                print(f"  Columns: {df['shape'][1]}")
                print(f"  Rows: {df['shape'][0]}")
                print("\nFirst few columns:", df['columns'][:10])
        except ImportError:
            print("\n📊 Data retrieved (pandas not available):")
            print(f"  Columns: {len(df['columns'])}")
            print(f"  Rows: {len(df['rows'])}")
    else:
        print("❌ Failed to create DataFrame")

