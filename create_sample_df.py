#!/usr/bin/env python3
"""
Create a sample DataFrame from ClickHouse database and display first 10 rows.
"""

import clickhouse_connect
import sys

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available. Displaying as table format instead.")

def get_sample_data(table_name='ig_post', limit=10):
    """Query ClickHouse and return data."""
    # Connect using direct credentials
    client = clickhouse_connect.create_client(
        host='clickhouse.bragmant.noooo.art',
        port=443,
        username='dev2',
        password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
        database='crawler',
        secure=True
    )
    
    print(f"🔗 Connected to ClickHouse")
    print(f"📊 Querying table: {table_name}")
    print(f"📏 Fetching: {limit} rows\n")
    
    # Query the table
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    return result

def display_with_pandas(result):
    """Display results using pandas DataFrame."""
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    
    print("=" * 100)
    print(f"PANDAS DATAFRAME - {len(df)} rows × {len(df.columns)} columns")
    print("=" * 100)
    
    # Display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 60)
    
    print("\n📋 COLUMNS:")
    print("-" * 100)
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")
    
    print(f"\n📊 FIRST 10 ROWS:")
    print("=" * 100)
    print(df.head(10).to_string())
    
    print(f"\n📈 DATAFRAME INFO:")
    print("-" * 100)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:")
    print(df.dtypes.to_string())
    
    # Show numeric column statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\n📊 SUMMARY STATISTICS (numeric columns):")
        print("-" * 100)
        print(df[numeric_cols].describe().to_string())
    
    # Show memory usage
    print(f"\n💾 MEMORY USAGE:")
    print("-" * 100)
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    return df

def display_without_pandas(result):
    """Display results as formatted table without pandas."""
    columns = result.column_names
    rows = result.result_rows
    
    print("=" * 100)
    print(f"DATA TABLE - {len(rows)} rows × {len(columns)} columns")
    print("=" * 100)
    
    print("\n📋 COLUMNS:")
    print("-" * 100)
    for i, col in enumerate(columns, 1):
        print(f"  {i:2}. {col}")
    
    print(f"\n📊 FIRST 10 ROWS:")
    print("=" * 100)
    
    # Print header
    print(" | ".join(f"{col[:20]:20}" for col in columns[:8]))  # Show first 8 columns
    print("-" * 100)
    
    # Print rows (limited to first 8 columns for readability)
    for i, row in enumerate(rows[:10], 1):
        row_str = " | ".join(f"{str(val)[:20]:20}" if val is not None else f"{'None':20}" 
                            for val in row[:8])
        print(f"{i:2}. {row_str}")
    
    if len(columns) > 8:
        print(f"\n... (showing first 8 of {len(columns)} columns)")

if __name__ == "__main__":
    # Default table
    table_name = 'ig_post'
    
    # Allow table name as argument
    if len(sys.argv) > 1:
        table_name = sys.argv[1]
    
    try:
        # Get data from ClickHouse
        result = get_sample_data(table_name, limit=10)
        
        # Display based on pandas availability
        if HAS_PANDAS:
            df = display_with_pandas(result)
            
            print("\n" + "=" * 100)
            print("✅ DataFrame created successfully!")
            print("=" * 100)
            print(f"\nYou can use the DataFrame with:")
            print("  - df.shape: {df.shape}")
            print("  - df.columns: {list(df.columns)[:5]}...")
            print("  - df.head(10)")
            print("  - df.describe()")
        else:
            display_without_pandas(result)
            print("\n" + "=" * 100)
            print("✅ Data retrieved successfully!")
            print("=" * 100)
            print("\n💡 Install pandas for better DataFrame support:")
            print("   pip install pandas")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

