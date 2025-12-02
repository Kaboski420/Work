#!/usr/bin/env python3
"""
Get a pandas DataFrame from ClickHouse database.
Returns a DataFrame with first 10 rows that you can use directly.

Usage:
    from get_dataframe import get_sample_dataframe
    df = get_sample_dataframe('ig_post', limit=10)
    print(df.head())
"""

import clickhouse_connect

def get_sample_dataframe(table_name='ig_post', limit=10):
    """
    Get a pandas DataFrame from ClickHouse table.
    
    Args:
        table_name: Name of the table (default: 'ig_post')
        limit: Number of rows to fetch (default: 10)
        
    Returns:
        pandas.DataFrame with the queried data
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
    
    # Query data
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    # Try to create pandas DataFrame
    try:
        import pandas as pd
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        return df
    except ImportError:
        print("⚠️  pandas not available. Returning raw data as dict.")
        # Return as list of dictionaries
        return [
            dict(zip(result.column_names, row))
            for row in result.result_rows
        ]

if __name__ == "__main__":
    import sys
    
    table_name = sys.argv[1] if len(sys.argv) > 1 else 'ig_post'
    
    print("=" * 80)
    print("GETTING SAMPLE DATAFRAME")
    print("=" * 80)
    
    # Get DataFrame
    df = get_sample_dataframe(table_name, limit=10)
    
    # Check type and display
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            print(f"\n✅ Created pandas DataFrame!")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
            
            # Display first 10 rows
            print("=" * 80)
            print("FIRST 10 ROWS:")
            print("=" * 80)
            
            # Show key columns
            key_cols = ['post_id', 'owner_ig_username', 'caption', 'like_count', 
                       'comment_count', 'is_video', 'post_type']
            available_cols = [col for col in key_cols if col in df.columns]
            
            if available_cols:
                print(df[available_cols].head(10).to_string())
            else:
                print(df.head(10).to_string())
            
            print("\n" + "=" * 80)
            print("✅ DataFrame ready to use!")
            print("=" * 80)
            print("\nThe DataFrame 'df' is available with these operations:")
            print("  - df.head()")
            print("  - df.columns")
            print("  - df.describe()")
            print("  - df['like_count']")
            print("  - df[df['is_video'] == True]")
        else:
            # List of dicts
            print(f"\n✅ Created data structure (list of dictionaries)")
            print(f"   Rows: {len(df)}\n")
            print("First row:")
            if df:
                for key, value in list(df[0].items())[:10]:
                    print(f"  {key}: {value}")
    except ImportError:
        print(f"\n✅ Created data structure (list of dictionaries)")
        print(f"   Rows: {len(df)}")

