#!/usr/bin/env python3
"""
Query ClickHouse and display first 10 rows as a sample DataFrame.
Works with or without pandas.
"""

import clickhouse_connect

def get_sample_data(table_name='ig_post', limit=10):
    """Query ClickHouse and return data."""
    client = clickhouse_connect.create_client(
        host='clickhouse.bragmant.noooo.art',
        port=443,
        username='dev2',
        password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
        database='crawler',
        secure=True
    )
    
    print(f"🔗 Connected to ClickHouse")
    print(f"📊 Table: {table_name}")
    print(f"📏 Fetching: {limit} rows\n")
    
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    return result.column_names, result.result_rows

def print_sample_dataframe(columns, rows, table_name='ig_post'):
    """Print data in DataFrame-like format."""
    print("=" * 120)
    print(f"SAMPLE DATAFRAME: {table_name.upper()}")
    print("=" * 120)
    print(f"\nShape: {len(rows)} rows × {len(columns)} columns")
    
    # Show all columns
    print(f"\n📋 All {len(columns)} Columns:")
    print("-" * 120)
    for i, col in enumerate(columns, 1):
        print(f"  {i:2}. {col}")
    
    # Select key columns to display
    key_cols = ['post_id', 'owner_ig_username', 'caption', 'like_count', 
                'comment_count', 'is_video', 'post_type', 'created_at']
    display_cols = [col for col in key_cols if col in columns]
    
    if not display_cols:
        display_cols = columns[:10]  # Show first 10 if key cols not found
    
    print(f"\n📊 First 10 Rows (showing {len(display_cols)} key columns):")
    print("=" * 120)
    
    # Print header
    header = " | ".join(f"{col:20}" for col in display_cols)
    print(header)
    print("-" * 120)
    
    # Print rows
    for i, row in enumerate(rows, 1):
        row_dict = dict(zip(columns, row))
        row_values = []
        for col in display_cols:
            val = row_dict.get(col, 'N/A')
            val_str = str(val)
            # Truncate long values
            if len(val_str) > 18:
                val_str = val_str[:15] + "..."
            row_values.append(f"{val_str:20}")
        print(f"{i:2}. {' | '.join(row_values)}")
    
    if len(columns) > len(display_cols):
        print(f"\n... (showing {len(display_cols)} of {len(columns)} total columns)")
    
    # Show statistics for numeric columns
    print(f"\n📈 Statistics (first 10 rows):")
    print("=" * 120)
    
    numeric_cols = ['like_count', 'comment_count', 'video_view_count', 
                   'follower_count', 'following_count']
    numeric_cols = [col for col in numeric_cols if col in columns]
    
    if numeric_cols:
        print(f"\nNumeric columns: {', '.join(numeric_cols)}")
        print("-" * 120)
        for col in numeric_cols:
            values = [row[columns.index(col)] for row in rows if row[columns.index(col)] is not None]
            if values:
                try:
                    numeric_vals = [float(v) for v in values if v != '']
                    if numeric_vals:
                        print(f"  {col:25}: min={min(numeric_vals)}, max={max(numeric_vals)}, "
                              f"mean={sum(numeric_vals)/len(numeric_vals):.1f}")
                except:
                    pass

def create_dict_dataframe(columns, rows):
    """Create a list of dictionaries (DataFrame-like structure)."""
    return [dict(zip(columns, row)) for row in rows]

if __name__ == "__main__":
    import sys
    
    table_name = sys.argv[1] if len(sys.argv) > 1 else 'ig_post'
    
    try:
        # Get data
        columns, rows = get_sample_data(table_name, limit=10)
        
        # Display
        print_sample_dataframe(columns, rows, table_name)
        
        # Create dict-based "DataFrame"
        df_dict = create_dict_dataframe(columns, rows)
        
        print("\n" + "=" * 120)
        print("✅ Sample DataFrame created!")
        print("=" * 120)
        print(f"\nData structure: {len(rows)} rows × {len(columns)} columns")
        print(f"\nTo use as a list of dictionaries:")
        print(f"  df = [dict(zip(columns, row)) for row in rows]")
        print(f"  # Access: df[0]['like_count']")
        
        # Example: show first row as dict
        if df_dict:
            print(f"\n📝 Example - First row as dictionary:")
            print("-" * 120)
            first_row = df_dict[0]
            for key in list(first_row.keys())[:10]:
                val = str(first_row[key])
                if len(val) > 60:
                    val = val[:57] + "..."
                print(f"  {key:30}: {val}")
            if len(first_row) > 10:
                print(f"  ... and {len(first_row) - 10} more fields")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

