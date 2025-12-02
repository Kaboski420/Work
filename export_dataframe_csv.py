#!/usr/bin/env python3
"""
Create a DataFrame from ClickHouse with ALL columns and export to CSV.
"""

import clickhouse_connect
import csv
from datetime import datetime

# Connection details
HOST = 'clickhouse.bragmant.noooo.art'
PORT = 443
USER = 'dev2'
PASSWORD = '730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102'
DATABASE = 'crawler'
SECURE = True

def export_to_csv(table_name='ig_post', limit=10, output_file=None):
    """
    Query ClickHouse, create DataFrame with all columns, and export to CSV.
    
    Args:
        table_name: Name of the table to query
        limit: Number of rows to fetch
        output_file: Output CSV filename (default: auto-generate)
    """
    # Connect to ClickHouse
    print(f"🔗 Connecting to ClickHouse...")
    client = clickhouse_connect.create_client(
        host=HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        database=DATABASE,
        secure=SECURE
    )
    
    print(f"✅ Connected!")
    print(f"📊 Table: {table_name}")
    print(f"📏 Fetching: {limit} rows\n")
    
    # Query all columns
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    result = client.query(query)
    
    columns = result.column_names
    rows = result.result_rows
    
    print(f"✅ Retrieved {len(rows)} rows × {len(columns)} columns")
    print(f"📋 All {len(columns)} columns will be included\n")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{table_name}_sample_{limit}rows_{timestamp}.csv"
    
    # Export to CSV
    print(f"💾 Exporting to CSV: {output_file}")
    
    try:
        # Try using pandas for better CSV export
        import pandas as pd
        df = pd.DataFrame(rows, columns=columns)
        
        # Export to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Successfully exported using pandas!")
        print(f"   File: {output_file}")
        print(f"   Size: {len(rows)} rows × {len(columns)} columns")
        print(f"   All {len(columns)} columns included")
        
        # Show first few rows info
        print(f"\n📊 DataFrame Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"\n📋 Column names:")
        for i, col in enumerate(columns, 1):
            print(f"   {i:2}. {col}")
        
        return df, output_file
        
    except ImportError:
        # Fallback: Use standard csv module
        print(f"⚠️  pandas not available, using standard csv module...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(columns)
            
            # Write rows
            for row in rows:
                # Convert each value to string, handle None
                row_str = [str(val) if val is not None else '' for val in row]
                writer.writerow(row_str)
        
        print(f"✅ Successfully exported to CSV!")
        print(f"   File: {output_file}")
        print(f"   Rows: {len(rows)}")
        print(f"   Columns: {len(columns)}")
        print(f"\n📋 All {len(columns)} columns exported:")
        for i, col in enumerate(columns, 1):
            print(f"   {i:2}. {col}")
        
        return rows, output_file

def print_preview(csv_file, num_rows=3):
    """Print a preview of the CSV file."""
    print(f"\n📖 Preview of {csv_file} (first {num_rows} rows):")
    print("=" * 100)
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_file, nrows=num_rows)
        
        # Show all columns in preview
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.width', 200)
        
        print(f"\n{df.to_string()}")
        
    except ImportError:
        # Fallback: read CSV manually
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    print(f"Header ({len(row)} columns):")
                    print(", ".join(row[:10]) + "..." if len(row) > 10 else ", ".join(row))
                    print()
                    print(f"First {num_rows} rows:")
                elif i <= num_rows:
                    print(f"Row {i}: {', '.join(str(val)[:20] for val in row[:5])}...")
                else:
                    break

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    table_name = 'ig_post'
    limit = 10
    output_file = None
    
    if len(sys.argv) > 1:
        table_name = sys.argv[1]
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    print("=" * 100)
    print("EXPORT DATAFRAME TO CSV")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Table: {table_name}")
    print(f"  Rows: {limit}")
    print(f"  Include ALL columns: Yes")
    print()
    
    try:
        # Export to CSV
        result, csv_file = export_to_csv(table_name, limit, output_file)
        
        # Show preview
        print_preview(csv_file, num_rows=3)
        
        print("\n" + "=" * 100)
        print("✅ EXPORT COMPLETE!")
        print("=" * 100)
        print(f"\n📁 File saved: {csv_file}")
        print(f"\nYou can now:")
        print(f"  - Open it in Excel/LibreOffice")
        print(f"  - Load it in pandas: df = pd.read_csv('{csv_file}')")
        print(f"  - Use it in your analysis")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

