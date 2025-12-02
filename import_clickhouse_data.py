#!/usr/bin/env python3
"""
Import first 100 rows from ClickHouse for testing.
Stores data locally for model training and testing.
"""

import clickhouse_connect
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def import_clickhouse_data(table_name='ig_post', limit=100):
    """
    Import data from ClickHouse table.
    
    Args:
        table_name: Name of the table (default: 'ig_post')
        limit: Number of rows to import (default: 100)
        
    Returns:
        pandas.DataFrame with the imported data
    """
    print("=" * 80)
    print(f"IMPORTING {limit} ROWS FROM CLICKHOUSE")
    print("=" * 80)
    
    # Connect to ClickHouse
    try:
        client = clickhouse_connect.create_client(
            host='clickhouse.bragmant.noooo.art',
            port=443,
            username='dev2',
            password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
            database='crawler',
            secure=True
        )
        print("✅ Connected to ClickHouse")
    except Exception as e:
        print(f"❌ Error connecting to ClickHouse: {e}")
        return None
    
    # Query data
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        print(f"📊 Querying: {query}")
        result = client.query(query)
        print(f"✅ Retrieved {len(result.result_rows)} rows")
    except Exception as e:
        print(f"❌ Error querying data: {e}")
        return None
    
    # Create pandas DataFrame
    try:
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        print(f"✅ Created DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Save to CSV
        output_dir = Path("data/clickhouse_imports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{table_name}_{limit}rows_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved to: {csv_path}")
        
        # Save metadata
        metadata = {
            "table_name": table_name,
            "rows_imported": len(df),
            "columns": list(df.columns),
            "timestamp": timestamp,
            "import_date": datetime.now().isoformat()
        }
        
        metadata_path = output_dir / f"{table_name}_{limit}rows_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to: {metadata_path}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nFirst few rows:")
        print(df.head(3).to_string())
        
        print("\n" + "=" * 80)
        print("✅ IMPORT COMPLETE")
        print("=" * 80)
        
        return df
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    table_name = sys.argv[1] if len(sys.argv) > 1 else 'ig_post'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    df = import_clickhouse_data(table_name, limit)
    
    if df is not None:
        print(f"\n✅ Successfully imported {len(df)} rows from {table_name}")
    else:
        print("\n❌ Import failed")
        sys.exit(1)

