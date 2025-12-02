#!/usr/bin/env python3
"""
Query a specific ClickHouse table by name.

Usage:
    python query_table.py <table_name> [--limit N] [--columns]
    
Examples:
    python query_table.py my_table
    python query_table.py my_table --limit 10
    python query_table.py my_table --columns  # Show column names only
"""

import clickhouse_connect
import sys
import json
import argparse

# Connection config
HOST = 'clickhouse.bragmant.noooo.art'
PORT = 443
USER = 'dev2'
PASSWORD = '730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102'
DATABASE = 'crawler'
SECURE = True


def connect():
    """Connect to ClickHouse."""
    return clickhouse_connect.create_client(
        host=HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        database=DATABASE,
        secure=SECURE
    )


def show_table_schema(client, table_name):
    """Show table schema."""
    try:
        result = client.query(f"DESCRIBE TABLE {table_name}")
        print(f"\n📋 Schema for table '{table_name}':")
        print("=" * 80)
        for row in result.result_rows:
            col_name, col_type, default_type, default_expr, comment = row[:5]
            print(f"  {col_name:40} {col_type:30} {comment or ''}")
        return [row[0] for row in result.result_rows]
    except Exception as e:
        print(f"❌ Error getting schema: {e}")
        return None


def query_table(client, table_name, limit=10, show_columns_only=False):
    """Query a table."""
    print(f"\n🔍 Querying table '{table_name}'...")
    
    # Get schema first
    columns = show_table_schema(client, table_name)
    if not columns:
        return
    
    # Check for raw_json field
    has_raw_json = 'raw_json' in columns
    if has_raw_json:
        print("\n✅ This table has a 'raw_json' field (API request table)")
    
    if show_columns_only:
        return
    
    # Get row count
    try:
        count_result = client.query(f"SELECT count() FROM {table_name}")
        total_rows = count_result.result_rows[0][0]
        print(f"\n📊 Total rows: {total_rows:,}")
    except Exception as e:
        print(f"⚠️  Could not get row count: {e}")
        total_rows = None
    
    # Query data
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        result = client.query(query)
        
        if not result.result_rows:
            print(f"\n⚠️  Table is empty")
            return
        
        print(f"\n📄 Sample data (showing {len(result.result_rows)} of {total_rows or '?'} rows):")
        print("=" * 80)
        
        for i, row in enumerate(result.result_rows, 1):
            print(f"\nRow {i}:")
            print("-" * 80)
            for col_name, val in zip(result.column_names, row):
                val_str = str(val)
                
                # Pretty print raw_json
                if col_name == 'raw_json' and val and val_str != "''":
                    try:
                        if isinstance(val, str):
                            parsed = json.loads(val)
                            val_str = json.dumps(parsed, indent=2)
                            if len(val_str) > 500:
                                val_str = val_str[:500] + "\n... (truncated)"
                    except:
                        pass
                
                # Truncate long values
                if len(val_str) > 300:
                    val_str = val_str[:300] + "..."
                
                print(f"  {col_name:30}: {val_str}")
    
    except Exception as e:
        print(f"❌ Error querying table: {e}")


def main():
    parser = argparse.ArgumentParser(description='Query a ClickHouse table')
    parser.add_argument('table_name', help='Name of the table to query')
    parser.add_argument('--limit', type=int, default=10, help='Number of rows to show (default: 10)')
    parser.add_argument('--columns', action='store_true', help='Show column names only')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CLICKHOUSE TABLE QUERY TOOL")
    print("=" * 80)
    print(f"Database: {DATABASE}")
    print(f"Table: {args.table_name}")
    print("=" * 80)
    
    try:
        client = connect()
        print(f"✅ Connected! (Ping: {client.ping()})")
        
        query_table(client, args.table_name, limit=args.limit, show_columns_only=args.columns)
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_table.py <table_name> [--limit N] [--columns]")
        print("\nExample:")
        print("  python query_table.py my_table")
        print("  python query_table.py my_table --limit 5")
        print("  python query_table.py my_table --columns")
        sys.exit(1)
    main()

