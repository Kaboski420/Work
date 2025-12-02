#!/usr/bin/env python3
"""
Explore ClickHouse crawler database - show tables and data.
"""

import clickhouse_connect
import json

# Connect to ClickHouse
client = clickhouse_connect.create_client(
    host='clickhouse.bragmant.noooo.art',
    port=443,
    username='dev2',
    password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
    database='crawler',
    secure=True
)

print("=" * 80)
print("CLICKHOUSE DATABASE EXPLORATION")
print("=" * 80)
print(f"Connected! Ping: {client.ping()}")
print(f"ClickHouse version: {client.query('SELECT version()').result_rows[0][0]}\n")

# Try to show databases (may have permission issues)
print("=" * 80)
print("DATABASES:")
print("=" * 80)
try:
    databases = client.query("SHOW DATABASES")
    for db in databases.result_rows:
        print(f"  • {db[0]}")
    if not databases.result_rows:
        print("  (no databases visible with current permissions)")
except Exception as e:
    print(f"  Could not list databases: {e}")

# Show current database
print(f"\nCurrent database: crawler\n")

# List tables
print("=" * 80)
print("TABLES IN 'crawler' DATABASE:")
print("=" * 80)
try:
    tables = client.query("SHOW TABLES")
    if tables.result_rows:
        for table in tables.result_rows:
            print(f"  • {table[0]}")
    else:
        print("  (no tables found)")
except Exception as e:
    print(f"  Error listing tables: {e}")
    print("  Trying alternative query...")
    try:
        # Try system.tables
        tables = client.query("SELECT name FROM system.tables WHERE database = 'crawler'")
        if tables.result_rows:
            for table in tables.result_rows:
                print(f"  • {table[0]}")
        else:
            print("  (no tables found via system.tables)")
    except Exception as e2:
        print(f"  Alternative query also failed: {e2}")

# If we have tables, show details
print("\n" + "=" * 80)
print("TABLE DETAILS:")
print("=" * 80)

try:
    # Try to get tables from system schema
    tables_query = client.query("""
        SELECT name, engine, total_rows, total_bytes
        FROM system.tables 
        WHERE database = 'crawler'
        ORDER BY name
    """)
    
    if tables_query.result_rows:
        for row in tables_query.result_rows:
            table_name, engine, rows, bytes = row
            rows_str = f"{rows:,}" if rows else "N/A"
            bytes_str = f"{bytes / (1024*1024):.2f} MB" if bytes else "N/A"
            print(f"\n📊 Table: {table_name}")
            print(f"   Engine: {engine}")
            print(f"   Rows: {rows_str}")
            print(f"   Size: {bytes_str}")
            
            # Show schema
            try:
                schema = client.query(f"DESCRIBE TABLE {table_name}")
                print(f"   Columns ({len(schema.result_rows)}):")
                has_raw_json = False
                for col_row in schema.result_rows:
                    col_name, col_type = col_row[0], col_row[1]
                    print(f"     - {col_name:30} {col_type}")
                    if col_name == 'raw_json':
                        has_raw_json = True
                
                if has_raw_json:
                    print(f"   ✅ Has 'raw_json' field (API request table)")
                
                # Show sample data
                try:
                    sample = client.query(f"SELECT * FROM {table_name} LIMIT 3")
                    if sample.result_rows:
                        print(f"   Sample rows:")
                        for i, row in enumerate(sample.result_rows, 1):
                            print(f"     Row {i}:")
                            for col_name, val in zip(sample.column_names, row):
                                val_str = str(val)
                                if len(val_str) > 150:
                                    val_str = val_str[:150] + "..."
                                # Pretty print raw_json if present
                                if col_name == 'raw_json' and val_str and val_str != "''":
                                    try:
                                        if isinstance(val, str):
                                            parsed = json.loads(val)
                                            val_str = json.dumps(parsed, indent=2)[:200] + "..."
                                    except:
                                        pass
                                print(f"       {col_name}: {val_str}")
                except Exception as e:
                    print(f"   Could not get sample data: {e}")
                    
            except Exception as e:
                print(f"   Could not get schema: {e}")
                
    else:
        print("\nNo tables found in system.tables")
        
        # Try to query common table names
        print("\nTrying to query common table patterns...")
        common_names = ['requests', 'api_requests', 'data', 'logs', 'events', 'crawler', 'crawls']
        for name in common_names:
            try:
                count = client.query(f"SELECT count() FROM {name}").result_rows[0][0]
                print(f"  ✅ Found table '{name}' with {count:,} rows")
            except:
                pass
                
except Exception as e:
    print(f"Error querying tables: {e}")
    print("\nTrying direct queries on common table names...")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)

