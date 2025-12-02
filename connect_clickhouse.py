#!/usr/bin/env python3
"""
Quick script to connect to ClickHouse crawler database.

Usage:
    python connect_clickhouse.py
    
Or with environment variables:
    export CLICKHOUSE_HOST=clickhouse.bragmant.noooo.art
    export CLICKHOUSE_PORT=443
    export CLICKHOUSE_USER=dev2
    export CLICKHOUSE_PASSWORD=AEY3pWbQjbGHLrRAnVzzJXDuDGg1sUhdCdfkwQ0LRFNPHMOHee
    export CLICKHOUSE_SECURE=true
    python connect_clickhouse.py
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import clickhouse_connect
except ImportError:
    logger.error("clickhouse_connect not installed. Install it with: pip install clickhouse-connect")
    sys.exit(1)

# Connection parameters - can be overridden by environment variables
# NOTE: Password must be SHA256 hash, not plain text
HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse.bragmant.noooo.art")
PORT = int(os.getenv("CLICKHOUSE_PORT", "443"))
USER = os.getenv("CLICKHOUSE_USER", "dev2")
# SHA256 hash of the password: AEY3pWbQjbGHLrRAnVzzJXDuDGg1sUhdCdfkwQ0LRFNPHMOHee
PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102")
SECURE = os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true"
DATABASE = os.getenv("CLICKHOUSE_DB", "crawler")


def connect(database=None):
    """Connect to ClickHouse and return the client."""
    db = database or DATABASE
    logger.info(f"Connecting to ClickHouse...")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Secure: {SECURE}")
    logger.info(f"  User: {USER}")
    logger.info(f"  Database: {db if db else '(not specified)'}")
    
    try:
        # Use create_client (can also use get_client - both work)
        client = clickhouse_connect.create_client(
            host=HOST,
            port=PORT,
            database=db if db else DATABASE,
            username=USER,
            password=PASSWORD,
            secure=SECURE
        )
        
        # Test connection
        result = client.query("SELECT version()")
        version = result.result_rows[0][0]
        logger.info(f"✅ Connected successfully! ClickHouse version: {version}\n")
        return client
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Connection failed: {error_msg}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check your network connection")
        logger.error("  2. Verify credentials are correct")
        logger.error("  3. The user might not be created yet (Mohamed said it would take a few minutes)")
        logger.error("  4. Ensure clickhouse-connect is installed: pip install clickhouse-connect")
        
        # Try alternative connection methods
        if "Authentication failed" in error_msg or "AUTHENTICATION_FAILED" in error_msg:
            logger.error("\n💡 Authentication failed. Possible issues:")
            logger.error("   - User 'dev2' might not be created yet")
            logger.error("   - Password might be incorrect")
            logger.error("   - Try waiting a few minutes if the user was just created")
        
        return None


def list_databases(client):
    """List all databases."""
    logger.info("=" * 80)
    logger.info("Available Databases:")
    logger.info("=" * 80)
    
    try:
        result = client.query("SHOW DATABASES")
        databases = [row[0] for row in result.result_rows]
        
        for db in databases:
            logger.info(f"  • {db}")
        
        return databases
    except Exception as e:
        logger.error(f"Error listing databases: {e}")
        return []


def list_tables(client, database=None):
    """List tables in current or specified database."""
    if database:
        try:
            client.command(f"USE {database}")
        except Exception as e:
            logger.warning(f"Could not switch to {database}: {e}")
            return []
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Tables in database '{database or DATABASE}':")
    logger.info("=" * 80)
    
    try:
        result = client.query("SHOW TABLES")
        tables = [row[0] for row in result.result_rows]
        
        if not tables:
            logger.info("  (no tables found)")
        else:
            for table in tables:
                logger.info(f"  • {table}")
        
        return tables
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return []


def show_table_info(client, table_name):
    """Show schema and sample data for a table."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Table: {table_name}")
    logger.info("=" * 80)
    
    # Show schema
    try:
        result = client.query(f"DESCRIBE TABLE {table_name}")
        logger.info("\nSchema:")
        logger.info("-" * 80)
        for row in result.result_rows:
            field_name, field_type = row[0], row[1]
            logger.info(f"  {field_name:40} {field_type}")
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
    
    # Check for raw_json field
    try:
        result = client.query(f"DESCRIBE TABLE {table_name}")
        has_raw_json = any(row[0] == 'raw_json' for row in result.result_rows)
        if has_raw_json:
            logger.info("\n✅ This table has a 'raw_json' field (API request table)")
    except Exception:
        pass
    
    # Show row count
    try:
        result = client.query(f"SELECT count() FROM {table_name}")
        count = result.result_rows[0][0]
        logger.info(f"\nRow count: {count:,}")
    except Exception as e:
        logger.warning(f"Could not get row count: {e}")
    
    # Show sample data
    try:
        result = client.query(f"SELECT * FROM {table_name} LIMIT 1")
        if result.result_rows:
            logger.info("\nSample row (first row):")
            logger.info("-" * 80)
            columns = [col[0] for col in result.column_names]
            row = result.result_rows[0]
            
            for col, val in zip(columns, row):
                val_str = str(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                logger.info(f"  {col}: {val_str}")
    except Exception as e:
        logger.warning(f"Could not get sample data: {e}")


def find_raw_json_tables(client, database=None):
    """Find all tables with raw_json field."""
    if database:
        try:
            client.command(f"USE {database}")
        except Exception:
            pass
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Finding tables with 'raw_json' field (API request tables):")
    logger.info("=" * 80)
    
    try:
        tables = list_tables(client)
        raw_json_tables = []
        
        for table in tables:
            try:
                result = client.query(f"DESCRIBE TABLE {table}")
                field_names = [row[0] for row in result.result_rows]
                if 'raw_json' in field_names:
                    raw_json_tables.append(table)
                    logger.info(f"  ✅ {table}")
            except Exception:
                continue
        
        if not raw_json_tables:
            logger.info("  (no tables with 'raw_json' field found)")
        
        return raw_json_tables
    except Exception as e:
        logger.error(f"Error finding raw_json tables: {e}")
        return []


def main():
    """Main function."""
    client = connect()
    if not client:
        sys.exit(1)
    
    # List databases
    databases = list_databases(client)
    
    # Explore each database
    for db in databases:
        tables = list_tables(client, db)
        
        # Find and show info for raw_json tables
        if tables:
            raw_json_tables = find_raw_json_tables(client, db)
            
            # Show detailed info for first few raw_json tables
            for table in raw_json_tables[:5]:
                show_table_info(client, table)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Exploration complete!")
    logger.info("=" * 80)
    logger.info("\nTo query the database interactively, use:")
    logger.info(f"  client = clickhouse_connect.get_client(host='{HOST}', port={PORT}, username='{USER}', password='...', secure={SECURE})")
    logger.info("  result = client.query('YOUR SQL QUERY HERE')")


if __name__ == "__main__":
    main()

