#!/usr/bin/env python3
"""
Test script to connect to ClickHouse crawler database and explore available tables.

This script will:
1. Test the connection to the ClickHouse database
2. List all available databases
3. List all tables in each database
4. Show schema for tables with 'raw_json' field (API request tables)
"""

import logging
import sys
from src.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import clickhouse_connect
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    logger.error("clickhouse_connect not installed. Install it with: pip install clickhouse-connect")
    sys.exit(1)


def test_connection():
    """Test connection to ClickHouse database."""
    logger.info("=" * 80)
    logger.info("Testing ClickHouse Connection")
    logger.info("=" * 80)
    logger.info(f"Host: {settings.clickhouse_host}")
    logger.info(f"Port: {settings.clickhouse_port}")
    logger.info(f"Secure: {settings.clickhouse_secure}")
    logger.info(f"User: {settings.clickhouse_user}")
    logger.info(f"Database: {settings.clickhouse_db}")
    logger.info("=" * 80)
    
    try:
        client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_db,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            secure=settings.clickhouse_secure
        )
        
        # Test connection with a simple query
        result = client.query("SELECT version()")
        version = result.result_rows[0][0]
        logger.info(f"✅ Connection successful! ClickHouse version: {version}")
        return client
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return None


def list_databases(client):
    """List all available databases."""
    logger.info("\n" + "=" * 80)
    logger.info("Available Databases")
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
    """List all tables in the specified database or current database."""
    logger.info("\n" + "=" * 80)
    if database:
        logger.info(f"Tables in database: {database}")
        # Switch to the database
        client.command(f"USE {database}")
    else:
        logger.info("Tables in current database")
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


def show_table_schema(client, table_name):
    """Show the schema for a specific table."""
    try:
        result = client.query(f"DESCRIBE TABLE {table_name}")
        logger.info(f"\n  Schema for '{table_name}':")
        logger.info("  " + "-" * 76)
        for row in result.result_rows:
            field_name, field_type, default_type, default_expr, comment = row[:5]
            logger.info(f"    {field_name:30} {field_type:20} {comment or ''}")
        return result.result_rows
    except Exception as e:
        logger.error(f"  Error getting schema for {table_name}: {e}")
        return None


def find_raw_json_tables(client, database=None):
    """Find tables that have a 'raw_json' field (API request tables)."""
    logger.info("\n" + "=" * 80)
    logger.info("Finding tables with 'raw_json' field (API request tables)")
    logger.info("=" * 80)
    
    if database:
        client.command(f"USE {database}")
    
    try:
        tables = list_tables(client)
        raw_json_tables = []
        
        for table in tables:
            try:
                schema = show_table_schema(client, table)
                if schema:
                    field_names = [row[0] for row in schema]
                    if 'raw_json' in field_names:
                        raw_json_tables.append(table)
                        logger.info(f"\n  ✅ Found: {table} has 'raw_json' field")
            except Exception as e:
                logger.debug(f"  Could not check {table}: {e}")
                continue
        
        if not raw_json_tables:
            logger.info("\n  (no tables with 'raw_json' field found)")
        else:
            logger.info(f"\n  Total tables with 'raw_json': {len(raw_json_tables)}")
        
        return raw_json_tables
        
    except Exception as e:
        logger.error(f"Error finding raw_json tables: {e}")
        return []


def show_table_sample(client, table_name, limit=5):
    """Show a sample of data from a table."""
    try:
        result = client.query(f"SELECT * FROM {table_name} LIMIT {limit}")
        logger.info(f"\n  Sample data from '{table_name}' (first {limit} rows):")
        logger.info("  " + "-" * 76)
        
        if result.result_rows:
            # Show column names
            columns = [col[0] for col in result.column_names]
            logger.info(f"  Columns: {', '.join(columns)}")
            logger.info("")
            
            # Show a few sample rows
            for i, row in enumerate(result.result_rows[:limit], 1):
                logger.info(f"  Row {i}:")
                for col, val in zip(columns, row):
                    # Truncate long values
                    val_str = str(val)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    logger.info(f"    {col}: {val_str}")
                logger.info("")
        else:
            logger.info("  (table is empty)")
            
    except Exception as e:
        logger.error(f"  Error getting sample from {table_name}: {e}")


def main():
    """Main function to test connection and explore the database."""
    logger.info("Starting ClickHouse database exploration...")
    
    # Test connection
    client = test_connection()
    if not client:
        logger.error("Failed to connect. Exiting.")
        sys.exit(1)
    
    # List databases
    databases = list_databases(client)
    
    # Explore each database
    for db in databases:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Exploring database: {db}")
        logger.info("=" * 80)
        
        # Switch to database
        try:
            client.command(f"USE {db}")
        except Exception as e:
            logger.warning(f"Could not switch to database {db}: {e}")
            continue
        
        # List tables
        tables = list_tables(client)
        
        # Show schema for each table
        if tables:
            for table in tables:
                show_table_schema(client, table)
        
        # Find tables with raw_json
        if db == settings.clickhouse_db or db == 'default':
            raw_json_tables = find_raw_json_tables(client, db)
            
            # Show sample data for raw_json tables
            for table in raw_json_tables[:3]:  # Limit to first 3 tables
                show_table_sample(client, table, limit=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("Exploration complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

