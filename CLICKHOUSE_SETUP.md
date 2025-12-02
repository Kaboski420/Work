# ClickHouse Database Connection Setup

This document explains how to connect to the ClickHouse crawler database.

## Connection Details

- **Host**: `clickhouse.bragmant.noooo.art`
- **Port**: `443` (SSL/HTTPS)
- **Protocol**: HTTP (not native ClickHouse protocol)
- **Username**: `dev2`
- **Password**: `730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102` (SHA256 hash)
- **Database**: `crawler`
- **SSL**: Required (secure=true)

**Important**: The password must be the SHA256 hash, not the plain text password!

## Quick Start

### Option 1: Use the Connection Script

Run the provided connection script:

```bash
python connect_clickhouse.py
```

This will:
- Test the connection
- List all databases
- List all tables
- Find tables with `raw_json` field (API request tables)
- Show sample data

### Option 2: Set Environment Variables

Create a `.env` file in the project root (or set environment variables):

```bash
export CLICKHOUSE_HOST=clickhouse.bragmant.noooo.art
export CLICKHOUSE_PORT=443
export CLICKHOUSE_USER=dev2
export CLICKHOUSE_PASSWORD=730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102
export CLICKHOUSE_SECURE=true
export CLICKHOUSE_DB=crawler
```

**Note**: The password must be the SHA256 hash, not the plain text password.

Then use the configuration in your code:

```python
from src.config import settings
from src.utils.timeseries import TimeSeriesService

# TimeSeriesService will automatically use settings from .env
timeseries = TimeSeriesService(
    host=settings.clickhouse_host,
    port=settings.clickhouse_port,
    database=settings.clickhouse_db,
    username=settings.clickhouse_user,
    password=settings.clickhouse_password,
    secure=settings.clickhouse_secure
)
```

### Option 3: Direct Python Connection

```python
import clickhouse_connect

# Note: password must be SHA256 hash
client = clickhouse_connect.create_client(
    host='clickhouse.bragmant.noooo.art',
    port=443,
    username='dev2',
    password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
    database='crawler',
    secure=True
)

# Test connection
result = client.query("SELECT version()")
print(f"ClickHouse version: {result.result_rows[0][0]}")

# List databases
result = client.query("SHOW DATABASES")
print("Databases:", [row[0] for row in result.result_rows])

# List tables
result = client.query("SHOW TABLES")
print("Tables:", [row[0] for row in result.result_rows])
```

## Exploring the Database

### Common Queries

```python
# List all databases
client.query("SHOW DATABASES")

# Switch to a database
client.command("USE database_name")

# List tables in current database
client.query("SHOW TABLES")

# Describe table structure
client.query("DESCRIBE TABLE table_name")

# Count rows in a table
client.query("SELECT count() FROM table_name")

# Query data (tables with raw_json field)
client.query("SELECT * FROM table_name LIMIT 10")

# Query raw JSON data
client.query("SELECT raw_json FROM table_name WHERE condition LIMIT 10")
```

### Finding API Request Tables

Tables related to API requests have a `raw_json` field containing the raw JSON from the API. To find them:

```python
# Get all tables
tables = client.query("SHOW TABLES")

for table in tables.result_rows:
    table_name = table[0]
    # Check if table has raw_json field
    schema = client.query(f"DESCRIBE TABLE {table_name}")
    field_names = [row[0] for row in schema.result_rows]
    
    if 'raw_json' in field_names:
        print(f"Found API table: {table_name}")
```

## Troubleshooting

### Authentication Failed

If you get an authentication error:
1. **User might not be created yet**: Mohamed mentioned it takes a few minutes to create credentials. Wait a bit and try again.
2. **Check credentials**: Verify the username and password are correct.
3. **Check connection**: Ensure you can reach the host on port 443.

### Connection Timeout

- Check your network connection
- Verify the hostname is correct
- Check if firewall is blocking port 443

### SSL Certificate Issues

If you encounter SSL certificate errors, you might need to disable certificate verification (not recommended for production):

```python
client = clickhouse_connect.get_client(
    host=HOST,
    port=PORT,
    username=USER,
    password=PASSWORD,
    secure=True,
    verify=False  # Disable SSL verification (not recommended)
)
```

## Notes

- The database uses HTTP protocol over SSL (port 443), not the native ClickHouse protocol
- Each API request table has a `raw_json` field with the raw JSON from the API
- Use regular SQL queries to interact with the database
- The `clickhouse-connect` library handles the HTTP/HTTPS connection automatically

## Dependencies

Make sure `clickhouse-connect` is installed:

```bash
pip install clickhouse-connect
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

