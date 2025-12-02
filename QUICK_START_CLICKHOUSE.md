# Quick Start: ClickHouse Connection ✅

## Connection Status
✅ **Successfully Connected!**

The ClickHouse database is now accessible using:

## Quick Connection

```python
import clickhouse_connect

client = clickhouse_connect.create_client(
    host='clickhouse.bragmant.noooo.art',
    port=443,
    username='dev2',
    password='730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102',
    database='crawler',
    secure=True
)

# Test connection
print(f"Ping: {client.ping()}")
print(f"Version: {client.query('SELECT version()').result_rows[0][0]}")
```

## Important Notes

1. **Password**: Must use SHA256 hash (not plain text)
   - Hash: `730be85cdda226560007e8fe1ac01804f5136b7bfc1e97f5d67851c47e7e4102`
   - Original: `AEY3pWbQjbGHLrRAnVzzJXDuDGg1sUhdCdfkwQ0LRFNPHMOHee`

2. **Database**: Use `crawler` as the database name

3. **Method**: Use `create_client()` or `get_client()` (both work)

## Test Connection

Run:
```bash
python3 click_house.py
```

Or use the full exploration script:
```bash
python connect_clickhouse.py
```

## Configuration

The default configuration in `src/config.py` is now set to:
- Host: `clickhouse.bragmant.noooo.art`
- Port: `443`
- Database: `crawler`
- User: `dev2`
- Secure: `True`

## Querying Tables with raw_json

Tables related to API requests have a `raw_json` field. Example query:

```python
# Get sample data from a table with raw_json
result = client.query("SELECT * FROM table_name WHERE raw_json != '' LIMIT 10")

# Parse raw JSON
import json
for row in result.result_rows:
    raw_data = json.loads(row[raw_json_index])
    print(raw_data)
```

## Next Steps

1. Explore available tables (may require specific permissions)
2. Query tables with `raw_json` field for API request data
3. Integrate with your virality prediction algorithms

See `CLICKHOUSE_SETUP.md` for more detailed documentation.

