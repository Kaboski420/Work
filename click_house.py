import clickhouse_connect

# Connect to ClickHouse crawler database
# Password is SHA256 hash of the original password
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
print(f"ClickHouse version: {client.query('SELECT version()').result_rows[0][0]}")