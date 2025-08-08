import pandas as pd
import psycopg2
from glob import glob

# Read all CSVs
files = glob("pipeline/data/*.csv")
df = pd.concat([pd.read_csv(f) for f in files])

# Clean the data
df['item'] = df['item'].str.lower().str.strip()
df['item'] = df['item'].replace({'momu': 'momo', 'mo mo': 'momo'})
df['price'] = df['price'].fillna(40)
df['total'] = df['price'] * df['quantity']

# Connect to Postgres
conn = psycopg2.connect(
    host="localhost",
    database="momos",
    user="shreyarora",
    password=""  # leave blank if you're not using one
)

cur = conn.cursor()

# Upload rows one by one
for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO sales (branch, item, quantity, price, total)
        VALUES (%s, %s, %s, %s, %s)
    """, (row['branch'], row['item'], row['quantity'], row['price'], row['total']))

conn.commit()
cur.close()
conn.close()

print("✅ Data uploaded successfully.")
