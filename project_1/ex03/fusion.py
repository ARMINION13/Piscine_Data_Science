import psycopg2
from sqlalchemy import create_engine, types
import os


connect = psycopg2.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

cursor.execute("""
                ALTER TABLE customers
                ADD COLUMN IF NOT EXISTS category_id BIGINT,
                ADD COLUMN IF NOT EXISTS category_code VARCHAR(200),
                ADD COLUMN IF NOT EXISTS brand TEXT;
                """)

connect.commit()

cursor.execute("""
                UPDATE customers c
                SET
                    category_id = i.category_id,
                    category_code = i.category_code,
                    brand = i.brand
                FROM item i
                WHERE c.product_id = i.product_id
                    AND i.category_id IS NOT NULL
                    AND i.category_code IS NOT NULL
                    AND i.brand IS NOT NULL;
                """)

connect.commit()

cursor.close()
connect.close()