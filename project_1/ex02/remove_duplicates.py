import psycopg2
from sqlalchemy import create_engine, types
import os


connect = psycopg2.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()


cursor.execute("""
                DELETE FROM customers a
                USING customers b
                WHERE a.ctid < b.ctid
                AND a.event_type = b.event_type
                AND a.product_id = b.product_id
                AND a.price = b.price
                AND a.user_id = b.user_id
                AND a.user_session = b.user_session
                ;
                """)

connect.commit()
cursor.close()
connect.close()
