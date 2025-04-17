import psycopg2
from sqlalchemy import create_engine, types
import os


connect = psycopg2.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()


cursor.execute("""
                DROP TABLE IF EXISTS Customers_Temp;
                CREATE TABLE Customers_Temp AS
                WITH Temp AS (
                    SELECT *,
                        LAG(event_time) OVER (
                            PARTITION BY event_type, product_id, price, user_id, user_session
                            ORDER BY event_time
                        ) AS previous_time
                    FROM customers
                ),
                No_Lag AS (
                    SELECT *
                    FROM Temp
                    WHERE previous_time IS NULL
                    OR event_time - previous_time > interval '1 seconds'
                    ORDER BY event_time
                )
                SELECT * FROM No_Lag;
                """)

connect.commit()

cursor.execute("""
                DROP TABLE customers;
                ALTER TABLE Customers_Temp RENAME TO Customers;
                """)

connect.commit()

cursor.execute("""
                ALTER TABLE Customers
                DROP COLUMN previous_time
                """)

connect.commit()

cursor.close()
connect.close()

#For Evaluation
#SELECT * FROM public.customers WHERE product_id = 5809103 AND event_type = 'remove_from_cart' AND event_time = '2022-10-01 00:00:30'

#Rows 19175899