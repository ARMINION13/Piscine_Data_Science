import psycopg2
from sqlalchemy import create_engine, types
import os


def create_combined_table(csv_dir : str, csv_name : list) :

    csv_fd = []
    connect = psycopg2.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
    cursor = connect.cursor()

    cursor.execute(f"""CREATE TABLE Customers (
                    event_time TIMESTAMP,
                    event_type VARCHAR(200),
                    product_id BIGINT,
                    price DOUBLE PRECISION,
                    user_id INT,
                    user_session TEXT
                    );""")

    for name in csv_name:
        with open(csv_dir + name, "r") as f:
            next(f)
            cursor.copy_from(f, "customers", sep=",", columns=("event_time", "event_type", "product_id", "price", "user_id", "user_session"))

    connect.commit()
    cursor.close()
    connect.close()

csv_dir = "/home/rgirondo/subject/customer/"

csv_files = [
    f for f in os.listdir(csv_dir)
    if os.path.isfile(os.path.join(csv_dir, f))
]

create_combined_table(csv_dir, csv_files)