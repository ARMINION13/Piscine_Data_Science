import psycopg2
from sqlalchemy import create_engine, types
import os


def create_table_from_csv(csv_dir : str, csv_name : str) :

    csv_fd = []
    connect = psycopg2.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
    cursor = connect.cursor()

    cursor.execute(f"""CREATE TABLE {csv_name[0:-4]} (
                    product_id INTEGER,
                    category_id BIGINT,
                    category_code VARCHAR(200),
                    brand TEXT
                    );""")

    with open(csv_dir + csv_name, "r") as f:
        next(f)
        cursor.copy_from(f, csv_name[0:-4], sep=",", null="")

    connect.commit()
    cursor.close()
    connect.close()

csv_dir = "/home/rgirondo/goinfre/subject/customer/"

csv_files = [
    f for f in os.listdir(csv_dir)
    if os.path.isfile(os.path.join(csv_dir, f))
]

for file in csv_files:
    create_table_from_csv(csv_dir, file)