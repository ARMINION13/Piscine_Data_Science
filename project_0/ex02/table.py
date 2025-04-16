import psycopg2
from sqlalchemy import create_engine, types


def create_table_from_csv(csv_dir : str, csv_name : str) :

    csv_fd = []
    connect = psycopg2.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
    cursor = connect.cursor()

    cursor.execute(f"""CREATE TABLE {csv_name[0:-4]} (
                    event_time TIMESTAMP,
                    event_type VARCHAR(200),
                    product_id BIGINT,
                    price DOUBLE PRECISION,
                    user_id INT,
                    user_session TEXT
                    );""")

    with open(csv_dir + csv_name, "r") as f:
        next(f)
        cursor.copy_from(f, csv_name[0:-4], sep=",", columns=("event_time", "event_type", "product_id", "price", "user_id", "user_session"))

    connect.commit()
    cursor.close()
    connect.close()

create_table_from_csv("/home/rgirondo/subject/customer/", "data_2022_oct.csv")