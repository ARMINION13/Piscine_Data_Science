import pandas as pd
from sqlalchemy import create_engine, types
import os


def create_table_from_csv(csv_dir : str, csv_name : str) :

    csv_fd = pd.read_csv(csv_dir + csv_name, nrows=100)

    dtype_mapping = ({
        'category_code': types.VARCHAR(200),
        'product_id' : types.INTEGER()
    })

    engine = create_engine('postgresql://rgirondo:mysecretpassword@localhost:5432/piscineds')

    csv_fd.to_sql(csv_name[0:-4], engine, if_exists='replace', dtype = dtype_mapping)

csv_dir = "/home/rgirondo/subject/item/"

csv_files = [
    f for f in os.listdir(csv_dir)
    if os.path.isfile(os.path.join(csv_dir, f))
]

for file in csv_files:
    create_table_from_csv(csv_dir, file)