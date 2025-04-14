import psycopg2
import pandas as pd
from psycopg2 import sql

csv_file_dir = "/home/rgirondo/Piscine_Data_Science/project_0/subject/customer/data_2022_oct.csv"

csv_fd = pd.read_csv(csv_file_dir)

csv_line_1 = csv_fd.readline(2)

table_name = csv_file_dir.split("/")[-1][0:-4]

print(csv_line_1_2)

psycopg2 = psycopg2.connect(
    database = "piscineds",
    user = "rgirondo",
    host = "localhost",
    password = "mysecretpassword",
    port = "5432"
)

mycursor = psycopg2.cursor()
