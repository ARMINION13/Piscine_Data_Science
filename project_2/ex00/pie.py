import matplotlib.pyplot as plt
import psycopg2 as psy
import numpy as np

def make_pie_chart(data : dict) :  

    percentages : list = []
    fields : list = []

    for i in data:
        fields.append(i)
        percentages.append(data[i])

    plt.pie(np.array(percentages), labels = fields, autopct="%1.1f%%")
    plt.show()

def get_ex00_data() -> dict :

    rows : int = 0
    types : dict = {}

    cursor.execute("""
                    SELECT COUNT(*) FROM customers
                    """)

    rows = cursor.fetchall()[0][0]

    cursor.execute("""
                    SELECT 
                        DISTINCT event_type
                    FROM customers
                    """)

    fetch = cursor.fetchall()
    for it in fetch:
        for i in it:
            types[i] = 0

    for i in types:
        cursor.execute(f"SELECT COUNT(*) FROM customers WHERE event_type = '{i}'")
        types.update({i : (cursor.fetchall()[0][0] / rows * 100)})

    return types

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

pie_data = get_ex00_data()

make_pie_chart(pie_data)

cursor.close()
connect.close()