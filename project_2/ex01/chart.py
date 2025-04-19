import matplotlib.pyplot as plt
import psycopg2 as psy
import numpy as np

def make_line_chart(data : list) :

    dates : dict = {}

    for i in data:
        if (dates.get(i[0].strftime('%m/%d/%Y'))):
            dates[i[0].strftime('%m/%d/%Y')] += 1
        else:
            dates[i[0].strftime('%m/%d/%Y')] = 1


def get_ex01_data() -> list :

    data : list = []

    cursor.execute(f"""
                    SELECT DATE_TRUNC( 'day', event_time) AS days, 
                    COUNT(*) AS compras
                    FROM customers
                    WHERE event_type='purchase'
                    GROUP BY days
                    ORDER BY days
                    """)
    
    data = cursor.fetchall()
    print(data[0])
    print(data[1])

    return data

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

chart_data = get_ex01_data()
#make_line_chart(chart_data)

cursor.close()
connect.close()