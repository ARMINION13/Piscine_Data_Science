import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import psycopg2 as psy
import numpy as np


def make_bar_chart2(data : list) :

    axisX : list = []
    axisY : list = []


    for i in data:
        axisX.append(i[0])
        axisY.append(i[1])
    
    fig, bar_chart = plt.subplots()
    bar_chart.bar(axisX, axisY, width=44.5)
    bar_chart.xaxis.set_major_locator(MultipleLocator(50))
    bar_chart.yaxis.set_major_locator(MultipleLocator(5000))
    bar_chart.set_xlim(-25, 300)
    bar_chart.set_ylim(0, 50000)
    bar_chart.margins(x=15)
    bar_chart.set_axisbelow(True)
    bar_chart.grid(True)
    bar_chart.set_ylabel('customers')
    bar_chart.set_xlabel('monetary value in ₳')

    plt.tight_layout()

def make_bar_chart1(data : list) :

    axisX : list = []
    axisY : list = []


    for i in data:
        axisX.append(i[0])
        axisY.append(i[1])
    
    fig, bar_chart = plt.subplots()
    bar_chart.bar(axisX, axisY, align='edge', width=9)
    bar_chart.xaxis.set_major_locator(MultipleLocator(10))
    bar_chart.set_xlim(-2, 50)
    bar_chart.margins(x=5)
    bar_chart.set_axisbelow(True)
    bar_chart.grid(True)
    bar_chart.set_ylabel('customers')
    bar_chart.set_xlabel('frequency')

    plt.tight_layout()

def get_chart2_data() -> list :

    data : list = []

    cursor.execute(f"""
                    WITH temp AS(
                        SELECT 
                            user_id,
                            CASE
                                WHEN SUM(price)::float < 25 THEN 0::numeric
                                WHEN SUM(price)::float < 75 THEN 50::numeric
                                WHEN SUM(price)::float < 125 THEN 100::numeric
                                WHEN SUM(price)::float < 175 THEN 150::numeric
                                WHEN SUM(price)::float < 225 THEN 200::numeric
                            END AS gastos
                        FROM customers
                        WHERE event_type='purchase'
                        GROUP BY user_id
                        ORDER BY gastos
                    )
                    SELECT
                        gastos,
                        COUNT(*)
                    FROM temp
                    GROUP BY gastos
                    ORDER BY gastos
                    """)
    data = cursor.fetchall()
    return data

def get_chart1_data() -> list :

    data : list = []

    cursor.execute(f"""
                    WITH temp AS(
                        SELECT 
                            user_id,
                            CEIL(COUNT(*) / 10) * 10 AS compras
                        FROM customers
                        WHERE event_type='purchase'
                        GROUP BY user_id
                        ORDER BY compras
                    )
                    SELECT
                        compras,
                        COUNT(*)
                    FROM temp
                    GROUP BY compras
                    ORDER BY compras
                    """)
    data = cursor.fetchall()
    return data

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

#Grafica de numero de pedidos de acuerdo a la frecuencia
chart_data = get_chart1_data()
make_bar_chart1(chart_data)

#Grafica de altarians gastados por cliente
chart_data = get_chart2_data()
make_bar_chart2(chart_data)

plt.show()
cursor.close()
connect.close()