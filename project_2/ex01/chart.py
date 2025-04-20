import matplotlib.pyplot as plt
import psycopg2 as psy
import matplotlib.dates as mdates
import numpy as np

def make_line_chart(data : list) :

    axisX : list = []
    axisY : list = []


    for i in data:
        axisX.append(i[0])
        axisY.append(i[1])
    
    line_chart = plt.subplot() 
    line_chart.plot(axisX, axisY)
    line_chart.xaxis.set_major_locator(mdates.MonthLocator())
    line_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    line_chart.margins(x=0)
    line_chart.grid(True)

    plt.tight_layout()
    plt.ylabel('Number of customers')
    plt.show()

def make_bar_chart(data : list) :

    axisX : list = []
    axisY : list = []


    for i in data:
        axisX.append(i[0])
        axisY.append(i[1])
    
    bar_chart = plt.subplot() 
    bar_chart.bar(axisX, axisY, width=25)
    bar_chart.xaxis.set_major_locator(mdates.MonthLocator())
    bar_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    bar_chart.set_axisbelow(True)
    bar_chart.grid(True, axis='y')

    plt.tight_layout()
    plt.ylabel('total sales in million of ₳')
    plt.xlabel('month')
    plt.show()

def make_stack_chart(data : list) :

    axisX : list = []
    axisY : list = []


    for i in data:
        axisX.append(i[0])
        axisY.append(i[1])
    
    stack_chart = plt.subplot() 
    stack_chart.stackplot(axisX, axisY)
    stack_chart.xaxis.set_major_locator(mdates.MonthLocator())
    stack_chart.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    stack_chart.set_axisbelow(True)
    stack_chart.margins(x=0)
    stack_chart.grid(True)

    plt.tight_layout()
    plt.ylabel('average spend/customers in ₳')
    plt.show()

def get_chart1_data() -> list :

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
    return data

def get_chart2_data() -> list :

    data : list = []

    cursor.execute(f"""
                    SELECT DATE_TRUNC( 'month', event_time) AS month, 
                    SUM(price) AS revenue
                    FROM customers
                    WHERE event_type='purchase'
                    GROUP BY month
                    ORDER BY month
                    """)
    data = cursor.fetchall()
    return data

def get_chart3_data() -> list :

    data : list = []

    cursor.execute(f"""
                    SELECT DATE_TRUNC( 'day', event_time) AS days, 
                    SUM(price) / COUNT(DISTINCT user_id) AS user_spend
                    FROM customers
                    WHERE event_type='purchase'
                    GROUP BY days
                    ORDER BY days
                    """)
    data = cursor.fetchall()
    return data

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

#Grafica de lineas Clientes / Dia
chart_data = get_chart1_data()
make_line_chart(chart_data)

#Grafica de barras Beneficios en Altarians / Mes
chart_data = get_chart2_data()
make_bar_chart(chart_data)

#Grafica de area Media de gasto por usuario / Dia
chart_data = get_chart3_data()
make_stack_chart(chart_data)

cursor.close()
connect.close()