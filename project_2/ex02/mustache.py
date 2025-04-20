import matplotlib.pyplot as plt
import psycopg2 as psy
import matplotlib.dates as mdates
import numpy as np

def make_box_plot(data : list) :

    axisX : list = []

    for i in data:
        axisX.append(i[1])
    
    plt.figure(figsize=(8, 5))
    plt.boxplot(axisX, vert=False, boxprops=dict(facecolor="lightblue"), patch_artist=True, whis=0.20)

    plt.xticks(np.arange(int(min(axisX)), int(max(axisX)) + 1, step=2)) 
    plt.grid(True, axis='x')
    plt.xlim(min(axisX) - 1, max(axisX) + 1)
    plt.tight_layout()
    plt.yticks([])
    plt.show()

def get_box2_data() -> list :

    data : list = []

    cursor.execute(f"""
                    SELECT user_id, AVG(price) AS avg_cart_price
                    FROM customers
                    WHERE event_type = 'purchase'
                    GROUP BY user_id
                    HAVING AVG(price) BETWEEN 26 AND 43;
                    """)
    data = cursor.fetchall()
    return data

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

#Grafica de lineas Clientes / Dia
box_data = get_box2_data()
make_box_plot(box_data)

cursor.close()
connect.close()