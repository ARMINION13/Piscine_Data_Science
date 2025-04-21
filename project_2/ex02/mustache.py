import matplotlib.pyplot as plt
import psycopg2 as psy
import matplotlib.dates as mdates
import numpy as np

def make_box_plot1(data : list) :

    axisX : list = []

    for i in data:
        axisX.append(i[0])
    
    fig, box_plot = plt.subplots(figsize=(5, 5))
    box_plot.boxplot(axisX, vert=False, boxprops=dict(facecolor="lightgreen"), patch_artist=True)
    box_plot.grid(True, axis='x')

    plt.tight_layout()

def make_box_plot2(data : list) :

    axisX : list = []

    for i in data:
        axisX.append(i[1])
    
    fig, box_plot = plt.subplots(figsize=(5, 5))
    box_plot.boxplot(axisX, vert=False, boxprops=dict(facecolor="lightblue"), patch_artist=True)
    box_plot.grid(True, axis='x')

def get_box1_data() -> list :

    data : list = []

    cursor.execute(f"""
                    SELECT price
                    FROM customers
                    WHERE event_type = 'purchase'
                    """)
    data = cursor.fetchall()
    return data


def get_box2_data() -> list :

    data : list = []

    cursor.execute(f"""
                    SELECT user_id, SUM(price) AS avg_cart_price
                    FROM customers
                    WHERE event_type = 'purchase'
                    GROUP BY user_id, user_session
                    """)
    data = cursor.fetchall()
    return data

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

box_data = get_box1_data()
print_data = []

for i in box_data:
    print_data.append(i[0])

#Imprimir datos derivados del precio de los objetos
print(f"{'count':<8} {len(print_data):>15.6f}")
print(f"{'mean':<8} {np.mean(print_data):>15.6f}")
print(f"{'std':<8} {np.std(print_data):>15.6f}")
print(f"{'min':<8} {np.min(print_data):>15.6f}")
print(f"{'25%':<8} {np.percentile(print_data, 25):>15.6f}")
print(f"{'50%':<8} {np.percentile(print_data, 50):>15.6f}")
print(f"{'75%':<8} {np.percentile(print_data, 75):>15.6f}")
print(f"{'max':<8} {np.max(print_data):>15.6f}")

#Box plot del precio de los objetos comprados
make_box_plot1(box_data)

#Box plot cesta de la compra media por usuario
box_data = get_box2_data()
make_box_plot2(box_data)

plt.tight_layout()
plt.show()
cursor.close()
connect.close()