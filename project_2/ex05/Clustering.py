import matplotlib.pyplot as plt
import psycopg2 as psy
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans

def get_chart_data() -> list :

    data : list = []

    cursor.execute(f"""
                    WITH temp AS(
                        SELECT 
                            user_id,
                            CAST(COUNT(*) AS FLOAT) / 5 AS frequency,
                            (EXTRACT(EPOCH FROM (DATE '2023-03-01' 
                            - MAX(event_time))) / (60 * 60 * 24)) / 30.44 AS recency
                        FROM customers
                        WHERE event_type='purchase'
                        GROUP BY user_id
                    )
                    SELECT
                        frequency,
                        recency::float
                    FROM temp
                    """)
    data = cursor.fetchall()
    return data

def apply_elbow_method( data : list ) -> list :

    axisX = []
    axisY = []

    for i in data:
        axisX.append(i[1])
        axisY.append(i[0])
    
    arrayX = np.array(axisX)
    arrayY = np.array(axisY)
    arrayXY = np.array(list(zip(arrayX, arrayY))).reshape(len(arrayX), 2)

    kmean_model = KMeans(n_clusters=3, init='k-means++', random_state=9)
    label = kmean_model.fit_predict(arrayXY)

    return arrayXY, kmean_model, label

def make_scatter_chart( data, model, label):

    fig, scatter_chart = plt.subplots()
    scatter_chart.scatter(data[:, 0], data[:, 1], c=label,
    cmap='viridis', edgecolor='none', s=30)
    scatter_chart.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
    s=300, c='red', cmap='viridis', label='Centroids', edgecolor='none')
    scatter_chart.set_xlabel('Recency')
    scatter_chart.set_ylabel('Frequency')
    scatter_chart.set_title('Clusters of customers')

def make_bar_chart( data, model):

    unique, counts = np.unique(model.labels_, return_counts=True)
    fig, bar_chart = plt.subplots(figsize=(10, 5))
    bar_chart.barh(['new customers', 'loyal customers', 'inactive'], counts, color=['lightblue', 'moccasin', 'lightgreen'])
    bar_chart.set_axisbelow(True)
    bar_chart.grid(True)
    bar_chart.set_xlabel('number of customers')

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

chart_data = get_chart_data()
data, kmean_model, label = apply_elbow_method(chart_data)
make_scatter_chart(data, kmean_model, label)
make_bar_chart(data, kmean_model)

plt.show()
cursor.close()
connect.close()