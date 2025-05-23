import matplotlib.pyplot as plt
import psycopg2 as psy
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

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
        axisX.append(i[0])
        axisY.append(i[1])
    
    arrayX = np.array(axisX)
    arrayY = np.array(axisY)
    arrayXY = np.array(list(zip(arrayX, arrayY))).reshape(len(arrayX), 2)
    distortions = []
    inertias = []
    K = range(1, 11)

    for k in K:
        kmean_model = KMeans(n_clusters=k, random_state=42).fit(arrayXY)
        distortions.append(sum(np.min(cdist(arrayXY, kmean_model.cluster_centers_, 'euclidean'), axis=1)**2) / arrayXY.shape[0])
        inertias.append(kmean_model.inertia_)

    return K, distortions, inertias

def make_elbow_chart( K : list, data : list):

    fig, elbow_chart = plt.subplots()
    elbow_chart.plot(K, data)
    elbow_chart.xaxis.set_major_locator(MultipleLocator(2))
    elbow_chart.set_xlabel('Number of clusters')
    elbow_chart.set_title('The Elbow Method')
    elbow_chart.grid(True)

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

chart_data = get_chart_data()
K, distortions, inertias = apply_elbow_method(chart_data)
make_elbow_chart(K, distortions)

plt.show()
cursor.close()
connect.close()