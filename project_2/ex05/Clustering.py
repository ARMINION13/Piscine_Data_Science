import matplotlib.pyplot as plt
import psycopg2 as psy
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans

def get_chart_data():

    data : list = []

    cursor.execute(f"""
                    SELECT 
                        user_id,
                        MAX(event_time) AS recency,
                        SUM(price) AS frequency
                    FROM customers
                    WHERE event_type='purchase'
                    GROUP BY user_id
                    """)
    
    data = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(data, columns=colnames)
    df['recency'] = (pd.to_datetime('2023-03-01') - df['recency']).dt.days
    df.drop(columns=['user_id'], inplace=True)

    return df

def apply_elbow_method( data ):

    kmean_model = KMeans(n_clusters=4, init='k-means++', random_state=42)
    data['cluster'] = kmean_model.fit_predict(data[['recency', 'frequency']])

    return data, kmean_model

def make_scatter_chart( data, model):

    color_map = ListedColormap(['lightblue', 'moccasin', 'lightgreen', 'black'])
    fig, scatter_chart = plt.subplots()
    scatter_chart.scatter(data['recency'], data['frequency'], c=data['cluster'],
    cmap=color_map, edgecolor='none', s=50)
    scatter_chart.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
    s=300, c='red', cmap='viridis', label='Centroids')
    scatter_chart.set_xlabel('Recency')
    scatter_chart.set_ylabel('Frequency')
    scatter_chart.set_title('Clusters of customers')
    scatter_chart.legend()

def make_bar_chart( data, model):

    unique, counts = np.unique(model.labels_, return_counts=True)
    fig, bar_chart = plt.subplots(figsize=(10, 5))
    bar_chart.barh(['inactives', 'usual customers', 'legacy customers', 'new customers'], counts, color=['lightblue', 'moccasin', 'lightgreen', 'black'])
    bar_chart.set_axisbelow(True)
    bar_chart.grid(True)
    bar_chart.set_xlabel('number of customers')

connect = psy.connect(host="localhost", dbname="piscineds", user="rgirondo", password="mysecretpassword")
cursor = connect.cursor()

chart_data = get_chart_data()
data, kmean_model = apply_elbow_method(chart_data)
make_scatter_chart(data, kmean_model)
make_bar_chart(data, kmean_model)

plt.show()
cursor.close()
connect.close()