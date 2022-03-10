
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime as dt
from MeteoStat.data import open_data, scrapping_images
from MeteoStat.preprocessing import preproc_data
from MeteoStat.data import save_data
from MeteoStat.predict import predict
import numpy as np
from MeteoStat.predict import get_model
import matplotlib.image as mpimg
#rom visualization import superpose_image
from MeteoStat.visualization import make_gif
from fastapi.responses import  FileResponse


app = FastAPI()
carte_test = mpimg.imread("images/carte_test.png")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict", response_class=FileResponse)
def make_prediction(date):
    """
    input start date as a string
    returns 20 images (10 true, 10 pred)
    """
    #transform string into start date
    start = dt.datetime.strptime(date,'%Y-%m-%d_%H%M')
    finish = start + dt.timedelta(minutes=150)
    #save 10 images from start to finish as png in image_preproc folder
    saved_images = scrapping_images(start, finish)
    imgs = []
    preproc_images = []
    for date_save in saved_images:
        imgs.append(open_data(date_save))

    #transform them into preprocessed X images
    for img in imgs:
        preproc_images.append(preproc_data(img))

    X = np.array([x[0] for x in preproc_images])
    X = X[:, ::5, ::5]
    X = np.expand_dims(X, axis=3)
    X = np.expand_dims(X, axis=0)
    print(X.shape)
    y_pred = predict(X)

    X = np.squeeze(X, axis = 0)
    X = np.squeeze(X, axis = 3)

    y_pred = (255 * y_pred).astype(int)
    y_final = X + y_pred
    print(y_pred.shape)
    print(len(y_final))
    make_gif(y_final, "prediction.gif")

    return FileResponse("prediction.gif")
    #return y_pred

import os
from math import sqrt
from tkinter import Y
from webbrowser import get
from tensorflow.keras import models
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

PATH_TO_LOCAL_MODEL = 'model.joblib'



def get_model():

    """
    load weights from the trained model
    """
    return models.load_model("MeteoStat/data/AJ_my_model_mse")

def predict(X):

    """
    input X : ndarray of size (10,650,420) representing a set of 10 images

    returns y_pred as ndarray of size (10,130,84) representing a set of 10 images
    """

    model = get_model()

    y_pred = model.predict(X)

    # - 2 dimensions
    y_pred = np.squeeze(y_pred, axis = 0)
    y_pred = np.squeeze(y_pred, axis = 3)

    return y_pred











# from secrets import choice
# from webbrowser import get
# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import csv
"## Coucou"
# "# MeteoStat"


# @st.cache
# def get_map_data():

#     return pd.DataFrame(
#             np.random.randn(1000, 2) / [50, 50] + [48.856614, 2.3522219],
#             columns=['lat', 'lon']
#         )

# df = get_map_data()

# st.map(df)

# import matplotlib.pyplot as plt

# from scipy import misc

# fig, ax = plt.subplots()

# img = mpimg.imread('img.png')
# face = misc.face(gray=True)
# ax.imshow(img, cmap='gray')

# st.pyplot(fig)

# CSS = """
# h1 {
#     color: red;
# }
# .stApp {
#     background-image: url(https://contents.mediadecathlon.com/p1769786/k$1b57aa6cf0393de84238a7805f966f33/1180x0/2634pt1881/5269xcr2693/randonnee-sous-la-pluie.jpg?format=auto&quality=80);
#     background-size: cover;
# }
# """

# if st.checkbox('Inject CSS'):
#     st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# #BUTTON

# cities_df = pd.read_csv('data/lewagon_cities.csv')

# def get_coordinates(city_name):
#     try:
#         return cities_df.query(f'city == "{city_name}"').lat.values[0], cities_df.query(f'city == "{city_name}"').lon.values[0]
#     except:
#         return "Please choose a valid location"

# a = st.button('Paris')
# b = st.button('Fontainebleau')
# if a:
#     location = list(get_coordinates('Paris'))
# if b:
#     location = list(get_coordinates('Fontainebleau'))


#MAP



# from streamlit_folium import folium_static

# import folium

# import os

# import pandas as pd

# m = folium.Map(location, zoom_start=6)

# geojson_path = os.path.join("data", "departements.json")
# cities_path = os.path.join("data", "lewagon_cities.csv")

# for _, city in pd.read_csv(cities_path).iterrows():

#     folium.Marker(
#         location=[city.lat, city.lon],
#         popup=city.city,
#         icon=folium.Icon(color="red", icon="info-sign"),
#     ).add_to(m)

# def color_function(feat):
#     return "red" if int(feat["properties"]["code"][:1]) < 5 else "blue"

# folium.GeoJson(
#     geojson_path,
#     name="geojson",
#     style_function=lambda feat: {
#         "weight": 1,
#         "color": "black",
#         "opacity": 0.25,
#         "fillColor": color_function(feat),
#         "fillOpacity": 0.25,
#     },
#     highlight_function=lambda feat: {
#         "fillColor": color_function(feat),
#         "fillOpacity": .5,
#     },
#     tooltip=folium.GeoJsonTooltip(
#         fields=['code', 'nom'],
#         aliases=['Code', 'Name'],
#         localize=True
#     ),
# ).add_to(m)

# folium_static(m)

# st.write('# Full width map')

# not possible to demo this without setting the full site in wide mode




import folium
from folium import plugins
from matplotlib.image import imread
from folium.raster_layers import ImageOverlay
from streamlit_folium import folium_static

# boundary of the image on the map
min_lon = 47
max_lon = 49
min_lat = 2
max_lat = 3

# create the map
map_ = folium.Map(location=[48.8566, 2.3522],
                  tiles='Stamen Terrain', zoom_start = 8)

tooltip = "Fontainebleau"
folium.Marker([48.4193, 2.6330], popup="Forêt de Fontainebleau", tooltip=tooltip).add_to(map_)

# read in png file to numpy array
data = imread('data/lol.png')

# Overlay the image
map_.add_children(ImageOverlay(data, opacity=0.8, \
        bounds =[[min_lat, min_lon], [max_lat, max_lon]]))

folium_static(map_)


# import streamlit as st
# from streamlit_folium import folium_static
# import folium

# "# streamlit-folium"

# with st.echo():
#     import streamlit as st
#     from streamlit_folium import folium_static
#     import folium

#     min_lon = 2.5
#     max_lon = 2.7
#     min_lat = 48
#     max_lat = 48.7
#     # center on Liberty Bell
#     m = folium.Map(location=[48.8566, 2.3522], zoom_start=8)

#     # add marker for Liberty Bell
#     tooltip = "Liberty Bell"
#     folium.Marker(
#         [48.4193, 2.6330], popup="Forêt de Fontainebleau", tooltip=tooltip
#     ).add_to(m)

#     m.add_children(ImageOverlay(data, opacity=0.8, \
#         bounds =[[min_lat, min_lon], [max_lat, max_lon]]))

#     # call to render Folium map in Streamlit
#     folium_static(m)
