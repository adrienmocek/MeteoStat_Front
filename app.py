# from secrets import choice
# from webbrowser import get
# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import csv
# "## Coucou"
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

# boundary of the image on the map
min_lon = -123.5617
max_lon = -121.0617
min_lat = 37.382166
max_lat = 39.048834

# create the map
map_ = folium.Map(location=[48.8566, 2.3522],
                  tiles='Stamen Terrain', zoom_start = 8)

# read in png file to numpy array
data = imread('mockup-iphone-écran-et-arrière-plan-ont-png-isolé-en-ceci-pour-diverses-applications-158473491.jpeg')

# Overlay the image
map_.add_children(plugins.ImageOverlay(data, opacity=0.8, \
        bounds =[[min_lat, min_lon], [max_lat, max_lon]]))
map_
