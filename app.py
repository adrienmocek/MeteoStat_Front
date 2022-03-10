
from asyncio import open_unix_connection
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
from io import BytesIO
from datetime import date, timedelta, datetime
from PIL import Image, UnidentifiedImageError
import requests
import pandas as pd
import matplotlib.image as mpimg

st.title('MeteoStat')

def iteration_15min(start, finish):
    ## Generateur de (an, mois, jour, heure, minute)
     while finish > start:
        start = start + timedelta(minutes=15)
        yield (start.strftime("%Y"),
               start.strftime("%m"),
               start.strftime("%d"),
               start.strftime("%H"),
               start.strftime("%M")
               )

def open_save_data(url, date_save):
    ## Ouvre l'image pointee par url
    ## Enregistre l'image avec l'extention date_save

    print(url, date_save)

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    img.save( f"images/radar{date_save}.png")
    pass

def scrapping_images (start, finish) :
    """Scrape images radar en ligne toutes les 15 min
    entre deux dates donnees sous forme de datetime.datetime
    Sauvegarde les dates pour lesquelles la page n'existe pas.  """
    missing_times = []
    saved_images = []
    for (an, mois, jour, heure, minute) in iteration_15min(start, finish):
        ## url scrapping :
        url = (f"https://static.infoclimat.net/cartes/compo/{an}/{mois}/{jour}"
            f"/color_{jour}{heure}{minute}.jpg")
        date_save = f'{an}_{mois}_{jour}_{heure}{minute}'

        try :
            open_save_data(url, date_save)
            saved_images.append(date_save)


        except UnidentifiedImageError :
            print (date_save, ' --> Missing data')
            missing_times.append(date_save)
    ## Save missing data list :
    missing_data_name = f'missing_datetimes_{start.strftime("%Y")}\
        {start.strftime("%m")}{start.strftime("%d")}_to_{finish.strftime("%Y")}\
            {finish.strftime("%m")}{finish.strftime("%d")}'
    pd.DataFrame(missing_times).to_pickle(missing_data_name)
    print(missing_times)
    return saved_images

def open_data(date_save):
    print('Open '+date_save)
    img = mpimg.imread(f"images/radar{date_save}.png")
    return img

if st.button('Scrapping'):
# print is visible in the server output, not in the page
    start = datetime(2017, 5, 1, 00)
    finish = datetime(2017, 5, 1, 23, 45)

    scrapping_images (start, finish)
    open_data()
