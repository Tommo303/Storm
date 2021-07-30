import requests
from bs4 import BeautifulSoup
import pandas as pd


# Set the url of the webpage to be scraped
cyclone_id = input('Enter Storm ID: ')
url = 'http://ibtracs.unca.edu/index.php?name=v04r00-' + cyclone_id

# Request html for webpage
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

# Find the data table
table = soup.find_all('h1')
table = table[-1]


for heading in table:
    print(heading)