import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import urllib

quote_page = "https://int.piaget.com/watches"
page = urllib.request.urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')

name_box = soup.find('div', attrs={'class': 'h5 product__title heading-alt'})
name = name_box.text.strip()
print(name)

# text_file = open(r"results.txt", "w")
# text_file.write(name)
# text_file.close()
