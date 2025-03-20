# Quick start BS4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#
# https://brightdata.com/faqs/beautifulsoup/extract-text-from-html
import requests
from bs4 import BeautifulSoup

def scrapeWithSoup(url):
    # Loads in the HTML
    response = requests.get(url)
    html_content = response.text

    # Creates soup obj
    soup = BeautifulSoup(html_content,  "html.parser")

    # Extracts all the text from the HTML
    text = soup.get_text()

    return text


    