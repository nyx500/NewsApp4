# Quick start BS4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#
# https://brightdata.com/faqs/beautifulsoup/extract-text-from-html
import requests
from bs4 import BeautifulSoup

def scrapeWithSoup(url):
    # Create scraping headers
    # https://stackoverflow.com/questions/72484699/scrap-image-with-request-header-on-beautifulsoup
    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0", "cookie": "CONSENT=YES+cb.20230531-04-p0.en+FX+908"}

    # Loads in the HTML
    response = requests.get(url, headers=headers)
    html_content = response.text

    # Creates the beautiful soup object
    soup = BeautifulSoup(html_content,  "html.parser")

    # Extracts all the text from the HTML
    text = soup.get_text()

    return text


    