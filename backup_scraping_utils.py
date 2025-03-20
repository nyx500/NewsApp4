# Quick start BS4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#
# Reference: https://brightdata.com/faqs/beautifulsoup/extract-text-from-html
import requests
from bs4 import BeautifulSoup

def scrapeWithSoup(url):
    """"
    Creates a BeautifulSoup4 URL scraping back-up option if newspaper3k does not work.

            Input Parameters:
                url (str): the URL to scrape news content from

            Output:
                text (str): the news text to get out of the web page
    """


    # Creates scraping browser headers for requests library
    # Reference for creating request headers: https://stackoverflow.com/questions/72484699/scrap-image-with-request-header-on-beautifulsoup
    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0", "cookie": "CONSENT=YES+cb.20230531-04-p0.en+FX+908"}

    # Loads in the HTML using requests library
    response = requests.get(url, headers=headers)

    # Extracts the HTML from the URL response
    html_content = response.text

    # Creates the Beautiful Soup object
    soup = BeautifulSoup(html_content,  "html.parser")

    # Extracts all the text from the HTML, this post was very helpful for resolving the scraping problem from many websites!
    # Reference: https://stackoverflow.com/questions/64691432/scrape-news-article-from-scraped-link-from-news-website-beautifulsoup-python
    text = soup.find_all("div", {"class", "most__list clearfix"})

    return text


    