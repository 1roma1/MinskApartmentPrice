import requests
import pandas as pd

from multiprocessing import Pool
from bs4 import BeautifulSoup


URL = 'https://realt.by/sale/flats/?search=eJwryS%2FPi89MUTV1SlU1dbE1NTQwAgBBKAWZ'
HEADERS = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36',
           'accept': '*/*'}


def request_html(url, headers, params=None):
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code==200:
        return resp
    else:
        raise Exception(f"Request error. Status code {resp.status_code}")


def get_pages_count(url, headers):
    try:
        resp = request_html(url, headers)
    except Exception as e:
        print(e)

    soup = BeautifulSoup(resp.text, 'html.parser')
    items = soup.find('div', class_='pagination')
    num_pages = int(items.find(
            'ul').find_all('li')[-1].get_text())
    return num_pages


def get_page_links(url, num_pages):
    page_links = []
    for i in range(0, num_pages):
        page_link = url if i == 0 else url+f'&page={i}'
        page_links.append(page_link)
    return page_links


def parse_links(url, headers=HEADERS):
    try:
        resp = request_html(url, headers)
    except Exception as e:
        print(e)
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    items = soup.find_all('div', class_='showcase-objects-item')
    links = []
    for item in items:
        links.append(item.find('a').get('href'))
    print(f"{url} is successfully parsed")
    return links


def parse_flat_info(url, headers=HEADERS):
    try:
        resp = request_html(url, headers)
    except Exception as e:
        print(e)
        return {}

    soup = BeautifulSoup(resp.text, 'html.parser')
    items = soup.find_all('div', class_='parameters')

    flat_info = {}
    price = soup.find('div', class_='nowrap').find_all('div')[1].get_text()
    flat_info['Цена USD'] = price

    for item in items:
        item = item.find('table', class_='parameters-table')
        params = item.find_all('tr')
        for param in params:
            feature_value = param.find_all('td')
            if len(feature_value) > 1:
                feature, value = feature_value
                flat_info[feature.get_text()] = value.get_text()
    print(f"{url} is successfully parsed")
    return flat_info


def dict_to_df(flats_dict):
    features = set()

    for flat in flats_dict:
        for key, value in flat.items():
            features.add(key)
    df = pd.DataFrame(flats_dict, columns=features)
    return df


if __name__ == "__main__":
    num_pages = get_pages_count(URL, HEADERS)
    page_links = get_page_links(URL, num_pages)
    links = []
    
    with Pool(10) as p:
        link_lists = p.map(parse_links, page_links)

    for link_list in link_lists:
        for link in link_list:
            links.append(link)
    
    with Pool(10) as p:
        flats_info = p.map(parse_flat_info, links)

    df = dict_to_df(flats_info)
    df.to_csv("data/parsed_data.csv", index=False)
