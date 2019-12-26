import requests
import random 
import argparse
import json
import csv
import re
import time
from nltk import word_tokenize
from bs4 import BeautifulSoup


class LyricsScraper():

    #TODO Change to work with Archive -> https://web.archive.org/web/20160925055506/http://www.azlyrics.com/r/rollingstones.html
    def __init__(self, base_url, output):
        self.base_url = base_url
        self.base_artist_url = "https://www.azlyrics.com/lyrics/rollingstones/"
        self.output = output
        self.USER_AGENTS = ['Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
                            'Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4',
                            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
                            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393'
                            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0',
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36']
        self.lyrics = []
        self.content = self.get_request(self.base_url)
        self.scrape()

    def scrape(self): 
        print("Starting scraping {} for lyrics...".format(self.base_url))
        print("Searching for albums...")
        album_list = self.get_album_list(self.base_url)
        print("Found {} albums!".format(len(album_list)))
        print("Getting lyrics...")
        for album, lyrics in self.get_songs_and_lyrics(album_list):
            print("Saving lyrics...")
            self.save_lyrics(album, lyrics)
            print("Save succesful!")

    def get_album_list(self, base_url):
        album_list = []
        soup = BeautifulSoup(self.content, 'lxml')

        for album in soup.find_all(class_='album'):
            album_list.append(album.text)

        return album_list

    def get_songs_and_lyrics(self, album_list):
        soup = BeautifulSoup(self.content, 'lxml')
        current_album = album_list[0]
        albums = soup.find_all("div", {"id": "listAlbum"})
        
        children = albums[0].find_all(['a','div'], recursive=False)
        for child in children: 
            if child['class'][0] == 'album' and child.text != current_album:
                current_album = album_list[album_list.index(current_album) + 1]
            elif child['class'][0] == 'listalbum-item': 
                song_name = child.text
                lyrics = self.get_song_lyrics(song_name)
                lyrics = self.clean_lyrics(lyrics)
                lyrics_tuple = (song_name, lyrics)
                yield (current_album, lyrics_tuple)

    def clean_lyrics(self, lyrics):
        lyrics = lyrics.strip('\r\n')
        lyrics = [sent for sent in lyrics.split('\n') if sent.strip() != '']
        lyrics = ". ".join(lyrics)
        return lyrics

    def get_song_lyrics(self, song_name):
        stripped_song_name = song_name.lower().replace(" ", "").rstrip()
        print("Getting lyrics for {}...".format(song_name))
        content = self.get_request(self.base_artist_url + stripped_song_name + ".html")
        soup = BeautifulSoup(content, 'lxml')
        lyrics = soup.find_all("div", limit=20)[-1].text
        return lyrics

        
    def get_request(self, url): 
        content = requests.get(url, headers={'User-Agent': random.choice(self.USER_AGENTS)}).content
        time.sleep(1)
        return content

    def save_lyrics(self, album, lyrics):
        with open(self.output, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            # writer.writerow(['album_type', 'album_name', 'album_year', 'song_name', 'song_lyrics'])

            album_type, album_name, album_year = self.get_album_meta_data(album)
            song_name, song_lyrics = lyrics
            writer.writerow([album_type, album_name, album_year, song_name, song_lyrics])

    def get_album_meta_data(self, album):
        album_type = album.split(":")[0]
        regex_result = re.search(r'\"(\w|\s)*\"', album)
        album_name = regex_result.group(0).replace("\"", "") if regex_result != None else 'Unknown'
        regex_result = re.search(r'\(\d{4}\)', album)
        album_year = regex_result.group(0).replace("(", "").replace(")", "") if regex_result != None else 'Unknown'
        return album_type, album_name, album_year
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', required=True, help='Base url where scraper will start')
    parser.add_argument('-o', '--output', required=True, help='Output path where results will be placed')
    args = parser.parse_args()
    LyricsScraper(args.base_url, args.output)