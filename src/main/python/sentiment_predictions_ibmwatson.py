from ibm_watson import ToneAnalyzerV3, ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pandas as pd
import json
import csv

API_KEY = ''
VERSION = '2017-09-21'
SERVICE_URL = 'https://api.eu-de.tone-analyzer.watson.cloud.ibm.com/instances/cee6f56f-a93c-487a-b21d-0ee417bc6df6'


def save_results(results): 
    with open('results/ibm_watson_results.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        for result in results: 
            writer.writerow(result)

            
def analyze(tone_analyzer, song_lyric): 
    try: 
        tone_analysis = tone_analyzer.tone(
            {'text': song_lyric},
            content_type='application/json', 
            sentences=False, 
        ).get_result()
        return tone_analysis
    except ApiException as ex:
        print("Failed to analyze song " + str(ex.code) + ": " + ex.message)

        
def analyze_data(tone_analyzer, data): 
    results = []
    for index, song in data.iterrows(): 
        
        analyzation = analyze(tone_analyzer, song.song_lyrics)

        for tone in analyzation['document_tone']['tones']: 
            score = tone['score']
            tone = tone['tone_id']
            results.append([song.album_name, song.song_name, tone, score])

    return results


def main():
    authenticator = IAMAuthenticator(API_KEY)
    tone_analyzer = ToneAnalyzerV3(
        version=VERSION,
        authenticator=authenticator
    )

    tone_analyzer.set_service_url(SERVICE_URL)

    data = pd.read_csv('data/rolling_stones_lyrics_archive.csv')
    data = data[data.album_type == 'album']

    results = analyze_data(tone_analyzer, data)

    save_results(results)

if __name__ == "__main__":
    main()
