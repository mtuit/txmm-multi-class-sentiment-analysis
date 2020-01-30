import pandas as pd
from transformers import AlbertForSequenceClassification, AlbertModel, AlbertTokenizer, AlbertConfig, PretrainedConfig
import torch
import csv

LABELS = {0: 'anger', 1: 'anticipation', 2: 'disgust', 3: 'fear', 4: 'joy', 
            5: 'negative', 6: 'positive', 7: 'sadness', 8: 'surprise', 9: 'trust'}

LABELS = {0: 'anger', 1: 'anticipation', 2: 'disgust', 3: 'fear', 4: 'joy', 
            5: 'sadness', 6: 'surprise', 7: 'trust'}

# def save_results(results): 
#     with open('results/albert_nrc_emotion_results_v3.csv', 'a', newline='') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

#         for result in results: 
#             writer.writerow(result)

def predict(model, tokenizer, song_lyric, max_len): 
    model.eval()

    with torch.no_grad(): 
        
        input_ids = torch.tensor(tokenizer.encode(song_lyric, add_special_tokens=True, max_length=max_len)).unsqueeze(0) 

        preds = model(input_ids)

        max_score, max_score_class = preds[0].max(dim=1)
        
    return max_score_class.item(), max_score.item()

def main():
    data = pd.read_csv('data/rolling_stones_lyrics_archive.csv')
    data = data[data.album_type == 'album']

    # albert_config = AlbertConfig.from_json_file('models/config.json')
    # state_dict = torch.load('models/pytorch_model.bin')

    # albert_config.max_position_embeddings = 512 # Change the embeddings to 512, otherwise Transformers has problems with loading weights for some reason
    # model = AlbertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=None, config=albert_config, state_dict=state_dict)
    model = AlbertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='models/v3/')
    tokenizer = AlbertTokenizer.from_pretrained('tokenizers/')

    model.resize_token_embeddings(len(tokenizer))

    result = []
    with open('results/albert_nrc_emotion_results_v3.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        for index, song in data.iterrows(): 
            score_class, score = predict(model, tokenizer, song.song_lyrics, 512)
            print(result)
            result = [song.album_name, song.song_name, LABELS[score_class], score]
            writer.writerow(result)

if __name__ == "__main__":
    main()