import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification
from transformers import  AlbertTokenizer
import pandas as pd
import time
import numpy as np

NUM_LABELS = 8  # Amount of labels, [anger, anticipation, disgust, fear, joy, sadness, surprise, trust]
N_EPOCHS = 5

class NRCEmotionDataset(Dataset):

    def __init__(self, filename, max_len, tokenizer):
        self.df = pd.read_csv(filename, delimiter = '\t')
        self.tokenizer = tokenizer
        self.max_len = max_len

        new_tokens = self.df['word'].tolist()
        self.tokenizer.add_tokens(new_tokens)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        word = self.df.loc[index, 'word']
        label = self.df.loc[index, 'label_int']

        # Preprocessing the text to be suitable for Albert
        tokens = self.tokenizer.tokenize(word) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))] 
        else:
            tokens = tokens[:self.max_len-1] + ['[SEP]'] # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) # Obtaining the indices of the tokens in the Albert Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label

def get_accuracy_from_predictions(predictions, label):
    max_preds = predictions.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(label)
    return correct.sum() / torch.FloatTensor([label.shape[0]])

def get_pred_classes_from_predictions(predictions):
    _, max_scores = predictions.max(dim = 1)
    return max_scores

def evaluate(model, criterion, dataloader):
    model.eval()

    mean_accuracy, mean_loss = 0, 0
    count = 0

    with torch.no_grad():

        for seqs, attn_masks, labels in dataloader:

            loss, logits = model(input_ids=seqs, attention_mask=attn_masks, labels=labels)

            mean_loss += loss.item()

            mean_accuracy += get_accuracy_from_predictions(logits, labels).item()

    return mean_loss / len(dataloader), mean_accuracy / len(dataloader)

def train(model, criterion, optimizer, train_loader): 
    model.train()

    epoch_loss = 0
    epoch_accuracy = 0
    preds_classes_result = []
    preds_trues_result = []
    
    for it, (seqs, attn_masks, labels) in enumerate(train_loader):
        optimizer.zero_grad()  

        loss, logits = model(input_ids=seqs, attention_mask=attn_masks, labels=labels)
        accuracy = get_accuracy_from_predictions(predictions=logits, label=labels)

        preds_classes = get_pred_classes_from_predictions(predictions=logits)
        
        preds_classes_result.append(preds_classes.tolist())
        preds_trues_result.append(labels.tolist())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

        # print(preds_classes_result)
        # print(preds_trues_result)

        preds_classes_result_np = np.array(preds_classes_result).flatten()
        preds_trues_result_np = np.array(preds_trues_result).flatten()
        preds_with_true_preds = [preds_classes_result_np, preds_trues_result_np]    


        if it % 25 == 0:
            accuracy = get_accuracy_from_predictions(logits, labels)
            print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it, epoch, loss.item(), accuracy.item()))

    return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader), preds_with_true_preds

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    # print("Instantiating Albert Model...")
    # start = time.time()
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v1", num_labels=NUM_LABELS)
    tokenizer = AlbertTokenizer.from_pretrained('tokenizers/')
    # print("Done in {} seconds".format(time.time() - start))

    # print("Instantiating criterian and optimizer...")
    # start = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 2e-5)
    # print("Done in {} seconds".format(time.time() - start))

    # print("Instantiating train and val loaders...")
    # start = time.time()
    train_set = NRCEmotionDataset(filename='C:/Users/Mick/Documents/RU/Master Data Science/Text and Multimedia Mining/Project/data/nrc_emotion_train_v2.txt', max_len=10, tokenizer=tokenizer)
    val_set = NRCEmotionDataset(filename='C:/Users/Mick/Documents/RU/Master Data Science/Text and Multimedia Mining/Project/data/nrc_emotion_val_v2.txt', max_len=10, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=64, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=5)
    # print("Done in {} seconds".format(time.time() - start))

    model.resize_token_embeddings(len(tokenizer)) # Resize since we added tokens to vocab from our dataset

    print("Starting training... This may take a while")

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc, train_preds_with_true_preds = train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader)
        valid_loss, valid_acc = evaluate(model=model, criterion=criterion, dataloader=val_loader)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/v3/nrc_emotion_model_'.format(epoch))
            model.save_pretrained('models/v3/')


        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print('\t      Preds: {}'.format(train_preds_with_true_preds))

    tokenizer.save_pretrained('tokenizers/v3/')

    

