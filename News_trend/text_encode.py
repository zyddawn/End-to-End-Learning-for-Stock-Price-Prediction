import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

MAX_LEN = 100
data_path = sys.argv[1]

comb = pd.read_csv(os.path.join(data_path, "Combined_News_DJIA.csv"))
news_text = comb[['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7', 'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15', 'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23', 'Top24', 'Top25']].values

labels = np.expand_dims(np.array(comb["Label"].values, dtype=np.float32), axis=1)
np.save("label.npy", labels)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')
model = model.cuda()
model.eval()

input_ids = []
for day in tqdm(news_text):
    tokenized_day = []
    for t in day:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))
        pad_ids = ids + [0, ]*(MAX_LEN-len(ids))
        tokenized_day.append(np.array(pad_ids))
    input_ids.append(np.array(tokenized_day))
input_ids = np.array(input_ids)

sentence_embeddings = []
for day in tqdm(input_ids):
    tokens_tensor = torch.LongTensor(day).cuda()
    encoded_layers, _ = model(tokens_tensor)
    embed = torch.mean(encoded_layers[-2], axis=1).detach().numpy()
    sentence_embeddings.append(embed)

features = np.array(sentence_embeddings)

print(np.shape(features))
print(np.shape(labels))

np.save("news.npy", features)



















