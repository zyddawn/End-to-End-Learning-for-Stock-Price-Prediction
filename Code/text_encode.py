import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

MAX_LEN = 128
data_path = sys.argv[1]

comb = pd.read_csv(os.path.join(data_path, "Combined_News_DJIA.csv"))
top1_news = comb[["Date", "Top1", "Label"]]
news_text = top1_news["Top1"].values


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


tokenized_news_text = []
for t in tqdm(news_text):
    tokenized_news_text.append(tokenizer.tokenize(t))

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_news_text]
input_ids = [x+[0]*(MAX_LEN-len(x)) for x in input_ids]
# attention_masks = []
# for seq in input_ids:
#     seq_mask = [np.double(i>0) for i in seq]
#     attention_masks.append(seq_mask)
np_input_ids = np.array([np.array(x) for x in input_ids])


sentence_embeddings = []
for ids in tqdm(np_input_ids):
    tokens_tensor = torch.LongTensor([ids])
    encoded_layers, _ = model(tokens_tensor)
    embed = torch.mean(encoded_layers[-2].squeeze(0), axis=0)
    sentence_embeddings.append(embed)
features = torch.stack(sentence_embeddings).detach().numpy()
labels = np.expand_dims(top1_news["Label"].values, axis=1)
data_label = np.hstack((labels, features))

np.save("data_label.npy", data_label)



















