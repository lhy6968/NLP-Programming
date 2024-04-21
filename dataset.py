import gzip
import json
from torch.utils.data import Dataset
from text_clear import clear_text
from split_words import tokenize_text
from set_table import build_vocab
from generator_word_vector_2 import word_embedding
from generate_word_vector import one_hot_encoding


class ClothDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.renttherunway_data = []
        self.reviews = []
        self.ratings = []
        self.reviews_and_ratings = []
    def load_data(self):
        renttherunway_file = gzip.open(self.root_dir)
        for rtrw_item in renttherunway_file:
            self.renttherunway_data.append(json.loads(rtrw_item))
        for item in self.renttherunway_data:
            if item['rating'] is not None and item['review_summary'] is not None:
                self.reviews_and_ratings.append([item['review_summary'],int(item['rating'])])
    def preprocess_data(self):
        for i in range(len(self.reviews_and_ratings)):
            text = self.reviews_and_ratings[i][0]
            new_text = clear_text(text)
            new_text = tokenize_text(new_text)
            if new_text != []:
                self.reviews.append(new_text)
                self.ratings.append(self.reviews_and_ratings[i][1])
        self.reviews = word_embedding(self.reviews)
        max_len = 0
        for i in range(len(self.reviews)):
            if len(self.reviews[i]) > max_len:
                max_len = len(self.reviews[i])
            for j in range(len(self.reviews[i])):
                if len(self.reviews[i][j]) != 100:
                    print("error")
                    print(len(self.reviews[i][j]))
        print(max_len)

    def __getitem__(self, idx):
        data = self.reviews[idx]
        rating = self.ratings[idx]
        return data,rating
    def __len__(self):
        return len(self.reviews) 




#a = ClothDataset("renttherunway_final_data.json.gz")
#a.load_data()
#a.preprocess_data()
#print(a[1])
#print(a.ratings[0])
#print(type(a.ratings[0]))
