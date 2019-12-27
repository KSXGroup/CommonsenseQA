import argparse
import json
import os
import pickle
import nltk
import numpy as np
from numba import jit

word_set = set()

class Example:
    def __init__(self, tokens, text_length, start, end, uid):
        self.tokens = []
        self.question_unique_id = uid
        self.text_tokens = tokens
        self.text_length = text_length
        self.start_pos = start
        self.end_pos = end


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def get_word2vec():
    vocab_size = len(word_set)
    print("vocab size: %d" % vocab_size)
    word_to_index = dict()
    for i, word in  enumerate(word_set):
        word_to_index[word] = i + 1
    glove_path = "../data/dataset/GloVe/glove.840B.300d.txt"
    embedding_matrix = np.zeros((vocab_size + 1, 300), dtype=np.float32)
    valid_cnt = 0
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            print('check vocab %d' % i, end='\r')
            vector = line.strip('\n').split(' ')
            word = vector[0]
            if word in word_set:
                vector = [float(x) for x in vector[1:]]
                embedding_matrix[word_to_index[word]] = np.array(vector)
                valid_cnt += 1
            elif word.capitalize() != word and word.capitalize() in word_set:
                vector = [float(x) for x in vector[1:]]
                embedding_matrix[word_to_index[word.capitalize()]] = np.array(vector)
                valid_cnt += 1
            elif word.lower() != word and word.lower() in word_set:
                vector = [float(x) for x in vector[1:]]
                valid_cnt += 1
                embedding_matrix[word_to_index[word.lower()]] = np.array(vector)
    print()
    print("valid vocab %d" % valid_cnt)
    return word_to_index, embedding_matrix


@jit
def process_data(dataset):
    examples = []
    data = dataset
    for i, example in enumerate(data):
        print("process data %d" % i ,end='\r')
        print(i, end='\r')
        text = example["passage"]["text"]
        charpos_to_tokenpos = np.zeros(len(text), dtype=int)
        for question in example["qas"]:
            len_text = len(word_tokenize(text))
            tokens = word_tokenize(text + question["query"])
            startpos = question["answers"][0]["start"]
            endpos = question["answers"][0]["end"]
            last_pos = 0
            for i, token in enumerate(tokens):
                word_set.add(token)
                word_set.add(token.lower())
                word_set.add(token.capitalize())
                pos = text[last_pos:].find(token)
                charpos_to_tokenpos[last_pos:last_pos + pos + len(token)] = i
                last_pos += pos + len(token)
            token_start = charpos_to_tokenpos[startpos]
            token_end = charpos_to_tokenpos[endpos]
            examples.append(Example(tokens, len_text, token_start, token_end, question["id"]))
    return examples

def load_data():
    with open("../data/dataset/train_example.pkl", "rb") as f:
        train_examples = pickle.load(f)
    with open("../data/dataset/dev_example.pkl", "rb") as f:
        dev_examples = pickle.load(f)
    with open("../data/dataset/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)
    with open("../data/dataset/embedding_matrix", "rb") as f:
        embedding_matrix = pickle.load(f)
    return train_examples, dev_examples, word_to_index, embedding_matrix

def text_to_index(ptrain, word_to_index, embedding_matrix):
    for i in range(len(ptrain)):
        for token in ptrain[i].text_tokens:
            idx = (word_to_index[token])
            if np.sum(np.abs(embedding_matrix[idx])) == 0:
                idx = word_to_index[token.lower()]
                if np.sum(np.abs(embedding_matrix[idx])) == 0:
                    idx = word_to_index[token.capitalize()]
            ptrain[i].tokens.append(idx)
    return ptrain


def main():
    with open("../data/dataset/record/train.json", "r") as f:
        train = json.load(f)["data"]
        print()
    with open("../data/dataset/record/dev.json", "r") as f:
        dev = json.load(f)["data"]
        print()
    ptrain = process_data(train)
    print("train data finished.")
    pdev = process_data(dev)
    print("dev data finished.")

    word_to_index, embedding_matrix = get_word2vec()

    ptrain = text_to_index(ptrain, word_to_index, embedding_matrix)
    pdev = text_to_index(pdev, word_to_index, embedding_matrix)

    print("saving files")
    with open("../data/dataset/train_example.pkl", "wb") as f:
        pickle.dump(ptrain, f)
    with open("../data/dataset/dev_example.pkl", "wb") as f:
        pickle.dump(pdev, f)
    with open("../data/dataset/word_to_index.pkl", "wb") as f:
        pickle.dump(word_to_index, f)
    with open("../data/dataset/embedding_matrix", "wb") as f:
        pickle.dump(embedding_matrix, f)


if __name__ == "__main__":
    main()