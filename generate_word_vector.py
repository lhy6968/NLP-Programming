from sklearn.preprocessing import OneHotEncoder


def one_hot_encoding(reviews):
    encoder = OneHotEncoder()
    vocab = []
    for review in reviews:
        for word in review:
            vocab.append(word)

    vocab = set(vocab)

    int_vocab = {}
    for integer,word in enumerate(vocab):
        int_vocab[word] = integer

    review_vectors_list = []
    for review in reviews:
        review_indices = []
        for word in review:
            review_indices.append(int_vocab[word])
        #the one shot encoder code (23 line) is got from chatgpt
        review_onehot = encoder.fit_transform([[i] for i in review_indices]).toarray()
        review_onehot = review_onehot.tolist()
        review_vectors_list.append(review_onehot)
    return review_vectors_list


