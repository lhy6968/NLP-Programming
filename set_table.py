def build_vocab(texts):
    vocab = {}
    for text in texts:
        for word in text:
            if word not in vocab:
                vocab[word] = len(vocab) + 1

    return vocab
