from gensim.models import Word2Vec

def word_embedding(reviews, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences=reviews, vector_size=vector_size, window=window, min_count=min_count)

    vocab = model.wv.key_to_index
    embedding_vectors = model.wv.vectors
    review_vectors = []
    for review in reviews:
        review_vector_list = []
        for word in review:
            if word in vocab:
                review_vector = embedding_vectors[vocab[word]]
                review_vector = review_vector.tolist()
                review_vector_list.append(review_vector)
        review_vectors.append(review_vector_list)
    return review_vectors
