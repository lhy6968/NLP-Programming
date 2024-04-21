import nltk

def tokenize_text(text):
    nltk.download('punkt')
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)

    return tokens
