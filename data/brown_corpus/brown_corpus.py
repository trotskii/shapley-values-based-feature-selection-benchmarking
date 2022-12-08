from nltk.corpus import brown
import pandas as pd
import nltk 

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    nltk.download('brown')
    docs = []
    for f_id in brown.fileids():
        sentences = list(brown.sents(fileids=f_id))
        label = brown.categories(fileids=f_id)[0]
        new_docs = chunks(sentences, 10)
        for i in new_docs:
            flat_list = [item for sublist in i for item in sublist]
            docs.append({'Label': label, 'Text': ' '.join(flat_list)})
    df = pd.DataFrame(docs)
    df.to_csv('brown_corpus.csv', sep=';')
    


if __name__ == '__main__':
    main()