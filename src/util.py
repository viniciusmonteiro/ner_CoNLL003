import pandas as pd

def load_sentences(filepath):

    final = []
    sentences = []

    with open(filepath, 'r') as f:
        
        for line in f.readlines():
            
            if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                if len(sentences) > 0:
                    final.append(sentences)
                    sentences = []
            else:
                l = line.split(' ')
                sentences.append((l[0], l[3].strip('\n')))
    
    return final


def preprocess_sentences(lst_sentences):
    sentences = []
    num_sentence = 1
    for sentence in lst_sentences:
        
        for word_tag in sentence:
            sentences.append([num_sentence, word_tag[0], word_tag[1]])
        num_sentence += 1
    
    return pd.DataFrame(sentences, columns=["sentence_id", "words", "labels"])
    