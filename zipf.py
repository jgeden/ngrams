import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

CUTOFF_PERCENT = 0.66

# Parse text file into an array of words
def parse_text(filename):
    text = []
    with open(filename) as f:
        for line in f.readlines():
            for char in ['\n', '.', ',', '"', "'", '-', '(', ')']:
                line = line.replace(char, ' ')
            words = line.split(' ')
            for word in words:
                text.append(word)

    text = list(filter(lambda word: word != '' and word != ' ', text))
    return text


def toki_pona_parse(n):
    freq_dict = {}
    with open(os.path.join('texts', 'toki_pona', f'{n}-gram.txt')) as f:
        for line in f.readlines()[1:]:
            count = line.strip().split(' ')[0]
            ngram = ' '.join(line.strip().split(' ')[1:])
            freq_dict[ngram] = int(count)
    return freq_dict


# Return a dict of n-gram/frequency pairs
def count_ngrams(text, n):
    freq_dict = {}
    for n in range(1, n+1):
        for i in range(len(text) - n):
            word = ' '.join(text[i:i+n])
            if word not in freq_dict:
                freq_dict[word] = 0

            freq_dict[word] += 1
    return freq_dict


# Plot the rank of a word against its frequency
def plot_rank_freq(freq_dict, n, filename, shortened=False):
    rank_list = [(word, freq_dict[word]) for word in freq_dict.keys()]
    rank_list = list(map(lambda term: (term[0], math.log(term[1])), rank_list))
    rank_list.sort(reverse=True, key=lambda term: term[1])

    ranks = [math.log(i+1) for i in range(len(rank_list))]
    print(min(ranks), max(ranks))

    if shortened:
        upper_bound = max(ranks)*0.75
        cutoff = 0
        for rank in ranks:
            if rank >= upper_bound:
                break
            cutoff += 1
    else:
        cutoff = len(ranks)

    ranks = ranks[:cutoff]
    print(min(ranks), max(ranks))
    rank_list = rank_list[:cutoff]

    plt.figure()
    plt.plot(ranks, [term[1] for term in rank_list])

    plt.title(f'Rank-Frequency n-Gram Distribution (n={n})')
    plt.xlabel('rank (log scale)')
    plt.ylabel('frequency (log scale)')
    plt.savefig(filename)

def run_linear_reg(freq_dict,n,filename, shortened=False):
    rank_list = [(word, freq_dict[word]) for word in freq_dict.keys()]
    rank_list = list(map(lambda term: (term[0], math.log(term[1])), rank_list))
    rank_list.sort(reverse=True, key=lambda term: term[1])

    ranks = [math.log(i+1) for i in range(len(rank_list))]

    df = pd.DataFrame()
    df['rank'] = ranks
    df['frequency'] = [term[1] for term in rank_list]

    X = df['rank'].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df['frequency'].values.reshape(-1, 1)

    if shortened:
        upper_bound = max(X)*0.75
        cutoff = 0
        for rank in X:
            if rank >= upper_bound:
                break
            cutoff += 1
    else:
        cutoff = len(X)

    X = X[:cutoff]
    print(X[-1])
    Y = Y[:cutoff]

    model = LinearRegression().fit(X, Y)

    r_sq = model.score(X, Y)
    print(f"coefficient of determination: {r_sq}")
    print(n)
    print(filename)
    print()


def main():
    texts = [
        'esperanto.txt',                          # Esperanto
        'InterlinguaSentences_shortened.txt'      # Interlingua
    ]

    for text in texts:
        words = parse_text(os.path.join('texts', text))

        
        for n in [1, 2, 3, 4]:
            freq_dict = count_ngrams(words, n)
            
            run_linear_reg(freq_dict,n,os.path.join('figures',
                                    f'{text.replace(".txt", "")}-n_{n}.png'))
            
            plot_rank_freq(freq_dict, n, 
                        os.path.join('figures',
                                     f'{text.replace(".txt", "")}-n_{n}.png'))

    # Toki Pona texts
    for n in [1, 2, 3, 4]:
        text = 'toki_pona'
        freq_dict = toki_pona_parse(n)
        run_linear_reg(freq_dict,n,os.path.join('figures',
                                    f'{text.replace(".toki_pona-n", "")}-n_{n}.png'))
        plot_rank_freq(freq_dict, n,
                       os.path.join('figures', f'toki_pona-n_{n}.png'))



if __name__ == '__main__':
    main()
