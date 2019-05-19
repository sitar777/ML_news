import string
import pickle
import re
import math


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


print('Training...')
categories = ['business', 'culture', 'science', 'economics', 'forces', 'life', 'media', 'sport', 'style', 'travel']
n = len(categories)
words_in_category_count = [0] * 10
documents_in_category_count = [0] * 10
words = {}
train = open('news_train.txt', 'r')

for line_in_file in train:
    table = str.maketrans({key: None for key in string.punctuation})
    line_in_file = re.sub('[-.»«0-9]', ' ', line_in_file)
    line_in_file = line_in_file.translate(table)
    line_in_file = re.sub(r'\b\w{1,2}\b', ' ', line_in_file)
    line_in_file = line_in_file.lower()
    line_split = [s.strip() for s in line_in_file.split("\t")]
    category = line_split[0].split()
    words_from_file = line_split[1].split() + line_split[2].split()
    category_number = categories.index(category[0])
    words_in_category_count[category_number] += 1
    documents_in_category_count[category_number] += 1
    for word in words_from_file:
        if words.__contains__(word):
            words[word][category_number] += 1
        else:
            words[word] = [1] * 10
            words[word][category_number] += 1

for i in range(n):
    words_in_category_count[i] += len(words)

train.close()
print('Text reading complete!')
print('Counting probabilities...')
word_probability = {}
for word in words:
    word_count = sum(words[word])
    word_probability[word] = [0]*10
    for i in range(n):
        word_probability[word][i] = words[word][i]/word_count

print('Probability counting complete!')
print('Normalization...')

documents_count = sum(documents_in_category_count)
category_probability_ln = [0]*10
for i in range(n):
    category_probability_ln[i] = math.log(documents_in_category_count[i]/documents_count)

word_probability_ln_normalized = {}
for word in words:
    word_probability_ln_normalized[word] = [0] * 10
    for i in range(n):
        denominator = 0
        numerator = word_probability[word][i] / category_probability_ln[i]
        for j in range(n):
            denominator += word_probability[word][j] / category_probability_ln[i]
        word_probability_ln_normalized[word][i] = math.log(numerator / denominator)

print('Normalization complete!')

print('Training complete!')
print('Saving variables...')
save_obj(word_probability_ln_normalized, 'word_probability_ln_normalized')
print('Saving complete!')

print('Recognition...')
unknown_word_probability = [0.1]*10
unknown_word_probability_ln_normalized = [0]*10
for i in range(n):
    denominator = 0
    numerator = unknown_word_probability[i] / words_in_category_count[i]
    for j in range(n):
        denominator += unknown_word_probability[j] / words_in_category_count[j]
    unknown_word_probability_ln_normalized[i] = math.log(numerator / denominator)

test = open('news_test.txt', 'r')
output = open('output.txt', 'w+')

for line_in_file in test:
    line_in_file = re.sub('[-.»«0-9]', ' ', line_in_file)
    line_in_file = line_in_file.translate(table)
    line_in_file = re.sub(r'\b\w{1,2}\b', ' ', line_in_file)
    line_in_file = line_in_file.lower()
    line_split = [s.strip() for s in line_in_file.split("\t")]
    words_from_file = line_split[0].split() + line_split[1].split()
    p = [0]*10
    for word in words_from_file:
        if word_probability_ln_normalized.__contains__(word):
            p = [a + b for a, b in zip(word_probability_ln_normalized[word], p)]
        else:
            p = (a + b for a, b in zip(unknown_word_probability_ln_normalized, p))
    p = [a + b for a, b in zip(category_probability_ln, p)]
    category_p = max(p)
    category = p.index(category_p)
    output.write("%s\n" % categories[category])
test.close()
output.close()
print('Recognition complete!')
