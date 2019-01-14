import math
import os.path
import re
from string import ascii_lowercase
from string import ascii_letters


lst_unigram = []
lst_bigram = []

lst_unigram_ex = []
lst_bigram_ex = []

lst_unigram_prob_EN = []
lst_unigram_prob_FR = []
lst_unigram_prob_GR = []

lst_unigram_prob_EN_ex = []
lst_unigram_prob_FR_ex = []
lst_unigram_prob_GR_ex = []

lst_biagram_prob_EN = []
lst_biagram_prob_FR = []
lst_biagram_prob_GR = []

lst_biagram_prob_EN_ex = []
lst_biagram_prob_FR_ex = []
lst_biagram_prob_GR_ex = []

chars_ex = str(ascii_letters)+"'- "


# create unigram language model for basic setup
def mdl_unigram_bs(lst):
    m = {}
    for c in ascii_lowercase:
        m[c] = (lst.count(c) + 0.5) / (len(lst_unigram) + 26 * 0.5)
    return m


# create biogram language model for basic setup
def mdl_bigram_bs(lst):
    m = {}
    for c in ascii_lowercase:
        for h in ascii_lowercase:
            m[(c, h)] = (lst.count((c, h)) + 0.5) / (lst_unigram.count(c) + math.pow(26, 2) * 0.5)
    return m


# tokenizing for ungigram model basic setup
def tokenize_unigram_bs(input_file='', sentence=''):
    if input_file != '':
        lst_unigram.clear()
        with open(input_file, 'r', errors='ignore') as f:
            f = f.readlines()
            for line in f:
                line = line.replace('\n', '')
                line = line.lower()
                line = ''.join(re.sub("[^a-z]+", "", line))
                for letter in line:
                    lst_unigram.append(letter)
        return lst_unigram
    if sentence != '':
        lst = []
        sentence = sentence.replace('\n', '')
        sentence = sentence.lower()
        sentence = ''.join(re.sub("[^a-z]+", "", sentence))
        for c in sentence:
            c = c.lower()
            lst.append(c)

    return lst


# tokenizing for bigram model basic setup
def tokenize_biagram_bs(input_file='', sentence=''):
    if input_file != '':
        lst_bigram.clear()
        newline = ''
        with open(input_file, 'r', errors='ignore') as f:
            f = f.readlines()
            for line in f:
                line = line.replace('\n', '')
                line = line.lower()
                newline += re.sub("[^a-z]+", "", line)
            for i in range(len(newline) - 1):
                lst_bigram.append((newline[i], newline[i + 1]))
        return lst_bigram
    if sentence != '':
        lst_b = []
        sentence = sentence.replace('\n', '')
        sentence = sentence.lower()
        sentence = ''.join(re.sub("[^a-z]+", "", sentence))
        for i in range(len(sentence) - 1):
            lst_b.append((sentence[i], sentence[i + 1]))

    return lst_b


# create unigram language model for experience
def mdl_unigram_ex(lst):
    m = {}
    for c in chars_ex:
        m[c] = (lst.count(c) + 0.5) / (len(lst_unigram_ex) + 56 * 0.5)
    return m


# create biogram language model for experience
def mdl_bigram_ex(lst):
    m = {}
    for c in chars_ex:
        for h in chars_ex:
            m[(c, h)] = (lst.count((c, h)) + 0.5) / (lst_unigram_ex.count(c) + math.pow(56, 2) * 0.5)
    return m


# tokenizing for unigram model experience
def tokenize_unigram_ex(input_file='', sentence=''):
    if input_file != '':
        lst_unigram_ex.clear()
        with open(input_file, 'r', errors='ignore') as f:
            f = f.readlines()
            for line in f:
                line = line.replace('\n', '')
                line = ''.join(re.sub("[^a-zA-Z '\-]", "", line))
                for letter in line:
                    lst_unigram_ex.append(letter)
        return lst_unigram_ex
    if sentence != '':
        lst = []
        sentence = sentence.replace('\n', '')
        sentence = ''.join(re.sub("[^a-zA-Z '\-]", "", sentence))
        for c in sentence:
            lst.append(c)

    return lst


# tokenizing for bigram model experience
def tokenize_biagram_ex(input_file='', sentence=''):
    if input_file != '':
        lst_bigram_ex.clear()
        newline = ''
        with open(input_file, 'r', errors='ignore') as f:
            f = f.readlines()
            for line in f:
                line = line.replace('\n', '')
                newline += re.sub("[^a-zA-Z '\-]", "", line)
            for i in range(len(newline) - 1):
                lst_bigram_ex.append((newline[i], newline[i + 1]))
        return lst_bigram_ex
    if sentence != '':
        lst_b = []
        sentence = sentence.replace('\n', '')
        sentence = ''.join(re.sub("[^a-zA-Z '\-]", "", sentence))
        for i in range(len(sentence) - 1):
            lst_b.append((sentence[i], sentence[i + 1]))

    return lst_b


if __name__ == '__main__':

    lst_uni_EN = tokenize_unigram_bs('trainEN.txt')
    lst_unigram_prob_EN = mdl_unigram_bs(lst_uni_EN)
    completeName = os.path.join("unigramEN.txt")
    f = open(completeName, 'w+')
    for lu in lst_unigram_prob_EN:
        f.write('(' + lu + ')' + ' = ' + str(lst_unigram_prob_EN.get(lu)) + '\n')
    f.close()

    # lst_uni_EN_ex = tokenize_unigram_ex('trainEN.txt')
    # lst_unigram_prob_EN_ex = mdl_unigram_ex(lst_uni_EN_ex)

    lst_bio_EN = tokenize_biagram_bs('trainEN.txt')
    lst_biagram_prob_EN = mdl_bigram_bs(lst_bio_EN)
    completeName = os.path.join("bigramEN.txt")
    f = open(completeName, 'w+')
    for lb in lst_biagram_prob_EN:
        f.write('( ' + lb[1] + ' | ' + lb[0] + ' )' + ' = ' + str(lst_biagram_prob_EN.get(lb)) + '\n')
    f.close()

    # lst_bio_EN_ex = tokenize_biagram_ex('trainEN.txt')
    # lst_biagram_prob_EN_ex = mdl_bigram_ex(lst_bio_EN_ex)

    lst_uni_FR = tokenize_unigram_bs('trainFR.txt')
    lst_unigram_prob_FR = mdl_unigram_bs(lst_uni_FR)
    completeName = os.path.join("unigramFR.txt")
    f = open(completeName, 'w+')
    for lu in lst_unigram_prob_FR:
        f.write('(' + lu + ')' + ' = ' + str(lst_unigram_prob_FR.get(lu)) + '\n')
    f.close()

    # lst_uni_FR_ex = tokenize_unigram_ex('trainFR.txt')
    # lst_unigram_prob_FR_ex = mdl_unigram_ex(lst_uni_FR_ex)

    lst_bio_FR = tokenize_biagram_bs('trainFR.txt')
    lst_biagram_prob_FR = mdl_bigram_bs(lst_bio_FR)
    completeName = os.path.join("bigramFR.txt")
    f = open(completeName, 'w+')
    for lb in lst_biagram_prob_FR:
        f.write('( ' + lb[1] + ' | ' + lb[0] + ' )' + ' = ' + str(lst_biagram_prob_FR.get(lb)) + '\n')
    f.close()

    # lst_bio_FR_ex = tokenize_biagram_ex('trainFR.txt')
    # lst_biagram_prob_FR_ex = mdl_bigram_ex(lst_bio_FR_ex)

    lst_uni_GR = tokenize_unigram_bs('trainGR.txt')
    lst_unigram_prob_GR = mdl_unigram_bs(lst_uni_GR)
    completeName = os.path.join("unigramGR.txt")
    f = open(completeName, 'w+')
    for lu in lst_unigram_prob_GR:
        f.write('(' + lu + ')' + ' = ' + str(lst_unigram_prob_GR.get(lu)) + '\n')
    f.close()

    # lst_uni_GR_ex = tokenize_unigram_ex('trainGR.txt')
    # lst_unigram_prob_GR_ex = mdl_unigram_ex(lst_uni_GR_ex)

    lst_bio_GR = tokenize_biagram_bs('trainGR.txt')
    lst_biagram_prob_GR = mdl_bigram_bs(lst_bio_GR)
    completeName = os.path.join("bigramGR.txt")
    f = open(completeName, 'w+')
    for lb in lst_biagram_prob_GR:
        f.write('( ' + lb[1] + ' | ' + lb[0] + ' )' + ' = ' + str(lst_biagram_prob_GR.get(lb)) + '\n')
    f.close()

    # lst_bio_GR_ex = tokenize_biagram_ex('trainGR.txt')
    # lst_biagram_prob_GR_ex = mdl_bigram_ex(lst_bio_GR_ex)

    with open('testsentences.txt', 'r', errors='ignore') as file:
        count = 0
        f = file.readlines()
        for line in f:
            probability_EN = 0
            probability_FR = 0
            probability_GR = 0
            probability_EN_b = 0
            probability_FR_b = 0
            probability_GR_b = 0
            # probability_EN_ex = 0
            # probability_FR_ex = 0
            # probability_GR_ex = 0
            # probability_EN_b_ex = 0
            # probability_FR_b_ex = 0
            # probability_GR_b_ex = 0
            count += 1
            completeName = os.path.join("out" + str(count) + ".txt")
            f = open(completeName, 'w+')
            f.write(line + '\n')
            f.write('UNIGRAM MODEL:' + '\n')

            lst_u = tokenize_unigram_bs(sentence=line)
            lst_b = tokenize_biagram_bs(sentence=line)
            # lst_u_ex = tokenize_unigram_ex(sentence=line)
            # lst_b_ex = tokenize_biagram_ex(sentence=line)

            for element in lst_u:
                f.write('UNIGRAM: ' + element + '\n')

                probability_FR += math.log10(lst_unigram_prob_FR.get(element))
                f.write('FRENCH: ' + 'P(' + element + ') = ' + str(
                    lst_unigram_prob_FR.get(element)) + '==>log prob of sentence so far: ' + str(probability_FR) + '\n')
                probability_EN += math.log10(lst_unigram_prob_EN.get(element))
                f.write('ENGLISH: ' + 'P(' + element + ') = ' + str(
                    lst_unigram_prob_EN.get(element)) + '==>log prob of sentence so far: ' + str(probability_EN) + '\n')
                probability_GR += math.log10(lst_unigram_prob_GR.get(element))
                f.write('GERMANY: ' + 'P(' + element + ') = ' + str(
                    lst_unigram_prob_GR.get(element)) + '==>log prob of sentence so far: ' + str(probability_GR) + '\n')
            max_prob_lang = 'ENGLISH'
            if probability_FR > probability_EN:
                max_prob_lang = 'FRENCH'
            if probability_GR > probability_EN:
                max_prob_lang = 'GERMANY'
                if probability_GR < probability_FR:
                    max_prob_lang = 'FRENCH'
            print(line.replace('\n', '') + '(' + max_prob_lang + ')')
            f.write('\nAccording to the unigram model, the sentence is in ' + max_prob_lang + '\n')
            f.write('--------------------------------------------------------------------------\n')

            for element in lst_b:
                f.write('BIGRAM ' + element[0] + element[1] + '\n')

                probability_FR_b += math.log10(lst_biagram_prob_FR.get(element))
                f.write('FRENCH: ' + 'P(' + element[1] + '|' + element[0] + ')' + '= ' + str(
                    lst_biagram_prob_FR.get(element)) + '==>log prob of sentence so far: ' + str(
                    probability_FR_b) + '\n')
                probability_EN_b += math.log10(lst_biagram_prob_EN.get(element))
                f.write('ENGLISH: ' + 'P(' + element[1] + '|' + element[0] + ')' + '= ' + str(
                    lst_biagram_prob_EN.get(element)) + '==>log prob of sentence so far: ' + str(
                    probability_EN_b) + '\n')
                probability_GR_b += math.log10(lst_biagram_prob_GR.get(element))
                f.write('GERMANY: ' + 'P(' + element[1] + '|' + element[0] + ')' + '= ' + str(
                    lst_biagram_prob_GR.get(element)) + '==>log prob of sentence so far: ' + str(
                    probability_GR_b) + '\n')
            max_prob_lang = 'ENGLISH'
            if probability_FR_b > probability_EN_b:
                max_prob_lang = 'FRENCH'
            if probability_GR_b > probability_EN_b:
                max_prob_lang = 'GERMANY'
                if probability_GR_b < probability_FR_b:
                    max_prob_lang = 'FRENCH'
            print(line.replace('\n', '') + '(' + max_prob_lang + ')')
            f.write('\nAccording to the biagram model, the sentence is in ' + max_prob_lang)
            f.close()
            # for element in lst_u_ex:
            #     f.write('UNIGRAM: ' + element + '\n')
            #
            #     probability_FR_ex += math.log10(lst_unigram_prob_FR_ex.get(element))
            #     f.write('FRENCH: ' + 'P(' + element + ') = ' + str(
            #         lst_unigram_prob_FR_ex.get(element)) + '==>log prob of sentence so far: ' + str(probability_FR_ex) + '\n')
            #     probability_EN_ex += math.log10(lst_unigram_prob_EN_ex.get(element))
            #     f.write('ENGLISH: ' + 'P(' + element + ') = ' + str(
            #         lst_unigram_prob_EN_ex.get(element)) + '==>log prob of sentence so far: ' + str(probability_EN_ex) + '\n')
            #     probability_GR_ex += math.log10(lst_unigram_prob_GR_ex.get(element))
            #     f.write('GERMANY: ' + 'P(' + element + ') = ' + str(
            #         lst_unigram_prob_GR_ex.get(element)) + '==>log prob of sentence so far: ' + str(probability_GR_ex) + '\n')
            # max_prob_lang = 'ENGLISH'
            # if probability_FR_ex > probability_EN_ex:
            #     max_prob_lang = 'FRENCH'
            # if probability_GR_ex > probability_EN_ex:
            #     max_prob_lang = 'GERMANY'
            #     if probability_GR_ex < probability_FR_ex:
            #         max_prob_lang = 'FRENCH'
            # print(line.replace('\n', '') + '(' + max_prob_lang + ')')
            # f.write('\nAccording to the unigram model, the sentence is in ' + max_prob_lang + '\n')
            # f.write('--------------------------------------------------------------------------\n')
            #
            # for element in lst_b_ex:
            #     f.write('BIGRAM ' + element[0] + element[1] + '\n')
            #
            #     probability_FR_b_ex += math.log10(lst_biagram_prob_FR_ex.get(element))
            #     f.write('FRENCH: ' + 'P(' + element[1] + '|' + element[0] + ')' + '= ' + str(
            #         lst_biagram_prob_FR_ex.get(element)) + '==>log prob of sentence so far: ' + str(
            #         probability_FR_b_ex) + '\n')
            #     probability_EN_b_ex += math.log10(lst_biagram_prob_EN_ex.get(element))
            #     f.write('ENGLISH: ' + 'P(' + element[1] + '|' + element[0] + ')' + '= ' + str(
            #         lst_biagram_prob_EN_ex.get(element)) + '==>log prob of sentence so far: ' + str(
            #         probability_EN_b_ex) + '\n')
            #     probability_GR_b_ex += math.log10(lst_biagram_prob_GR_ex.get(element))
            #     f.write('GERMANY: ' + 'P(' + element[1] + '|' + element[0] + ')' + '= ' + str(
            #         lst_biagram_prob_GR_ex.get(element)) + '==>log prob of sentence so far: ' + str(
            #         probability_GR_b_ex) + '\n')
            # max_prob_lang = 'ENGLISH'
            # if probability_FR_b_ex > probability_EN_b_ex:
            #     max_prob_lang = 'FRENCH'
            # if probability_GR_b_ex > probability_EN_b_ex:
            #     max_prob_lang = 'GERMANY'
            #     if probability_GR_b_ex < probability_FR_b_ex:
            #         max_prob_lang = 'FRENCH'
            # print(line.replace('\n', '') + '(' + max_prob_lang + ')')
            # f.write('\nAccording to the biagram model, the sentence is in ' + max_prob_lang)
            # f.close()
