# general training data maker thing 

# data libraries
import pandas as pd
import numpy as np


# NLP Libraries
import re
import fugashi
tagger = fugashi.Tagger()


# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()


# functions 

# for cleaning the raw paragraphs 
def remove_numbers(string):
    '''literally just remove all numbers from a string, we don't want them'''
    cleaned = re.sub(pattern = '\d+', repl = '', string = string)
    return cleaned

def remove_special_characters(string):
    '''using the list of pattern characters, remove them from the original data'''
    cleaned = string[:] # makes a copy -- don't overwrite the original!
    patterns = removal_characters['character'].values
    for pat in patterns:
        cleaned = re.sub(pattern = pat, repl = '', string = cleaned)
    return cleaned

def remove_whitespace(string):
    '''remove all whitespace charaters from a string'''
    return re.sub(pattern = ' ', repl = '', string = string)


def clean_string(string):
    '''perform a complete cleaning on the given string for training'''
    temp = remove_special_characters(string)
    temp = remove_numbers(temp)
    temp = remove_whitespace(temp)
    return temp


def make_hit_arrays(string, hit, r):
    '''return a shortened version of STRING that is centered around HIT with R characters on each side of it
    STRING: any string
    HIT: any word
    returns:
    STRING wihout HIT, N: any positive integer that is the location of HIT in STRING. will return the first occurence of the first character of HIT'''
    ns = [] # this means there is no occurence of HIT if empty
    hit_strings = []
    
    if (hit in string): # first see that HIT is actually in STRING to avoid errors
        size_hit = len(hit) # how many characters to examine at once 
        
        for n in range(len(string) - size_hit): # itterate through STRING until you reach HIT
            
            if string[n:n+size_hit] == hit: # iterating till we reach HIT
                ns.append(n)
                hit_strings.append(string[max(0,n - r): min(n + r, len(string))])
    return hit_strings

def remove_hit(string, hit):
    if (hit in string): # first see that HIT is actually in STRING to avoid errors
        size_hit = len(hit) # how many characters to examine at once 

    for n in range(len(string) - size_hit): # itterate through STRING until you reach HIT
        if string[n:n+size_hit] == hit: # iterating till we reach HIT
            mod_string = string[0:n] + string[n+size_hit:] # create a modified string without HIT
            return mod_string

def tokenize(string):
    '''given a string STRING, return the tokens in lemma form in the form of a numpy array'''
    tokens = np.array([word.surface for word in tagger(string)]) # store the raw tokens of each token
    return tokens


# supplemental functions I made in post: 

def remove_kana(arr):
    new_arr = []
    for a in arr:
        if a not in katakana and a not in hiragana: 
            new_arr.append(a)
    return new_arr 

def remove_none(arr):
    new_arr = []
    for a in arr:
        if a != None:
            new_arr.append(a)
    return new_arr 


# this turns the string into lemma (and are cleaned for issues found later)
def lemmize(string):
    lemma = np.array([word.feature.lemma for word in tagger(string)])# store the lemma 
    fin_lemma = remove_kana(remove_none(lemma)) # nested functions here 
    return fin_lemma



# now to clean the data fully: 

def make_training_matrix(data, hit = '二世', r = 50, num_occurences = True):
    '''data must have columns: 'link','numWords','paragraphs' 
        at a minimum, hit is a word that we want to fit for
        r sets the range of characters we keep on either side of the hit
        num_occurences sets if we want our output matrix to give the number of occurences of a lemma (True) 
            or just if the lemma is present in the array (False)'''
    # first things, filter out all data points that have no hits of the words we care about
    model_data = data[data['numWords'] > 0] 
    cleaned_texts = model_data['paragraphs'].apply(clean_string).values
    new_col = {'cleanText': cleaned_texts}
    model_data = model_data.assign(**new_col) # now we have a model data set that has an additional column 

    # making the hit arrays 

    new_data_points = [] # append this then make an array
    data_index = np.array([]) # an array that contains the original index value of the new data points 
    c = 0
    for s in model_data['cleanText'].values:
        new_strings = make_hit_arrays(s, hit = '二世', r = r)
        data_index = np.append(data_index, np.zeros(len(new_strings)) + c)
        new_data_points.extend(new_strings)
        c += 1

    feature_data = pd.DataFrame(data = {'text': new_data_points})# the final data frame with cleaned arrays with hits at the cetner 
    data_index = [int(i) for i in data_index] # so the index values are ints in the final column 

    # take out the hits from the data since we don't want to include that in our clustering 
    no_hits = feature_data['text'].apply(remove_hit, hit = hit)
    feature_data = feature_data.assign(**{'noHit':no_hits})

    # create lemma arrays
    lem = feature_data['noHit'].apply(lemmize)
    new_col = {'lemma': lem}
    feature_data = feature_data.assign(**new_col)
    print(feature_data.head())
    
    # now find the unique lemma to create our columns for our training matrix
    unique_lemma = []
    lemma_counts = {} # empty dictionary 
    for lemma in feature_data['lemma']:
        for lem in lemma:
            if lem not in unique_lemma: 
                unique_lemma.append(lem)
                lemma_counts[lem] = 1
            else:
                lemma_counts[lem] +=1
                
    lemma_df = pd.DataFrame.from_dict(data = lemma_counts, orient = 'index').rename({0:'Occurences'}, axis = 1)
    # testing error ---------
    print(lemma_df.head())
    lemma_df = lemma_df[lemma_df['Occurences'] >= 10]
    training_lemma = lemma_df.index # list of the lemma in order from the lemma data frame 

    # now to create the values for the number of occurences in each array:
    data_point_lemma_counts = {} # empty dictionary that will be used to make a data frame
    count_lemma = len(training_lemma)
    # for i in range(count_lemma):
    #     data_point_lemma_counts[i] = [] # set them to empty arrays 
    
    # c = 0 # this keeps track of the index value we are on -- don't need since using enumerate 
    for c,dp in enumerate(feature_data_v2.values): # dp meaning data point 
        # print('first loop:',dp)
        count_array = np.zeros(count_lemma)
        # print('count array:', count_array)
        for d in dp[0]:  # iterate through the individual lemma from the array for each data point 
            # print('\tsecond loop',d)
            for i in range(count_lemma): # this gets the index of the lemma that we can add to the count_lemma array 
                if d == training_lemma[i]:
                    # print("\t\tfound hit" ,training_lemma[i])
                    if num_occurences:
                        count_array[i] += 1
                    count_array[i] = 1 # just set to 1 otherwise 
        data_point_lemma_counts[c] = count_array
        # c += 1
    theta_matrix = pd.DataFrame.from_dict(data = data_point_lemma_counts, orient = 'index', columns = training_lemma)

    return theta_matrix # and this should be everything ! 
            



# put all the tables of filtering data here : 

patterns = (['！', '!','\?','@','#','\$','%','\^','&', '#', '\d+', '[a-zA-Z]+', '〇', '「','」',
             '-','_','№', r'\\','\.','\,','/','~','`','|','〆', "'", '，',
            ':',';','\(', '\)','：','；', '、' , '｜', '»','«','•', "＊", '|', '|','“' ]) # these are characters found manually through scanning raw data points
stop_words_data = pd.read_csv('stopwords-ja.txt', sep = ' ', header = None) # these are known japanese stop words
patterns.extend(stop_words_data[0].values)
removal_characters = pd.DataFrame(data = {'character':patterns}) # put the removal characters into a data frame 


# removing single character tokens using these two arrays: 
katakana = (['ア','イ','ウ','エ','オ',
             'カ','キ','ク','ケ','コ',
             'ガ','ギ','グ','ゲ','ゴ',
             'サ','シ','ス','セ','ソ',
             'ザ','ジ','ズ','ゼ','ゾ',
             'タ','チ','ツ','テ','ト',
             'ダ','ヂ','ヅ','デ','ド',
             'ナ','ニ','ヌ','ネ','ノ',
             'ハ','ヒ','フ','ヘ','ホ',
             'パ','ピ','プ','ペ','ポ',
             'バ','ビ','ブ','ベ','ボ',
             'マ','ミ','ム','メ','モ',
             'ヤ','ヨ','ユ',
             'ラ','リ','ル','レ','ロ',
             'ワ','ヲ','ン']) # note I didn't remove the 2 character kana out of laziness 
hiragana = (['あ','い','う','え','お',
             'か','き','く','け','こ',
             'が','ぎ','ぐ','げ','ご',
             'さ','し','す','せ','そ',
             'ざ','じ','ず','ぜ','ぞ',
             'た','ち','つ','て','と',
             'だ','ぢ','づ','で','ど',
             'な','に','ぬ','ね','の',
             'は','ひ','ふ','へ','ほ',
             'ば','び','ぶ','べ','ぼ',
             'ぱ','ぴ','ぷ','ぺ','ぽ',
             'ま','み','む','め','も',
             'や','ゆ','よ',
             'ら','り','る','れ','ろ',
             'わ','を','ん'])