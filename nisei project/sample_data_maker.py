### this makes a sample data set that can be classified by translators! ###

import pandas as pd
import numpy as np
import re
from pathlib import Path  


# args to use throughout #
data_path = '../../data/output_5000_redo' # where is the json file you want to read in?
filename = 'nisei_data_1900s' # what do you want to name your sample data to be labelled?
years = np.arange(1900,1910).astype(int).astype(str) # i think this works -- add one more year than you want
num_to_sample = 100 # how many data points do you want to collect and split?

# importing the data and some pre-filtering #
data = pd.read_json(f'{data_path}.json') # data set path for the big file Ryan made
data = data[data['numWords']> 0] # takes out false occurences
data_cleaned = data[['link','numWords','paragraphs']] # only information needed for these purposes


### cleaning functions ###

'''takes a long string of the raw OCR and splits it into shorter string with only
one occurence of the hit word with r characters on either side'''
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


'''filters DATA by the number of words you would like in your sample data set'''
def get_data_by_num_words(data, num_words = [1], num_samples =100):
    '''INPUT: Data: a data frame with the assumed column labls of link, numWords, paragraphs
        num_words: list-like of valid numWords in each data sample of the hit phrase
        num_samples: the number of data points you want back to categorize
    OUTPUT: a data frame that is ready to be translated'''
    
     # make an array of num_words 
    label_data = data[data['numWords'].isin(num_words)] # only keeps data that has the num words we want
    sample_table = label_data.sample(num_samples, random_state = 42) # sample the data randomly num_samples times 


    # add the label column: 
    labels_col = np.zeros(int(sample_table.shape[0])) - 99 # -99 so we know if it was labelled yet 
    data_final = sample_table.assign(**{'label': labels_col})

    return data_final
        
'''splits up your data points into sub data points by the number of hit words in each data point.'''
def split_data_by_num_words(data, hit = '二世', r = 25):
    '''given DATA with column numWords, split the data into rows of hit arrays 
    that share all column information except for paragraphs, which is now a split version, with numWords = 1. Then remove the original rows'''
    new_data_points = [] # an array of arrays that have the column values for each new data point 

    data_index = np.array([]) # an array that contains the original index value of the new data points 
    c = 0
    # for s in data['Paragraphs'].values:
    #     new_strings = make_hit_arrays(s, hit = '二世', r = r)
    #     data_index = np.append(data_index, np.zeros(len(new_strings)) + c)
    #     new_data_points.extend(new_strings)
    #     c += 1
    for s in data.values:
        hit_arrs = make_hit_arrays(s[2], hit = hit, r = r)
        for h in hit_arrs:
            new_data_points.append([s[0], 1, h]) # oiginal link, now only 1 numwords, new hit array, same label 
    
    new_data = pd.DataFrame(data = new_data_points, columns = data.columns)# the final data frame with cleaned arrays with hits at the center 
    return new_data


'''filters DATA to only return data points within the year range you give'''
def get_data_by_years(data, years = ['1923','1924','1925'], num_samples = 100):
    '''do the same thing as getting numwords data but now do it by getting the year out from the link
    YEARS must be a string'''
    year_arr = data['link'].str.extract(pat = r'(19\d\d)')
    data_with_year = data.assign(**{'year':year_arr})
    data_with_valid_years = data_with_year[data_with_year['year'].isin(years)]
    data_valid = (data_with_valid_years.sample(np.min([num_samples, data_with_valid_years.shape[0]]), # return the minimum of the number to sample and the number possible to sample
                          random_state = 42))
    return data_valid.drop('year', axis = 1)



### actually making sample data set ###

# use as a sandbox to make your data set
# EX: give a data set that only contains papers from the 1920s

data_to_label = (get_data_by_years(data = data_cleaned, 
                                   years = years,
                                   num_samples = num_to_sample))

data_to_label_split = split_data_by_num_words(data_to_label)


labels_arr = np.zeros(data_to_label_split.shape[0]) - 99 # assign labels that show it is not translated yet  

sample_table = data_to_label_split.assign(**{'label': labels_arr}).drop('numWords', axis = 1)


### Now export the new table as a CSV ###
filepath = Path(f'C:\\Users\\alica\\Documents\\URAP\\data\\{filename}.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)  # makes a directory for the CSV file to be written into 
sample_table.to_csv(filepath) 

print(f'Data set has been successfully saved to \nlocation: {filepath} \nwith name: {filename}.csv')

