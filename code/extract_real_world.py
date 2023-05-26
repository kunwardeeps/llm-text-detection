import pandas as pd
import os, sys

directory = 'hotels/data/'
cities = ['chicago/', 'new-york-city/', 'san-francisco/', 'london/', 'las-vegas/']
review_list = []
for city in cities:
    for filename in os.listdir(directory + city):
        file_with_path = directory + city + filename
        #print(file_with_path)
        try:
            df = pd.read_csv(file_with_path, sep='\t', names=["date", "title", "review"], index_col=False)
        except:
            #print('Parse error in file ' + filename)
            continue
        for review in df['review']:
            if not isinstance(review, str):
                continue
            review = ' '.join(review.split()[:100])
            review_list.append([review, 'real'])
            if len(review_list) > 1999:
                output_df = pd.DataFrame(review_list, columns=['review', 'model_type'])
                output_df.to_csv('real.csv', sep=',', index=False)
                sys.exit()
