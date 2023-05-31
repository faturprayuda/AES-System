# import libraries

# import dependencies of nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# using pandas
import pandas as pd

def similarity(text, check_text) :

    # tokenization
    X_list = word_tokenize(text)
    Y_list = word_tokenize(check_text)

    # sw contains the list of stopwords
    sw = stopwords.words('indonesian')
    l1 =[];l2 =[]

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)

    # cosine formula
    c = 0
    for i in range(len(rvector)):
        c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return round(cosine * 100)

def checkPlag(pathfile) :

  df = pd.read_excel(pathfile, engine='openpyxl')
  df.dropna(axis=0, how='all', inplace=True)
  df.drop(columns='Timestamp', inplace=True, axis=1)

  temp_data = []

  for i in df:
      temp_data.append(df[i])

  data = []
  for i in range(len(temp_data[0])):
      tmp_separate = []
      for j in temp_data:
          tmp_separate.append(j[i])
      data.append(tmp_separate)

  arr = []
  plag_arr0 = []
  plag_arr1 = []
  plag_arr2 = []
  plag_arr3 = []
  plag_arr4 = []
  for i in range(len(data)) :
#   for i in range(1) :
      key = {}
      key['nama'] = data[i][0]
      key['kelas'] = data[i][1]
      key['absen'] = str(data[i][2])
      for j in range(0, len(data)) :
          plagiarism0 = similarity(data[i][3], data[j][3])
          plagiarism1 = similarity(data[i][4], data[j][4])
          plagiarism2 = similarity(data[i][5], data[j][5])
          plagiarism3 = similarity(data[i][6], data[j][6])
          plagiarism4 = similarity(data[i][7], data[j][7])
          plag_arr0.append(plagiarism0)
          plag_arr1.append(plagiarism1)
          plag_arr2.append(plagiarism2)
          plag_arr3.append(plagiarism3)
          plag_arr4.append(plagiarism4)
      key['plagiarism1'] = round(sum(plag_arr0) / len(plag_arr0))
      key['plagiarism2'] = round(sum(plag_arr1) / len(plag_arr1))
      key['plagiarism3'] = round(sum(plag_arr2) / len(plag_arr2))
      key['plagiarism4'] = round(sum(plag_arr3) / len(plag_arr3))
      key['plagiarism5'] = round(sum(plag_arr4) / len(plag_arr4))
      arr.append(key)
      
  return arr
