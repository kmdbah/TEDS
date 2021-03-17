# The training data file name has to be provided in line 172, validation data in 176
# Dependent variable should be named "binary".Text data, when present, should be the last column, and named "text"
# The confusion matrix will be in the output file nn_validation_confusion.png and validation data probabilities in Validation_NN.csv
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn import model_selection
import pandas
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

kf = KFold(n_splits=10)


#Assumptions - Dependent variable is named Binary, Text is the last column and is named text

stemmer = SnowballStemmer("english")
i=0

#Plotting Confusion Matrix
def show_confusion_matrix(C,class_labels=['0','1']):

    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            '%d'%(tn),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            '%d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            '%d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            '%d'%(tp),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'Error: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'Error: %.2f'%(fn / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,' ',
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            ' ',
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()





def remove_punctuation(s):
    no_punct = ""
    for letter in s:
        if letter not in string_punctuation:
            no_punct += letter
    return no_punct

#Calculate Lift
def calc_Decile(y_pred,y_actual,y_prob,bins=10):
    cols = ['ACTUAL','PROB_POSITIVE','PREDICTED']
    data = [y_actual,y_prob[:,1],y_pred]
    dfa = pandas.DataFrame(dict(zip(cols,data)))

    #Observations where y=1
    total_positive_n = dfa['ACTUAL'].sum()
    #Total Observations
    dfa= dfa.reset_index()
    total_n = dfa.index.size
    natural_positive_prob = total_positive_n/float(total_n)
    dfa = dfa.sort_values(by=['PROB_POSITIVE'], ascending=[False])
    dfa['rank'] = dfa['PROB_POSITIVE'].rank(method='first')
    #Create Bins where First Bin has Observations with the
    #Highest Predicted Probability that y = 1
    dfa['BIN_POSITIVE'] = pandas.qcut(dfa['rank'],bins,labels=False)
    pos_group_dfa = dfa.groupby('BIN_POSITIVE')
    #Percentage of Observations in each Bin where y = 1
    lift_positive = pos_group_dfa['ACTUAL'].sum()/pos_group_dfa['ACTUAL'].count()
    lift_index_positive = lift_positive/natural_positive_prob

    #result1 = result.reset_index()
    #Consolidate Results into Output Dataframe
    lift_df = pandas.DataFrame({'LIFT_POSITIVE':lift_positive,
                               'LIFT_POSITIVE_INDEX':lift_index_positive,
                               'BASELINE_POSITIVE':natural_positive_prob})

    return lift_df



#Read file
##Training Data
df = pandas.read_csv('training data.csv',encoding="ISO-8859-1")

##Validation Data
df_test = pandas.read_csv('test data.csv',encoding="ISO-8859-1")

Text_present = input('Text in the data? Yes/No: ')
if Text_present =='Yes':
    #Read the text column---Last Column (Assumption)
    string_punctuation = '''()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''
    stop = stopwords.words('english')
    df.iloc[ :, -1] = df.iloc[ :, -1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    for row in df['text']:
        df.iloc[ i, -1] = remove_punctuation(row)
        i=i+1
    df['text'] = df['text'].str.replace("!"," !")
    df['text'] = df['text'].apply(word_tokenize)
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['text'] = df['text'].apply(lambda x : " ".join(x))
    Text_Column = df.iloc[ :, -1:]
    #Get TFIDF Scores
    sklearn_tfidf = TfidfVectorizer(min_df=.01, max_df =.95, stop_words="english",use_idf=True, smooth_idf=False, sublinear_tf=True)
    sklearn_representation = sklearn_tfidf.fit_transform(Text_Column.iloc[:, 0].tolist())
    Tfidf_Output = pandas.DataFrame(sklearn_representation.toarray(), columns=sklearn_tfidf.get_feature_names())

    #Append the column to the final dataset
    Input = pandas.concat([df, Tfidf_Output], axis=1)
    Input = Input.drop('text', 1)
else:
    Input = df

if Text_present =='Yes':
    #Read the text column---Last Column (Assumption)
    string_punctuation = '''()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''
    stop = stopwords.words('english')
    df_test.iloc[ :, -1] = df_test.iloc[ :, -1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    for row in df_test['text']:
        df_test.iloc[ i, -1] = remove_punctuation(row)
        i=i+1
    df_test['text'] = df_test['text'].str.replace("!"," !")
    df_test['text'] = df_test['text'].apply(word_tokenize)
    df_test['text'] = df_test['text'].apply(lambda x: [stemmer.stem(y) for y in x])
    df_test['text'] = df_test['text'].apply(lambda x : " ".join(x))
    Text_Column = df_test.iloc[ :, -1:]
    #Get TFIDF Scores
    sklearn_tfidf = TfidfVectorizer(min_df=.01, max_df =.95, stop_words="english",use_idf=True, smooth_idf=False, sublinear_tf=True)
    sklearn_representation = sklearn_tfidf.fit_transform(Text_Column.iloc[:, 0].tolist())
    Tfidf_Output = pandas.DataFrame(sklearn_representation.toarray(), columns=sklearn_tfidf.get_feature_names())

    #Append the column to the final dataset
    Input_test = pandas.concat([df_test, Tfidf_Output], axis=1)
    Input_test = Input.drop('text', 1)
else:
    Input_test = df_test

X = Input.loc[:, Input.columns != 'binary']
Y = Input['binary']
X_Validation_Score = X


X_test = Input_test.loc[:, Input_test.columns != 'binary']
Y_test = Input_test['binary']
X_test_Validation_Score = X_test
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, train_size=0.6, random_state=1)

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X)

# Transformations to the data:
# X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X = scaler.transform(X)

classifier = MLPClassifier(hidden_layer_sizes=(25,25,25)).fit(X,Y)
#mlp.fit(X,Y)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(25, 25, 25), learning_rate='constant',
       learning_rate_init=0.001, max_iter=30, momentum=0.6,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.01, validation_fraction=0.1,
       verbose=False, warm_start=False)

#Prediction
#predictions = mlp.predict(X_test)

Y_pred = model_selection.cross_val_predict(classifier, X_test, Y_test, cv=5)
confusion_matrix=confusion_matrix(np.array(Y_test),Y_pred)

#Validation score table
y_prob = cross_val_predict(classifier, X_test, Y_test, method='predict_proba')


validation_columns = ['Predicted_Probability','Y','Y_pred']
validation_data = [y_prob[:,1],Y_test,Y_pred]
Validation_NN = pandas.DataFrame(dict(zip(validation_columns,validation_data)))
Validation_NN = pandas.concat([Validation_NN, X_test_Validation_Score], axis=1)
Validation_NN = Validation_NN.sort_values(by=['Predicted_Probability'], ascending=[False])
Validation_NN.loc[Validation_NN['Predicted_Probability']>0.4999999, 'Y_pred'] = "1"
Validation_NN.loc[Validation_NN['Predicted_Probability']<0.5, 'Y_pred'] = "0"



#Decile chart
Decile_Chart = calc_Decile(Y_pred,Y_test,y_prob)
Decile_Chart['Bin']=abs(10-Decile_Chart.index)
Decile_Chart = Decile_Chart.sort_values(by=['Bin'], ascending=[False])

plt.subplot(221)
plt.bar(Decile_Chart['Bin'], Decile_Chart['LIFT_POSITIVE_INDEX'], align='center',)
plt.xlabel('Bins')
plt.title('Decile Chart')
plt.xticks(Decile_Chart['Bin'])

# Lift chart
cols = ['ACTUAL','PROB_POSITIVE','PREDICTED']
data = [Y_test,y_prob[:,1],Y_pred]
Lift_data = pandas.DataFrame(dict(zip(cols,data)))
Lift_data = Lift_data.sort_values(by=['PROB_POSITIVE'], ascending=[False])
Lift_data['cum_actual'] = Lift_data.ACTUAL.cumsum()

Lift_data = Lift_data.reset_index()
del Lift_data ['index']
p = Lift_data['cum_actual']
d = Lift_data.index+1

plt.subplot(222)
plt.plot(d,p,color='blue',marker='o',markersize=.2)
total_positive_n = Lift_data['ACTUAL'].sum()
total_positive_count = Lift_data['ACTUAL'].count()
plt.plot([1,total_positive_count],[1,total_positive_n],color='red',marker='o')

plt.legend(['Cumulative 1 when sorted using predicted values'])
plt.title("Lift Chart")
plt.xlabel("#Cases")
plt.grid()
plt.savefig('NN_Decile_Lift.png')
show_confusion_matrix(confusion_matrix, ['0', '1'])
plt.show()
plt.savefig('NN_Validation_Confusion.png')

Validation_NN = Validation_NN.rename(columns={'Predicted_Probability': 'Prob of 1', 'Y_pred': 'Predicted', 'Y': 'Actual'})
Validation_NN.to_csv('Validation_NN.csv', index_label=[Validation_NN.columns.name], index=False)
