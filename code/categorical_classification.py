import numpy
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import plot_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# fix random seed for reproducibility
numpy.random.seed(7)

gpt1_data = pd.read_csv('out_gpt1.csv', sep=',')
gpt2_data = pd.read_csv('out_gpt2.csv', sep=',')
xlnet_data = pd.read_csv('out_xlnet.csv', sep=',')
real_data = pd.read_csv('real.csv', sep=',')
X_train = numpy.concatenate((gpt1_data['review'].values, gpt2_data['review'].values, xlnet_data['review'].values, real_data['review'].values))
y_train = numpy.concatenate((numpy.zeros(2000, dtype=int),numpy.ones(2000, dtype=int), numpy.ones(2000, dtype=int) * 2, numpy.ones(2000, dtype=int) * 3), axis=0)
y_train = pd.get_dummies(y_train).values

vocabulary_size = 50000
embedding_vecor_length = 100
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen=embedding_vecor_length)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

# create the model
top_words = vocabulary_size
max_review_length = embedding_vecor_length
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_vs_epochs.png')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_vs_loss.png')
plt.clf()

y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred.argmax(axis=1), y_test.argmax(axis=1))
df_cm = pd.DataFrame(cm, index = ['gpt1','gpt2','xlnet','real'],
                  columns = ['gpt1','gpt2','xlnet','real'])
# plt.figure(figsize = (10,7))
sns_heatmap = sns.heatmap(df_cm, annot=True, fmt='g')
plt.savefig("sns_heatmap.png")