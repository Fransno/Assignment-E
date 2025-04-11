from preprocessing import preprocessing
from supervised import supervisedrf, supervisedxgb
from unsupervised import unsupervised_kmeans, unsupervised_dbscan
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


df = preprocessing()

sns.pairplot(df, hue="Sleep Disorder")
plt.suptitle("Sammenheng mellom variabler etter medikament", y=1.02)
plt.show()

#X = df.drop(columns=['Sleep Disorder'])
#y = df['Sleep Disorder']
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#supervisedrf(x_train, x_test, y_train, y_test)
#supervisedxgb(x_train, x_test, y_train, y_test)
#
#unsupervised_kmeans(X)
#unsupervised_dbscan(X)