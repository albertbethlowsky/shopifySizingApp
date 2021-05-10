from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn import svm, datasets
from matplotlib import style # setting styles for plots


#PLOTTING RUNWAY:
dataset = pd.read_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_runway.csv')
filename = '_runway.png'
dataset = dataset[['fit','product_size', 'bust_size_num_eu', 'bust_size_cat', 'height_meters', 'weight_kg', 'product_category', 'age', 'body_type','bmi']].copy()

#PLOTTING CHPT3
#dataset = pd.read_csv('C:/Users/Frederik/Desktop/shopifySizingApp/chpt3_sizing_app/Algorithm/Data/clean_chpt3.csv')
#filename = '_chpt3.png'
#dataset = dataset[['fit','product_size', 'bust_size_num_eu', 'bust_size_cat', 'height_meters', 'weight_kg', 'product_category', 'age', 'body_type', 'bmi','gender']].copy()

le.fit(dataset.fit)
fit_label = le.transform(dataset.fit)

le.fit(dataset.body_type)
body_type_label = le.transform(dataset.body_type)

le.fit(dataset.product_category)
product_category_label = le.transform(dataset.product_category)

le.fit(dataset.bust_size_cat)
bust_size_cat_label = le.transform(dataset.bust_size_cat)


dataset.fit                     = fit_label
dataset.body_type               = body_type_label
dataset.product_category        = product_category_label
dataset.bust_size_cat           = bust_size_cat_label

from sklearn.preprocessing import scale

corr_matrix = dataset.corr()

#corr_matrix
print(dataset.head(20))
print(corr_matrix.head(20))
fig = plt.figure()
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.show()
s = './plotting/heatmap' + filename
fig.savefig(s)

print('Pairs of columns that have correlation greater than 0.5: ')
lim = 0.5
corr_cols = []
for i in range(corr_matrix.shape[0]):
    for j in range(i + 1, corr_matrix.shape[1]):
        if corr_matrix.iloc[i, j] > lim:
            pair = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_cols.append(pair)
            print('({}, {})'.format(*pair))
            
print('These columns are to be inspected more closely')
print(corr_cols)


# # 4. Scatter Matrix of categories with corr>0.5:
dataset = dataset[['product_size', 'bust_size_cat', 'height_meters', 'weight_kg', 'product_category', 'age', 'bmi']].copy()
style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (16, 7)
sns.pairplot(data=dataset, hue="product_size")
s = './plotting/ScatterMatrix' + filename
plt.savefig(s)
plt.show()

# #get list of unique values
# #'product_size', 'bust_size_num_eu', 'bust_size_cat', 'bmi', 'product_category', 'age', 'body_type'
# #'fit','product_size', 'bust_size_num_eu', 'bust_size_cat', 'bmi', 'product_category', 'age', 'body_type'
# dataset = dataset[['fit','bmi','bust_size_num_eu']].copy() 
# #itemid_1
# #bottom
# #sizes[1-10]
# #small, fit, large
# #plot_correlation(dataset)
# X = dataset.iloc[:,1:3] 
# y = dataset.iloc[:,0] #size
# X = X.to_numpy()
# y = y.to_numpy()
# feature_names = ['bmi','bust_size_num_eu']
# classes = ['0','1','2']

# #SVM PLOTTING - 3D
# # #make it binary classification problem
# # X = X[np.logical_or(Y==0,Y==1)]
# # Y = Y[np.logical_or(Y==0,Y==1)]

# # model = svm.SVC(kernel='linear')
# # clf = model.fit(X, Y)
# # # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# # # Solve for w3 (z)
# # z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

# # tmp = np.linspace(-5,5,30)
# # x,y = np.meshgrid(tmp,tmp)

# # fig = plt.figure()
# # ax  = fig.add_subplot(111, projection='3d')
# # ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
# # ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
# # ax.plot_surface(x, y, z(x,y))
# # ax.view_init(30, 60)
# # plt.show()

# #SVM PLOTTING - 2D

# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# model = svm.SVC(kernel='linear')
# clf = model.fit(X, y)

# fig, ax = plt.subplots()
# # title for the plots
# title = ('Decision surface of linear SVC ')
# # Set-up grid for plotting.
# #print(X)

# X0, X1 = X[:,0], X[:,1] 
# xx, yy = make_meshgrid(X0, X1)

# plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel(feature_names[1])
# ax.set_xlabel(feature_names[0])
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)
# ax.legend()
# plt.show()