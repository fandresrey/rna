from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array as myarray
from joblib import dump, load



iris = load_iris()

# notas= pd.read_csv("DSN.csv")
# print(notas)
#abc = myarray.array('i', [2.5, 4.9, 6.7])


target = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                  4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, ])
targetName = np.array(['do', 're', 'mi', 'fa', 'sol', 'la', 'si'])
dataName = np.array(['FREQ'])
data = np.array([[266], [261], [263], [268], [262], [260], [267], [259], [256], [294], [298], [286], [292], [300], [296], [293], [291], [297], [331], [326], [328], [340], [327], [325], [322], [339], [336], [354], [345], [347], [
                352], [351], [341], [360], [356], [350], [385], [394], [387], [392], [383], [388], [390], [398], [395], [440], [430], [447], [446], [445], [434], [433], [432], [435], [488], [489], [499], [497], [486], [487], [500], [498], [493]])


X_entrena, X_test, y_entrena, y_test = train_test_split(data, target)


arbol = DecisionTreeClassifier()
arbol.fit(X_entrena, y_entrena)

arbol.score(X_test, y_test)  # 0.921052


export_graphviz(arbol, out_file='arbol.dot', class_names=targetName,
                feature_names=dataName, impurity=False, filled=True)

with open('arbol.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# grafico de barras
caract = data.shape[1]
plt.barh(range(caract), arbol.feature_importances_)
plt.yticks(np.arange(caract), dataName)
plt.xlabel('Importancia de las características')
plt.ylabel('Características')
plt.show()


# niveles de desicon del arbol
arbol = DecisionTreeClassifier(max_depth=4)
arbol.fit(X_entrena, y_entrena)
arbol.score(X_test, y_test)  # 0.921052
arbol.score(X_entrena, y_entrena)  # 0732142.9
dump(arbol, 'filename.joblib')
#arbol = load('filename.joblib')


test=arbol.predict(np.c_[498])
print(test)
# n_classes = 3
# plot_colors = 'bry'
# plot_step = 0.02

# for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
#                                [1, 2], [1, 3], [2, 3]]):

#     X = data[pair]
#     y = target

#     # entrena algoritmo
#     clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
#     plt.subplot(2, 3, pairidx + 1)

#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                          np.arange(y_min, y_max, plot_step))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

#     plt.xlabel(dataName[pair[0]])
#     plt.ylabel(dataName[pair[1]])
#     plt.axis('tight')

#     # plot puntos de entrenamiento
#     for i, color in zip(range(n_classes), plot_colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, label=targetName[i],
#                     cmap=plt.cm.Paired)
#     plt.axis('tight')

# plt.suptitle('Ejemplos de clasificador de arboles')

# plt.legend()
# plt.show()
