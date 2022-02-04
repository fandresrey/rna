from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load



target = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                  4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, ])
targetName = np.array(['do', 're', 'mi', 'fa', 'sol', 'la', 'si'])
dataName = np.array(['FREQ'])
data = np.array([[266], [261], [263], [268], [262], [260], [267], [259], [256], [294], [298], [286], [292], [300], [296], [293], [291], [297], [331], [326], [328], [340], [327], [325], [322], [339], [336], [354], [345], [347], [
                352], [351], [341], [360], [356], [350], [385], [394], [387], [392], [383], [388], [390], [398], [395], [440], [430], [447], [446], [445], [434], [433], [432], [435], [488], [489], [499], [497], [486], [487], [500], [498], [493]])







modelo = make_pipeline(StandardScaler(), SVC(gamma='auto'))
modelo.fit(data, target)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))])

dump(modelo, 'svc.joblib')

print(modelo.predict([[400]])) 