
import pickle

loadedmodel = pickle.load(open('file.pkl', 'rb'))

custom_input_DE = 0.064254
custom_input_FE = 0.038625
pred = loadedmodel.predict([[custom_input_DE, custom_input_FE]])
print(pred)