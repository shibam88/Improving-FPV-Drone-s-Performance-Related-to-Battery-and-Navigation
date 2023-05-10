from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import sklearn as sk

app = Flask(__name__)

model=pickle.load(open('model.sav','rb'))


@app.route('/')
def hello_world():
    return render_template("forest.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    sc=pickle.load(open('scaler.pkl','rb'))
    final=sc.transform(np.array(final))
    prediction=model.predict(final)
    
    output=f'{prediction}'

    return render_template('forest.html',pred='prediction for your class is  {}'.format(output),bhai="kuch karna hain iska ab?")
    

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')

