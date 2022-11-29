# # =[Modules dan Packages]========================

# from flask import Flask,render_template,request,jsonify
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from joblib import load

# # =[Variabel Global]=============================

# app   = Flask(__name__, static_url_path='/static')
# model = None

# # =[Routing]=====================================

# # [Routing untuk Halaman Utama atau Home]	
# @app.route("/")
# def beranda():
#     return render_template('index.html')

# # [Routing untuk API]	
# @app.route("/api/deteksi",methods=['POST'])
# def apiDeteksi():
# 	# Nilai default untuk variabel input atau features (X) ke model
# 	input_sepal_length = 5.1
# 	input_sepal_width  = 3.5
# 	input_petal_length = 1.4
# 	input_petal_width  = 0.2
	
# 	if request.method=='POST':
# 		# Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
# 		input_sepal_length = float(request.form['sepal_length'])
# 		input_sepal_width  = float(request.form['sepal_width'])
# 		input_petal_length = float(request.form['petal_length'])
# 		input_petal_width  = float(request.form['petal_width'])
		
# 		# Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
# 		df_test = pd.DataFrame(data={
# 			"SepalLengthCm" : [input_sepal_length],
# 			"SepalWidthCm"  : [input_sepal_width],
# 			"PetalLengthCm" : [input_petal_length],
# 			"PetalWidthCm"  : [input_petal_width]
# 		})

# 		hasil_prediksi = model.predict(df_test[0:1])[0]

# 		# Set Path untuk gambar hasil prediksi
# 		if hasil_prediksi == 'Iris-setosa':
# 			gambar_prediksi = '/static/images/iris_setosa.jpg'
# 		elif hasil_prediksi == 'Iris-versicolor':
# 			gambar_prediksi = '/static/images/iris_versicolor.jpg'
# 		else:
# 			gambar_prediksi = '/static/images/iris_virginica.jpg'
		
# 		# Return hasil prediksi dengan format JSON
# 		return jsonify({
# 			"prediksi": hasil_prediksi,
# 			"gambar_prediksi" : gambar_prediksi
# 		})

# # =[Main]========================================

# if __name__ == '__main__':
	
# 	# Load model yang telah ditraining
# 	model = load('model_iris_dt.model')

# 	# Run Flask di localhost 
# 	app.run(host="localhost", port=5000, debug=True)
	
	
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__, static_url_path='/static/css/style.css')


@app.route("/", methods=['GET', 'POST'])
def diabetes_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        print(dict(request.form))
        diabetes_features = dict(request.form).values()
        diabetes_features = np.array([float(x) for x in diabetes_features])
        print(diabetes_features)
        module_dir = os.path.dirname(__file__)
        f = os.path.join(module_dir, 'diabetes-classification-using-logistic-regression.pkl')
        model, std_scaler = joblib.load(f)
        diabetes_features = std_scaler.transform([diabetes_features])
        print(diabetes_features)
        result = model.predict(diabetes_features)
        diabetes = {
            '0': 'Not Having Diabetes',
            '1': 'Having Diabetes'
        }
        result = diabetes[str(result[0])]
        return render_template('index.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)

