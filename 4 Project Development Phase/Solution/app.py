from flask import Flask, render_template, request
from keras.models import load_model
import keras.utils as image
import numpy as np

app = Flask(__name__)

cats = ["AFRICAN LEOPARD", "CARACAL", "CHEETAH", "CLOUDED LEOPARD", "JAGUAR", "LION", "OCELOT", "PUMA", "SNOW LEOPARD", "TIGER"]
model = load_model('model.h5')

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1,224,224,3)
	p = model.predict(i)
	l = np.argmax(p, axis=1)
	return cats[l[0]]

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		p = predict_label(img_path)
	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	app.run(debug = True)
	