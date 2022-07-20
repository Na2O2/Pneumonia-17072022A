from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from keras.models import load_model
import cv2
from PIL import Image #use PIL
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        file.save("./static/"+filename)#Heroku no need static
        file = open("./static/"+filename, "r")#Heroku no need static
        #filename = r'C:\Users\ASUS\Documents\大学文档\作业\大三上\南洋理工\Week3\Pneumonia\chest_xray\val\NORMAL\NORMAL2-IM-1427-0001.jpeg'
        model = load_model("Pneumonia")
        #image = cv2.imread(filename)
        image = cv2.imdecode(np.fromfile("./static/"+filename, dtype=np.uint8), 0)#For non-ASCII characters
        #print(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.merge([gray, gray, gray])
        #print(type(img))
        img.resize((150, 150, 3))
        #print(img)
        img = np.asarray(img, dtype="float32")#need to transfer to np to reshape
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img.shape
        pred = model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))

if __name__ == "__main__":
    app.run()