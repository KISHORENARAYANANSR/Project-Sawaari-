from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    image = Image.open(file.stream)
    crop_coordinates = [
    (246,993,732,1448),
    (699,930,1155,1430),
    (1106,886,1562,1430),
    (2030,759,2376,1364),
    (2382,732,2750,1281),
    (699,655,974,864),
    (985,666,1254,825),
    (1249,622,1529,820),
    (1881,556,2090,726),
    (2118,512,2332,699),
    (908,523,1101,605),
    (1101,506,1315,605),
    (1326,490,1518,589),
    (1793,440,1969,523),
    (1975,413,2151,506)
    ]
    images = []
    # Crop the image and save the results
    for i, (x1, y1, x2, y2) in enumerate(crop_coordinates):
        cropped_image = image.crop((x1, y1, x2, y2))
        images.append(cropped_image)
        #cropped_image.save(f'cropped_image_{i+1}.jpg')  # Save each cropped image
    
    print("Cropped images saved successfully.")
    
    
    
    model_path = "keras_model.h5"
    
    from keras.models import load_model  # TensorFlow is required for Keras to work
    from PIL import ImageOps  # Install pillow instead of PIL
    import numpy as np
    
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    try:
        model = load_model(model_path, compile=False)
    except ValueError as e:
        if "DepthwiseConv2D" in str(e) and "'groups': 1" in str(e):
            print("Error: The model contains unsupported arguments in DepthwiseConv2D.")
            print("Please ensure the model is compatible with the current Keras version.")
            raise
    
    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    #class_names = open("labels.txt", "r").readlines()
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image_path = model_path
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    for i in range(15):
      images[i] = images[i].convert("RGB")
      images[i] = ImageOps.fit(images[i], size, Image.Resampling.LANCZOS)
      images[i] = np.asarray(images[i])
      images[i] = (images[i].astype(np.float32) / 127.5) - 1
    #image = Image.open(image_path).convert("RGB")
    
    
    # Turn the image into a numpy array
    
    # Normalize the image
    
    boolean = []
    # Load the image into the array
    for i in range(15):
     data[0] = images[i]
    # Predicts the model
     try:
         prediction = model.predict(data)
         index = np.argmax(prediction)
         #print(index)
         class_name = class_names[index]
         confidence_score = prediction[0][index]
         #print(class_name[2:])
         
         if confidence_score > 0.7:
            boolean.append(int(index))
        
     
         # Print prediction and confidence score
         print("Class:", class_name[2:], end="")
         print("Confidence Score:", confidence_score)
     except Exception as e:
         print("Prediction failed:", str(e))
    print(boolean) 

    return jsonify(boolean), 200
#public_url = ngrok.connect(port='5000')
#print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
    

if __name__ == '__main__':
    #public_url = ngrok.connect(port='5000')
    #print('Public URL:', public_url)
    app.run()