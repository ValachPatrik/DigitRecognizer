import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import numpy as np
from keras.models import load_model
import io
import skimage.io
import skimage.transform
from skimage.color import rgb2gray

# Load the trained model
model = load_model('digits.h5')

# Create a blank canvas for drawing
canvas_width = 200
canvas_height = 200

def clear_canvas():
    canvas.delete("all")

def predict_number():
    image = ImageGrab.grab(bbox=(0, 0, 200, 200))  # Adjust the coordinates as needed

    image_gray = image.convert('L')
    image_gray = image_gray.resize((28, 28))

    # Convert the resized image to a numpy array
    image_array = np.array(image_gray)
    image_array = image_array.reshape(1, 28, 28, 1)
    image_array = image_array.astype('float32') / 255.0

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_number = np.argmax(prediction)

    # Display the predicted number
    result_label.config(text=f"Predicted Number: {predicted_number}")

# Create the GUI
root = tk.Tk()
root.title("Number Recognition")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

result_label = tk.Label(root, text="Draw a number", font=("Arial", 10))
result_label.pack(pady=10)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=10)

predict_button = tk.Button(root, text="Predict", command=predict_number)
predict_button.pack(side=tk.RIGHT, padx=10)

# Implement drawing functionality
def start_drawing(event):
    canvas.create_oval(event.x, event.y, event.x, event.y, fill='black')

def draw(event):
    canvas.create_oval(event.x, event.y, event.x, event.y, fill='black')

canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)

root.mainloop()
