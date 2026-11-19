import numpy as np
import random
import tkinter as tk

from pathlib import Path
from os import listdir
from PIL import Image, ImageDraw

BASE = Path(__file__).parent

learning_rate = 3
sampleSize = 64

trainPath = BASE / "mnist_png" / "train"
testPath = BASE / "mnist_png" / "test"
weightPath = "weight.npz"
biasPath = "bias.npz"

# --- Forward and Back ---


def wbInit():

    W1 = np.sqrt(2 / 800) * np.random.randn(784, 16)
    W2 = np.sqrt(2 / 32) * np.random.randn(16, 16)
    W3 = np.sqrt(2 / 26) * np.random.randn(16, 10)

    B1 = np.full((16), 0.01)
    B2 = np.full((16), 0.01)
    B3 = np.full((10), 0.01)

    weight = [W1, W2, W3]
    bias = [B1, B2, B3]

    np.savez(weightPath, weight[0], weight[1], weight[2])
    np.savez(biasPath, bias[0], bias[1], bias[2])

    return weight, bias


with np.load(weightPath) as data:
    W1 = data["arr_0"]
    W2 = data["arr_1"]
    W3 = data["arr_2"]
weight = [W1, W2, W3]

with np.load(biasPath) as data:
    B1 = data["arr_0"]
    B2 = data["arr_1"]
    B3 = data["arr_2"]
bias = [B1, B2, B3]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forwardPass(a):

    A1 = a
    A2 = np.zeros(16)
    A3 = np.zeros(16)
    A4 = np.zeros(10)

    Z2 = np.zeros(16)
    Z3 = np.zeros(16)
    Z4 = np.zeros(10)

    layer = [A1, A2, A3, A4]
    weightIn = [Z2, Z3, Z4]

    for l in range(len(layer) - 1):
        weightIn[l] = (layer[l] @ weight[l]) + bias[l]
        layer[l + 1] = sigmoid(weightIn[l])

    return layer, weightIn


def backProp(layer, weightIn, y):

    dZ2 = np.zeros(16)
    dZ3 = np.zeros(16)
    dZ4 = np.zeros(10)
    dZ = [dZ2, dZ3, dZ4]

    dA = np.zeros(10)

    dW2 = np.zeros((784, 16))
    dW3 = np.zeros((16, 16))
    dW4 = np.zeros((16, 10))
    dW = [dW2, dW3, dW4]

    dB2 = np.zeros(16)
    dB3 = np.zeros(16)
    dB4 = np.zeros(10)
    dB = [dB2, dB3, dB4]

    for l in range(len(layer) - 1, 0, -1):
        if l <= (len(layer) - 2):
            dZ[l - 1] = np.multiply((weight[l] @ dZ[l]), sigmoid(weightIn[l - 1])) * (
                1 - sigmoid(weightIn[l - 1])
            )
            dW[l - 1] = np.outer(layer[l - 1], dZ[l - 1])
            dB[l - 1] = dZ[l - 1]

        else:
            dA = layer[l] - y
            dZ[l - 1] = np.multiply(dA, sigmoid(weightIn[l - 1])) * (
                1 - sigmoid(weightIn[l - 1])
            )
            dW[l - 1] = np.outer(layer[l - 1], dZ[l - 1])
            dB[l - 1] = dZ[l - 1]

    Cx = 0.5 * ((y - layer[len(layer) - 1]) ** 2)

    return dW, dB, Cx


def loadImages(path):

    index = []

    for y in range(10):
        imagesList = listdir(path + "/" + str(y))
        yList = np.zeros(10)
        yList[y] = 1

        for image in imagesList:
            img = Image.open(path + "/" + str(y) + "/" + image)
            aValues = np.array(img, dtype=np.float32).flatten() / 255
            index.append((aValues, yList))

    return index


# --- Training and Testing ---


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def train(e):

    index = loadImages(trainPath)

    for ii in range(e):
        random.shuffle(index)

        for b in chunks(index, sampleSize):
            W1 = np.zeros((784, 16))
            W2 = np.zeros((16, 16))
            W3 = np.zeros((16, 10))

            B1 = np.zeros((16))
            B2 = np.zeros((16))
            B3 = np.zeros((10))

            cost = 0

            totalW = [W1, W2, W3]
            totalB = [B1, B2, B3]

            for s in b:
                dW, dB, Cx = backProp(*forwardPass(s[0]), s[1])

                for i in range(len(dW)):
                    totalW[i] += dW[i]

                for i in range(len(dB)):
                    totalB[i] += dB[i]

                cost += sum(Cx)

            for i in range(len(weight)):
                weight[i] = weight[i] - totalW[i] / sampleSize * learning_rate

            for i in range(len(bias)):
                bias[i] = bias[i] - totalB[i] / sampleSize * learning_rate

    np.savez(weightPath, weight[0], weight[1], weight[2])
    np.savez(biasPath, bias[0], bias[1], bias[2])


def test():

    index = loadImages(testPath)
    total = 0

    for i in index:
        layer, weightIn = forwardPass(i[0])
        if np.argmax(layer[len(layer) - 1]) == np.argmax(i[1]):
            total += 1

    accuracy = total / (len(index))
    incorrect = len(index) - total
    print("Accuracy --- " + str(accuracy))

    return accuracy, total, incorrect


def testInd(a):

    layer, weightIn = forwardPass(a)
    guess = np.argmax(layer[len(layer) - 1])
    print(str(guess))
    certainty = round(layer[len(layer) - 1][np.argmax(layer[len(layer) - 1])], 3)

    return guess, certainty


# Tkinter

root = tk.Tk()
root.title("Neural Network")

w, h = 280, 280

line_id = None
line_points = []
line_path = []
line_options = {
    "capstyle": tk.ROUND,
    "fill": "white",
    "joinstyle": tk.ROUND,
    "width": 20,
}

printLabel = tk.Label(root)


def draw_line(event):
    global line_id
    line_points.extend((event.x, event.y))
    if line_id is not None:
        canvas.delete(line_id)
    line_id = canvas.create_line(line_points, **line_options)


def set_start(event):
    line_points.extend((event.x, event.y))


def end_line(event=None):
    global line_id
    line_path.append(line_points.copy())
    line_points.clear()
    line_id = None


def clear_line():
    global line_id
    line_path.clear()
    line_points.clear()
    line_id = None
    canvas.delete("all")


def submit_img():
    image = Image.new("L", (w, h), 0)
    img = ImageDraw.Draw(image)
    for p in line_path:
        img.line(p, fill="white", width=15, joint="curve")
    cropImg = image.crop(image.getbbox())
    print(str(cropImg.size))

    if cropImg.width >= cropImg.height:
        x = 20
        y = int(20 * (cropImg.height / cropImg.width))
    else:
        x = int(20 * (cropImg.width / cropImg.height))
        y = 20

    numImg = cropImg.resize([x, y])
    finalImg = Image.new("L", (28, 28), 0)
    finalImg.paste(numImg, (((28 - x) // 2), ((28 - y) // 2)))
    a = np.array(finalImg, dtype=np.float32).flatten() / 255
    guess, certainty = testInd(a)

    printLabel.config(text=f"Guess: {guess}\nCertainty: {certainty:%}")


def testLabel():
    accuracy, correct, incorrect = test()
    printLabel.config(
        text=f"Accuracy: {accuracy:%}\nCorrect: {correct}\n Incorrect: {incorrect}"
    )


def wbReset():
    global weight
    global bias
    weight, bias = wbInit()


clearButton = tk.Button(root, text="CLEAR", command=clear_line)
submitButton = tk.Button(root, text="SUBMIT", command=submit_img)
trainButton = tk.Button(root, text="TRAIN", command=lambda: train(30))
testButton = tk.Button(root, text="TEST", command=testLabel)
resetButton = tk.Button(root, text="RESET", command=wbReset)
canvas = tk.Canvas(root, width=w, height=h, bg="black")

clearButton.grid(row=0, column=0)
submitButton.grid(row=0, column=1)
trainButton.grid(row=0, column=2)
testButton.grid(row=0, column=3)
resetButton.grid(row=0, column=4)
printLabel.grid(row=2, column=0, columnspan=5)
canvas.grid(row=1, column=0, columnspan=5)

canvas.bind("<Button-1>", set_start)
canvas.bind("<B1-Motion>", draw_line)
canvas.bind("<ButtonRelease-1>", end_line)

root.mainloop()
