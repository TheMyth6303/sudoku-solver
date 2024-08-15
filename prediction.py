import cv2
import numpy as np
import torch.nn as nn
import torch
import model


def get_cells(puzzle):
    cells_array = []
    puzzle = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
    for i in range(9):
        for j in range(9):
            extract = puzzle[100*i+10:100*i+90, 100*j+10:100*j+90]
            thresh = cv2.threshold(extract, 150, 255, cv2.THRESH_BINARY)[1]
            resize = cv2.resize(thresh, (50, 50))
            cells_array.append(cv2.GaussianBlur(
                src=resize, ksize=(5, 5), sigmaX=0))
    cells_array = np.array(cells_array).reshape((9, 9, 50, 50))
    return cells_array


def predict(cell):
    cell = cell/255
    cell = cell.reshape(-1, 1, 50, 50)
    predictor = model.Net()
    predictor.load_state_dict(torch.load("model.pth"))

    predictor.eval()
    with torch.no_grad():
        outputs = predictor(torch.tensor(cell).to(torch.float32))
    predictions = outputs.detach().cpu().numpy()[0]
    return np.argmax(predictions)+1, max(predictions)
