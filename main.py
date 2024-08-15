import cv2
import numpy as np
from imgutils import findsudoku, draw_lines, draw_points
from prediction import *
import sudoku

if __name__ == "__main__":

    # load image
    print("-"*60)
    path = input("Enter file name: ")
    print("-"*60)
    path = "sudoku images/" + path
    image = cv2.imread(path)
    height = 600
    aspect_ratio = height / float(image.shape[0])
    width = int(aspect_ratio * image.shape[1])
    image = cv2.resize(image, (width, height))
    cv2.imshow("sudoku solver", image)
    cv2.waitKey(0)

    # preprocessing
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("sudoku solver", processed)
    cv2.waitKey(0)
    gamma = 2
    processed = np.power(processed / 255.0, gamma) * 255.0
    processed = np.uint8(processed)
    cv2.imshow("sudoku solver", processed)
    cv2.waitKey(0)
    processed = cv2.GaussianBlur(processed, (5, 5), 3)
    cv2.imshow("sudoku solver", processed)
    cv2.waitKey(0)
    processed = cv2.bilateralFilter(processed, 15, 0, 50)
    cv2.imshow("sudoku solver", processed)
    cv2.waitKey(0)
    processed = cv2.Canny(processed, 150, 225)
    cv2.imshow("sudoku solver", processed)
    cv2.waitKey(0)

    # hough transform to detect puzzle
    rho_res = 400
    theta_res = 60
    lines, corners = findsudoku(processed, rho_res, theta_res, showplots=False)
    cv2.imshow("sudoku solver", draw_lines(image, lines, rho_res, theta_res))
    cv2.waitKey(0)
    # cv2.imshow("sudoku solver", draw_points(image, corners))
    # cv2.waitKey(0)

    # transform to get puzzle
    side_length = 900
    src_points = np.array(corners, dtype=np.float32)
    dst_points = np.array([[0, 0], [side_length, 0], [side_length, side_length], [
                          0, side_length]], dtype=np.float32)
    M, _ = cv2.findHomography(src_points, dst_points)
    image = cv2.warpPerspective(image, M, (side_length, side_length))
    cv2.imshow("sudoku solver", image)
    cv2.waitKey(0)

    board = []
    # extract cells and predict
    cells = get_cells(image)
    for i in range(9):
        board.append([])
        for j in range(9):
            digit, conf = predict(cells[i][j])
            if (conf >= 0.9):
                board[i].append(digit)
            else:
                board[i].append(0)
            cv2.imshow('digit', cells[i][j])
            cv2.waitKey(0)

    puzzle = sudoku.Sudoku(3, 3, board)
    print(puzzle)
    puzzle.solve().show_full()
