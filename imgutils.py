import numpy as np
import math
import matplotlib.pyplot as plt


def _draw_line(image, rho, theta):
    h, w, depth = image.shape
    if (135 >= theta >= 45):
        for j in range(w):
            i = round((rho + j*math.cos(math.radians(theta))) /
                      math.sin(math.radians(theta)))
            if (0 <= i < h):
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 255
    else:
        for i in range(h):
            j = round((-rho + i*math.sin(math.radians(theta))) /
                      math.cos(math.radians(theta)))
            if (0 <= j < w):
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 255
    return image


def draw_lines(image, indices, rho_res, theta_res=60):
    image = image.copy()
    h, w, depth = image.shape
    d = math.sqrt(h*h + w*w)
    if rho_res == -1:
        rho_res = int(d/1.5)
    for indice in indices:
        rho = (indice[0]+0.5-rho_res/2)*2*d/rho_res
        theta = indice[1]*180/theta_res
        image = _draw_line(image, rho, theta)
    return image


def draw_points(image, points):
    image = image.copy()
    for y, x in points:
        x = round(x)
        y = round(y)
        for i in range(-2, 3):
            image[x+i][y+i][0] = 0
            image[x+i][y+i][1] = 255
            image[x+i][y+i][2] = 0
            image[x+i][y-i][0] = 0
            image[x+i][y-i][1] = 255
            image[x+i][y-i][2] = 0
    return image


def _get_local_max(arr, window):
    local_maximas = []
    for i in range(0, arr.shape[0]):
        valid = True
        for m in range(-window, window+1):
            if m == 0:
                continue
            temp = arr[(i+m) % arr.shape[0]]
            if temp > arr[i]:
                valid = False
                break
        if valid:
            local_maximas.append(i)
    return local_maximas


def _show_accumulator_matrix(acc_mat):
    fig, ax = plt.subplots()
    img = ax.imshow(acc_mat, cmap='viridis')
    ax.set_aspect(acc_mat.shape[1]/acc_mat.shape[0],
                  adjustable='box', anchor='C')
    plt.xlabel("theta")
    plt.ylabel("rho")
    plt.title("accumulator matrix")
    plt.show()


def _compute_accumulator_matrix(image, h, w, rho_res, theta_res, show=False):
    d = math.sqrt(h*h+w*w)
    acc_mat = np.zeros((rho_res, theta_res))
    thetas = np.linspace(0, 180-180/theta_res, theta_res)
    for i in range(0, h):
        for j in range(0, w):
            if image[i][j] == 255:
                for theta in thetas:
                    rho = i*math.sin(math.radians(theta)) - \
                        j*math.cos(math.radians(theta))
                    rho = rho/d
                    rho = rho*rho_res/2
                    rho = int(rho + rho_res/2)
                    acc_mat[rho][int(theta*theta_res/180)] += 1
    if show:
        _show_accumulator_matrix(acc_mat)
    return acc_mat


def _find_thetas(acc_mat, theta_res, show=False):
    std = np.std(acc_mat, axis=0)
    if show:
        plt.plot(std)
        plt.xlabel("theta")
        plt.ylabel("std")
        plt.title("standard deviation in accumulator matrix vs theta")
        plt.show()
    thetas = _get_local_max(std, theta_res//15)
    theta1 = None
    theta2 = None
    for theta in thetas:
        if 10 < theta*180/theta_res < 80 or 100 < theta*180/theta_res < 170:
            continue
        elif 10 >= theta*180/theta_res or theta*180/theta_res >= 170:
            theta1 = theta
        else:
            theta2 = theta
    return theta1, theta2


def _find_rhos(acc_mat_col, rho_res):
    maxVal = 0
    bestrho1 = 0
    bestrho2 = 0
    # assuming sudoku size length is atleast a fifth of diagonal length
    for rho1 in range(int(4*rho_res/5)):
        for rho2 in range(rho1 + int(rho_res/5), rho_res):
            rhos = []
            val = 0
            for n in range(0, 10):
                bestPeakVal = -1
                bestPeakRho = -1
                for i in range(-3, 3):
                    currRho = round(rho1 + (rho2-rho1)*n/9 + i)
                    currVal = acc_mat_col[currRho %
                                          rho_res]/((abs(i)+1)*(abs(i)+1))
                    if (currVal > bestPeakVal):
                        bestPeakVal = currVal
                        bestPeakRho = currRho
                val += bestPeakVal
                rhos.append(bestPeakRho)
            if (val > maxVal):
                maxVal = val
                bestrho1 = rho1
                bestrho2 = rho2
    return bestrho1, bestrho2


def _find_lines(acc_mat, thetas, show):
    if show:
        _, axs = plt.subplots(1, 2, figsize=(10, 4))
        for i in range(2):
            axs[i].plot(np.arange(acc_mat.shape[0]),
                        acc_mat[:, thetas[i]], color='blue')
            axs[i].set_title('theta='+str(thetas[i]*180/acc_mat.shape[1]))
            axs[i].set_xlabel('rhos')
            axs[i].set_ylabel('acccumulator matrix')
        plt.tight_layout()
        plt.show()
    lines = []
    for i in range(2):
        rhos = _find_rhos(acc_mat[:, thetas[i]], acc_mat.shape[0])
        for rho in rhos:
            lines.append((rho, thetas[i]))
    return lines


def _get_corners(lines, rho_res, theta_res, d):
    corners = []
    for rho1_index in range(2):
        for rho2_index in range(2):
            r1, theta1 = lines[rho1_index]
            r2, theta2 = lines[2+rho2_index]
            theta1 = theta1*180/theta_res
            theta2 = theta2*180/theta_res
            r1 = (r1+0.5-rho_res/2)*2*d/rho_res
            r2 = (r2+0.5-rho_res/2)*2*d/rho_res
            y = r1*math.cos(math.radians(theta2)) - r2 * \
                math.cos(math.radians(theta1))
            y = y/math.sin(math.radians(theta1 - theta2))
            x = r1*math.sin(math.radians(theta2)) - r2 * \
                math.sin(math.radians(theta1))
            x = x/math.sin(math.radians(theta1 - theta2))
            corners.append((x, y))
    # sorted_corners = []
    corners = sorted(corners, key=lambda x: x[1])
    top_corners = sorted(corners[:2], key=lambda x: x[0])
    bottom_corners = sorted(corners[2:], key=lambda x: -x[0])
    corners = top_corners + bottom_corners
    return corners


def findsudoku(image, rho_res, theta_res, showplots=False):
    h, w = image.shape
    accumulator_matrix = _compute_accumulator_matrix(
        image, h, w, rho_res, theta_res)
    d = math.sqrt(h*h+w*w)
    if showplots:
        _show_accumulator_matrix(accumulator_matrix)
    thetas = _find_thetas(accumulator_matrix, theta_res, showplots)
    lines = _find_lines(accumulator_matrix, thetas, showplots)
    corners = _get_corners(lines, rho_res, theta_res, d)
    return lines, corners
