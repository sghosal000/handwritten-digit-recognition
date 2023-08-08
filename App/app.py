from tokenize import Number
# from numpy import testing
# from numpy.lib.type_check import imag
import pygame, sys
from pygame.locals import *
import numpy as np
import pickle
import cv2

BOUNDRYINC = 5
WINDOWSIZEX = 1280
WINDOWSIZEY = 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

IMAGESAVE = False
img_count = 1

DR_model = pickle.load(open("Digit_Recognition_model_RFv2.sav", 'rb'))

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three",
          4: "Four", 5: "Five", 6: "Six",
          7: "Seven", 8: "Eight", 9: "Nine"}


# initialize pygame
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Writing Board")

iswriting = False

PREDICT = True

xcord_num = []
ycord_num = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            xcord_num.append(xcord)
            ycord_num.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            xcord_num = sorted(xcord_num)
            ycord_num = sorted(ycord_num)

            rect_min_x, rect_max_x = max(xcord_num[0]-BOUNDRYINC, 0), min(WINDOWSIZEX, xcord_num[-1]+BOUNDRYINC)
            rect_min_y, rect_max_y = max(ycord_num[0]-BOUNDRYINC, 0), min(WINDOWSIZEY, ycord_num[-1]+BOUNDRYINC)

            xcord_num = []
            ycord_num = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x: rect_max_x, rect_min_y: rect_max_y].T.astype(np.float32)
            print(img_arr.shape)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_count += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), "constant", constant_values=0)
                image = cv2.resize(image, (28, 28))/255
                image_in255 = (((image - image.min())/(image.max() - image.min()))*255).astype(np.int64)

                # label = str(LABELS[np.argmax(DR_model.predict(image_shaped.reshape(1, 784)))])
                label = str(LABELS[DR_model.predict(image_in255.reshape(1, 784))[0]])

                textSurface = FONT.render(label, True, GREEN, BLACK)
                # textRecObj = Number.get_rect()
                # textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y
                DISPLAYSURF.blit(textSurface, (rect_min_x, rect_max_y))

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

        pygame.display.update()
