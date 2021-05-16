import cv2
import numpy as np
import random

def AddGaussianNoise(img, k_size, sig):
    blurred_img = cv2.GaussianBlur(img, (k_size, k_size), sigma)
    return blurred_img

def AddSaltPepperNoise(img, prob):
    noise_img = np.zeros(img.shape, np.uint8)
    thresh = 1 - prob
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_num = random.random()
            
            if random_num < prob:
                noise_img[i][j] = 0
            elif random_num > thresh:
                noise_img[i][j] = 255
            else:
                noise_img[i][j] = img[i][j]
    
    return noise_img

def AddPoissonNoise(img, lam):
    noise_matrix = np.random.poisson(lam, img.shape)
    output = img + noise_matrix
    return output

cap = cv2.VideoCapture("Video/bot_run.mp4")

kernel_size = int(input("Enter the kernel size for Gaussian Noise : "))
sigma = int(input("Enter the sigma value for Gaussian Noise : "))

probability = float(input("Enter the probabilty factor for Salt and Pepper Noise : "))

lambda_poisson = int(input("Enter the Lambda for Poisson distribution : "))

while True:
    ret, frame = cap.read()
    
    cv2.imshow("Original Video", frame)
    
    blur = AddGaussianNoise(frame, kernel_size, sigma)
    cv2.imshow("Video after Gaussian Blur", blur)
    
    salt_pepper_img = AddSaltPepperNoise(frame, probability)
    cv2.imshow("Video after adding Salt & Pepper Noise", salt_pepper_img)
    
    poisson_noise_img = AddPoissonNoise(frame, lambda_poisson)
    cv2.imshow("Video after Poisson Noise", poisson_noise_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
