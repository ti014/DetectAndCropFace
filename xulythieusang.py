import cv2
from matplotlib import pyplot as plt
image = cv2.imread("images/img_1.jpg")
alpha = 2 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

cv2.putText(image, "alpha: " + str(alpha), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
cv2.putText(image, "beta: " + str(beta), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

img_output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)    
plt.imshow(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
cv2.imwrite("output.jpg", img_output)