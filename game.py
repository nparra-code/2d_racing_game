import pygame
import math
import cv2
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Load car image
car_img = pygame.image.load("car.png")  # Make sure you have a car.png image
car_img = pygame.transform.scale(car_img, (60, 30))

# Car state
car_pos = [400, 300]
car_angle = 0
car_speed = 0
max_speed = 2

# Hand tracking globals
orientation = 90  # Default angle
cY = 300  # Default hand position

# Function to rotate car
def rotate_center(image, angle):
    rotated_image = pygame.transform.rotate(image, -angle)
    rect = rotated_image.get_rect(center=image.get_rect(topleft=(0, 0)).center)
    return rotated_image, rect

# Capture from IP/webcam
cap = cv2.VideoCapture('http://192.168.134.12:8080/video')

def get_hand_control():
    global orientation, cY
    ret, frame = cap.read()
    if not ret:
        return
    
    

    frame = cv2.flip(frame, 1)  # 🔁 Mirror the frame horizontally

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hand = max(contours, key=cv2.contourArea)
        M = cv2.moments(hand)
        if M["m00"] != 0:
            cY = int(M["m01"] / M["m00"])

        if len(hand) >= 5:
            ellipse = cv2.fitEllipse(hand)
            orientation = ellipse[2]

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    get_hand_control()

    # Map hand orientation (0-180) to -45 to +45 degrees
    steer = np.interp(orientation, [0, 180], [45, -45])

    # Map cY (0-480) to 0-max_speed (invert: top = fast)
    speed = np.interp(cY, [480, 0], [0, max_speed])

    # Update car angle and position
    car_angle += steer * 0.05
    rad = math.radians(car_angle)
    car_pos[0] += speed * math.cos(rad)
    car_pos[1] += speed * math.sin(rad)

    # Clamp car position within window bounds
    car_pos[0] = max(0, min(800, car_pos[0]))
    car_pos[1] = max(0, min(600, car_pos[1]))

    # Draw car
    rotated_car, rect = rotate_center(car_img, car_angle)
    rect.center = car_pos
    screen.blit(rotated_car, rect.topleft)

    # Display info
    text = font.render(f"Orientation: {orientation:.1f}, cY: {cY}, Speed: {speed:.2f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
