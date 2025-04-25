import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsv)

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    # Binary image: skin range is 255, others are 0
    binary_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    cv2.imshow('Binary Skin Only', binary_skin)

    # Optional: Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(binary_skin, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)

        moments = cv2.moments(hand_contour)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Centroid: ({cX}, {cY})", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(hand_contour) >= 5:
            ellipse = cv2.fitEllipse(hand_contour)
            cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
            angle = ellipse[2]
            cv2.putText(frame, f"Orientation: {angle:.2f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Ellipse mask
            ellipse_mask = np.zeros_like(clean_mask)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)

            # Masked skin area inside the ellipse
            masked_region = cv2.bitwise_and(clean_mask, ellipse_mask)

            # Edge detection only in that area
            ellipse_edges = cv2.Canny(masked_region, 100, 200)
            cv2.imshow('Ellipse Border Only', ellipse_edges)

    return frame, hsv

# Replace with your camera IP or use 0 for local webcam
cap = cv2.VideoCapture('http://192.168.134.12:8080/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, hsv = process_hand(frame)
    cv2.imshow('Hand Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
