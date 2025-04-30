import pygame
import math
import cv2
import numpy as np
import time
import random

# Initialize Pygame
pygame.init()
pygame.mixer.init()
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# Load and scale images
track_mask = pygame.image.load('data/track.bmp').convert()
color_track = pygame.image.load('data/color_track.png').convert()
car_img = pygame.image.load('data/car.png').convert_alpha()

track_mask = pygame.transform.scale(track_mask, (screen_width, screen_height))
color_track = pygame.transform.scale(color_track, (screen_width, screen_height))
car_img = pygame.transform.scale(car_img, (30, 60))  # Adjust car size

# Load sounds
crash_sound = pygame.mixer.Sound('data/crash.mp3')
finish_sound = pygame.mixer.Sound('data/finish.mp3')

# Car state
start_pos = [int(screen_width * 0.9), int(screen_height * 0.9)]
car_pos = start_pos.copy()
car_angle = 0
car_speed = 0
max_speed = 3
score = 0

# Hand tracking globals
orientation = 90
cY = 300

# Lap timing
lap_start_time = time.time()
best_lap_time = None
current_lap_time = 0
new_record_active = False
new_record_timer = 0

# Countdown after scoring
countdown_active = False
countdown_start_time = 0

# Particles list
particles = []

# Define finish line zone
finish_rect = pygame.Rect(int(screen_width * 0.375), int(screen_height * 0.88), int(screen_width * 0.14), int(screen_height * 0.12))  # Adjust as needed

# Capture camera
cap = cv2.VideoCapture('http://192.168.1.2:8080/video')

def get_hand_control():
    global orientation, cY
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
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

            # Draw ellipse on frame
            cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return frame

def rotate_center(image, angle):
    rotated_image = pygame.transform.rotate(image, -angle)
    rect = rotated_image.get_rect(center=image.get_rect(topleft=(0, 0)).center)
    return rotated_image, rect

def spawn_particles(x, y):
    for _ in range(50):
        particles.append([
            x, y,
            random.uniform(-3, 3), random.uniform(-3, 3),
            random.randint(20, 40)
        ])

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Update hand control
    frame = get_hand_control()

    if countdown_active:
        elapsed = time.time() - countdown_start_time
        if elapsed < 1:
            countdown_text = "3"
        elif elapsed < 2:
            countdown_text = "2"
        elif elapsed < 3:
            countdown_text = "1"
        elif elapsed < 4:
            countdown_text = "GO!"
        else:
            countdown_active = False
            lap_start_time = time.time()  # Restart lap timer

        countdown_surface = font.render(countdown_text, True, (255, 255, 255))
        rect = countdown_surface.get_rect(center=(screen_width // 2, screen_height // 2))
        screen.blit(countdown_surface, rect)

    else:
        # Control car
        steer = np.interp(orientation, [0, 180], [45, -45])
        speed = np.interp(cY, [screen_height, 0], [0, max_speed])

        car_angle += steer * 0.05
        rad = math.radians(car_angle - 90)
        new_x = car_pos[0] + speed * math.cos(rad)
        new_y = car_pos[1] + speed * math.sin(rad)

        # Check bounds and track
        if 0 <= int(new_x) < screen_width and 0 <= int(new_y) < screen_height:
            pixel = track_mask.get_at((int(new_x), int(new_y)))
            if pixel.r == 0 and pixel.g == 0 and pixel.b == 0:  # Black pixel (valid track)
                car_pos = [new_x, new_y]
            else:
                crash_sound.play()
                car_pos = start_pos.copy()
                car_angle = 0
                spawn_particles(car_pos[0], car_pos[1])
        else:
            crash_sound.play()
            car_pos = start_pos.copy()
            car_angle = 0
            spawn_particles(car_pos[0], car_pos[1])

        # Check if car crossed the finish line
        if finish_rect.collidepoint(car_pos[0], car_pos[1]):
            score += 10
            spawn_particles(car_pos[0], car_pos[1])
            finish_sound.play()

            # Calculate lap time
            lap_time = time.time() - lap_start_time
            if best_lap_time is None or lap_time < best_lap_time:
                best_lap_time = lap_time
                new_record_active = True
                new_record_timer = pygame.time.get_ticks()
            current_lap_time = lap_time

            # Restart car
            car_pos = start_pos.copy()
            car_angle = 0

            # Start countdown
            countdown_active = True
            countdown_start_time = time.time()

    # Draw background
    screen.blit(color_track, (0, 0))

    # Draw car
    rotated_car, rect = rotate_center(car_img, car_angle)
    rect.center = car_pos
    screen.blit(rotated_car, rect.topleft)

    # Draw Finish Line
    pygame.draw.rect(screen, (0, 255, 0), finish_rect, 3)

    # Draw information
    info_text = font.render(f"Orientation: {orientation:.1f}°", True, (255, 255, 255))
    speed_text = font.render(f"Speed: {max_speed:.1f}", True, (255, 255, 255))
    score_text = font.render(f"Score: {score}", True, (255, 255, 0))
    lap_text = font.render(f"Lap: {current_lap_time:.2f}s", True, (200, 200, 255))
    best_text = font.render(f"Best: {best_lap_time:.2f}s" if best_lap_time else "Best: --", True, (255, 200, 0))

    screen.blit(info_text, (20, 20))
    screen.blit(speed_text, (20, 70))
    screen.blit(score_text, (20, 120))
    screen.blit(lap_text, (20, 170))
    screen.blit(best_text, (20, 220))

    # Draw New Record flashing
    if new_record_active:
        if pygame.time.get_ticks() - new_record_timer < 2000:
            if (pygame.time.get_ticks() // 300) % 2 == 0:
                record_text = font.render("NEW RECORD!", True, (255, 215, 0))
                record_rect = record_text.get_rect(center=(screen_width // 2, screen_height // 2 - 100))
                screen.blit(record_text, record_rect)
        else:
            new_record_active = False

    # Draw particles
    for p in particles[:]:
        p[0] += p[2]
        p[1] += p[3]
        p[4] -= 1
        pygame.draw.circle(screen, (255, 255, 0), (int(p[0]), int(p[1])), 3)
        if p[4] <= 0:
            particles.remove(p)

    # Show camera view
    if frame is not None:
        frame = cv2.resize(frame, (int(screen_width / 4), int(screen_height / 4)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        screen.blit(frame_surface, (screen_width - frame_surface.get_width(), 0))

    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
