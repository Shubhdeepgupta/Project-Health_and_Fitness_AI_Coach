# import cv2
# import mediapipe as mp
# import time

# class poseDetector():
#     def __init__(self):
#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose()

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
#         return img

# def main():
#     cap = cv2.VideoCapture('dada/AI_trainer/curls3.mp4')
#     pTime = 0
#     detector = poseDetector()
#     frame_count = 0
#     max_frames = 100  # You can adjust this value based on your requirement

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         img = detector.findPose(img)

#         cTimeDisplay = time.time()
#         fpsDisplay = 1 / (cTimeDisplay - pTime)
#         pTime = cTimeDisplay

#         cv2.putText(img, f'Display FPS: {int(fpsDisplay)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

#         cv2.imshow("Image", img)
#         cv2.waitKey(1)  # Add a slight delay

#         frame_count += 1

#         # Add break condition to exit the loop after processing max_frames
#         if frame_count >= max_frames:
#             break

#     # Release the video capture object
#     cap.release()

#     # Close all OpenCV windows
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture('dada/AI_trainer/curls3.mp4')
    pTime = 0
    detector = poseDetector()
    frame_count = 0
    max_frames = 100  # You can adjust this value based on your requirement

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)

        cTimeDisplay = time.time()
        fpsDisplay = 1 / (cTimeDisplay - pTime)
        pTime = cTimeDisplay

        cv2.putText(img, f'Display FPS: {int(fpsDisplay)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)  # Add a slight delay

        frame_count += 1

        # Add break condition to exit the loop after processing max_frames
        if frame_count >= max_frames:
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
