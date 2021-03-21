import mediapipe as mp
import numpy as np
import argparse
import glob
import time
import tqdm
import cv2
import os


class Holistic():

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(230, 224, 55), thickness=1, circle_radius=1)
        self.pose_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(230, 224, 55), thickness=2, circle_radius=2)
        self.face_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(230, 224, 55), thickness=1, circle_radius=1)
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False,
                                                  upper_body_only=False,
                                                  smooth_landmarks=True,
                                                  min_detection_confidence=0.4,
                                                  min_tracking_confidence=0.2)

    def rescale_frame(self, frame, percent=75):
        width = int((frame.shape[1] * percent / 100))
        height = int(frame.shape[0] * percent / 100)
        width = int(height*(3/4))
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def draw_landmark(self, img, results):

        annotated_image = img.copy()

        self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.hand_drawing_spec,
            connection_drawing_spec=self.hand_drawing_spec)

        self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.hand_drawing_spec,
            connection_drawing_spec=self.hand_drawing_spec)

        self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=self.mp_holistic.FACE_CONNECTIONS,
            landmark_drawing_spec=self.face_drawing_spec,
            connection_drawing_spec=self.face_drawing_spec)

        self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.pose_drawing_spec,
            connection_drawing_spec=self.pose_drawing_spec)

        return annotated_image

    def process_input_video(self, frame_save_path):

        if self.cap.isOpened() == False:
            print("ERROR FILE NOT FOUND OR WRONG CODEC USED!")

        count = 0

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if ret == True:
                frame = self.rescale_frame(frame, percent=40)
                results = self.holistic.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                height = frame.shape[0]
                width = frame.shape[1]
                white = np.full((height, width, 3), 255, dtype=np.uint8)

                white = self.draw_landmark(white, results)
                frame = self.draw_landmark(frame, results)

                stack = np.hstack((frame, white))

                cv2.imshow("stack", stack)
                fpath = os.path.join(frame_save_path, f"frame{count}.jpg")

                cv2.imwrite(fpath, stack)
                count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

        return None

    def write_output_video(self, video_path, frame_save_path):

        start = time.time()
        num_frames = len(glob.glob(frame_save_path+"/*"))

        img = []
        for i in tqdm.tqdm(range(num_frames)):
            fpath = os.path.join(frame_save_path, f"frame{i}.jpg")
            img.append(cv2.imread(fpath))

        height, width, _ = img[1].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (width, height)
        fps = self.fps
        video = cv2.VideoWriter(video_path, fourcc, fps, size)

        for j in tqdm.tqdm(range(len(img))):
            video.write(img[j])

        cv2.destroyAllWindows()
        video.release()

        end = time.time()
        print(f"{end-start} time elapsed")

        return None

    def show_output_video(self, video_path):

        capture = cv2.VideoCapture(video_path)

        if capture.isOpened() == False:
            print("ERROR FILE NOT FOUND !")

        while capture.isOpened():
            ret, frame = capture.read()
            if ret == True:
                cv2.imshow("Result", frame)
                wait_ms = int(1000/self.fps)
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break
            else:
                break

        capture.release()
        cv2.destroyAllWindows()

        return None


def main():

    parser = argparse.ArgumentParser(
        description="Mediapipe holistic tracking from video")
    parser.add_argument('-i', '--input', type=str, required=True,
                        metavar='', help='Path of input video')
    parser.add_argument('-o', '--output', type=str, required=True,
                        metavar='', help='Path for output video')
    parser.add_argument('-f', '--frame', type=str, required=True,
                        metavar='', help='Path for saving processed frames')
    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output
    frame_save_path = args.frame

    if os.path.exists(input_video_path):

        if not os.path.exists(frame_save_path):
            os.mkdir(frame_save_path)

        hol = Holistic(input_video_path)
        hol.process_input_video(frame_save_path)
        hol.write_output_video(output_video_path, frame_save_path)
        hol.show_output_video(output_video_path)

    return None


if __name__ == '__main__':
    main()
