import click
import numpy as np
import cv2
import imutils
from pathlib import Path

@click.command()
@click.option(
    "--input_path",
    "-ip",
    default="./",
    required=True,
    help="Path to model and labels file",
)

@click.option(
    "--output_path", "-op", default="./", required=True, help="Path to the output video"
)

@click.option("--save_video", "-s", default=True, type=bool, help="Want to save video?")

def detect_facial_attributes(input_path, output_path, save_video):
    path = Path(input_path)

    # Loading HAAR cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    if save_video:
	    out = cv2.VideoWriter(output_path + "output.avi", -1, 20.0, (640, 480))

    while True:
        # Capture frame-by-frame
        _ , frame = cap.read()
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find faces using Haar cascade
        face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        ## Looping through each face
        for coords in face_coord:
            
            ## Finding co-ordinates of face
            X, Y, w, h = coords
            # print(X, Y, w, h)

            ## Finding frame size
            H, W, _ = frame.shape

            ## Computing larger face co-ordinates
            X_1, X_2 = (max(0, X - int(w * 5)), min(X + int(5 * w), W))
            Y_1, Y_2 = (max(0, Y - int(5 * h)), min(Y + int(5 * h), H))

            ## Cropping face and changing BGR To RGB
            img_cp = frame[Y_1:Y_2, X_1:X_2].copy()
            img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)


            ## Drawing facial boundaries
            cv2.rectangle(
                img=frame,
                pt1=(X, Y),
                pt2=(X + w, Y + h),
                color=(128, 128, 0),
                thickness=2,
            )

        # Display the resulting frame
        cv2.imshow("frame", frame)

        ## Save the resulting frame
        if save_video:
            out.write(frame)

        ## Escape keys
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_facial_attributes()