import numpy as np
import cv2
import time
import tkinter
import threading


class MotionDetector:
    def __init__(self, video):
        self.cap = video
        self.contourThresholdLimit = 1000
        self.threshold = 20
        self.blurAmount = 5
        self.dilateIterations = 3
        self.frameSkip = 0
        
        detect_thread = threading.Thread(target=self.detect_motion)
        detect_thread.daemon = True
        detect_thread.start()

    def set_contour_threshold_limit(self, limit):
        self.contourThresholdLimit = limit

    def detect_motion(self):
        ret, frame1 = self.cap.read()
        ret, frame2 = self.cap.read()
        paused = False

        while self.cap.isOpened():
            diff = cv2.absdiff(frame1, frame2)  # Diff the frames
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
            blur = cv2.GaussianBlur(gray, (self.blurAmount,self.blurAmount), 0)  # Apply a blur
            _, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)  # Find thresholds
            dilated = cv2.dilate(thresh,
                                 None,
                                 iterations=self.dilateIterations)  # Dilate the thresholds to fill in gaps
            contours, _ = cv2.findContours(dilated,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Find the contours from dilated thresholds
            output = frame1.copy()
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)  # find bounding rectangles for each contour

                if cv2.contourArea(contour) < self.contourThresholdLimit:  # Area threshold for movement
                    continue
                
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw boxes from contours
                cv2.putText(output, "Status: {}".format('Movement'), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Show stuff
            # cv2.drawContours(frame1, contours, -1, (0,255,0), 2)  # overlay contours on image.
            # cv2.imshow("diff", diff)
            # cv2.imshow("Gray", gray)
            cv2.imshow("Blur", blur)
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Dilated", dilated)
            cv2.imshow("output", output)

            # Pause functionality, don't update frames if flag is set, otherwise continue. 
            if paused is False:  
                frame1 = frame2
                for i in range(self.frameSkip):
                    _ = self.cap.read()  # skip frame by reading it into a unused variable
                ret, frame2 = self.cap.read()

            k = cv2.waitKey(int((1/30) * 1000))
            if k == 27:  # ESC key
                break
            elif k == ord('p'):  # Pause
                paused = True
                continue
            elif k == ord('r'):  # Resume
                paused = False

        cv2.destroyAllWindows()
        self.cap.release()


class DebugGUI:
    def __init__(self, master, motion):
        self.master = master
        self.motion = motion
        master.title("Simple GUI for testing motion detector variables")
        self.contourThresholdLimit = tkinter.Scale(master, from_=500, to=4000, resolution=100,
                                                   label="Area limit for bounding box, px",
                                                   orient="horizontal", length="200",
                                                   command=self.set_motion_contour_threshold_limit)
        self.threshold = tkinter.Scale(master, from_=0, to=100, resolution=5,
                                       label="Threshold gate limit", orient="horizontal", length="200",
                                       command=self.set_threshold)

        self.blurAmount = tkinter.Scale(master, from_=2, to=21, resolution=2,
                                        label="Blur Amount", orient="horizontal", length="200",
                                        command=self.set_blur_amount)

        self.dilateIterations = tkinter.Scale(master, from_=1, to=10, resolution=1,
                                              label="Dilate Iterations", orient="horizontal", length="200",
                                              command=self.set_dilate_iterations)

        self.frameSkip = tkinter.Scale(master, from_=0, to=5, resolution=1,
                                       label="Frame Skip", orient="horizontal", length="200",
                                       command=self.set_frame_skip)

        self.contourThresholdLimit.pack()
        self.contourThresholdLimit.set(self.motion.contourThresholdLimit)

        self.threshold.pack()
        self.threshold.set(self.motion.threshold)

        self.blurAmount.pack()
        self.blurAmount.set(self.motion.blurAmount)

        self.dilateIterations.pack()
        self.dilateIterations.set(self.motion.dilateIterations)

        self.frameSkip.pack()
        self.frameSkip.set(self.motion.frameSkip)
        tkinter.Button(master, text='test', command=self.print_hello).pack()

    @staticmethod
    def print_hello():
        print("hello!")

    def set_motion_contour_threshold_limit(self, value):
        self.motion.contourThresholdLimit = int(value)

    def set_dilate_iterations(self, value):
        self.motion.dilateIterations = int(value)

    def set_threshold(self, value):
        self.motion.threshold = int(value)

    def set_blur_amount(self, value):
        self.motion.blurAmount = (int(value) + 1)

    def set_frame_skip(self, value):
        self.motion.frameSkip = int(value)


video = cv2.VideoCapture('flight_trimmed.mp4')

detector = MotionDetector(video)

master = tkinter.Tk()
gui = DebugGUI(master, detector)
master.mainloop()
