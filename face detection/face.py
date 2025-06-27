import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading

class FaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Detector App")
        self.root.geometry("800x600")

        # Initialize variables
        self.cap = None
        self.is_running = False
        self.frame = None
        self.face_count = 0
        self.save_dir = "detected_faces"
        os.makedirs(self.save_dir, exist_ok=True)

        # Load DNN model
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )

        # GUI elements
        
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack(pady=10)

        self.label_count = tk.Label(root, text="Faces detected: 0", font=("Arial", 12))
        self.label_count.pack(pady=5)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.btn_webcam = tk.Button(self.button_frame, text="Start Webcam", command=self.start_webcam, font=("Arial", 10))
        self.btn_webcam.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(self.button_frame, text="Stop Webcam", command=self.stop_webcam, font=("Arial", 10), state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_image = tk.Button(self.button_frame, text="Load Image", command=self.load_image, font=("Arial", 10))
        self.btn_image.pack(side=tk.LEFT, padx=5)

        self.btn_video = tk.Button(self.button_frame, text="Load Video", command=self.load_video, font=("Arial", 10))
        self.btn_video.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(self.button_frame, text="Save Screenshot", command=self.save_screenshot, font=("Arial", 10))
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_save_faces = tk.Button(self.button_frame, text="Save Faces", command=self.save_faces, font=("Arial", 10))
        self.btn_save_faces.pack(side=tk.LEFT, padx=5)

    def load_dnn_model(self):
        # Ensure DNN model files are available
        proto_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            messagebox.showerror("Error", "DNN model files (deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel) not found!")
            return False
        return True

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i+1}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.face_count = len(faces)
        self.label_count.config(text=f"Faces detected: {self.face_count}")
        return frame, faces

    def update_frame(self):
        if self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame, faces = self.detect_faces(frame)
                self.frame = frame  # Store current frame for saving
                self.faces = faces  # Store detected faces
                frame = cv2.resize(frame, (640, 480))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.root.after(10, self.update_frame)

    def start_webcam(self):
        if not self.load_dnn_model():
            return
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam!")
                return
            self.is_running = True
            self.btn_webcam.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.update_frame()

    def stop_webcam(self):
        if self.is_running:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.canvas.delete("all")
            self.label_count.config(text="Faces detected: 0")
            self.btn_webcam.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def load_image(self):
        if not self.load_dnn_model():
            return
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.stop_webcam()
            frame = cv2.imread(file_path)
            if frame is None:
                messagebox.showerror("Error", "Cannot load image!")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, faces = self.detect_faces(frame)
            self.frame = frame
            self.faces = faces
            frame = cv2.resize(frame, (640, 480))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def load_video(self):
        if not self.load_dnn_model():
            return
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.stop_webcam()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot load video!")
                return
            self.is_running = True
            self.btn_webcam.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.update_frame()

    def save_screenshot(self):
        if self.frame is not None:
            screenshot_path = os.path.join(self.save_dir, f"screenshot_{len(os.listdir(self.save_dir)) + 1}.png")
            frame_bgr = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(screenshot_path, frame_bgr)
            messagebox.showinfo("Success", f"Screenshot saved as {screenshot_path}")

    def save_faces(self):
        if self.frame is not None and hasattr(self, 'faces'):
            for i, (startX, startY, endX, endY) in enumerate(self.faces):
                face_img = self.frame[startY:endY, startX:endX]
                if face_img.size > 0:
                    face_path = os.path.join(self.save_dir, f"face_{len(os.listdir(self.save_dir)) + 1}.png")
                    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_path, face_bgr)
            if self.faces:
                messagebox.showinfo("Success", f"{len(self.faces)} face(s) saved in {self.save_dir}")
            else:
                messagebox.showwarning("Warning", "No faces detected to save!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectorApp(root)
    root.mainloop()
