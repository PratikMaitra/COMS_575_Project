import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
from ultralytics import YOLO 
import os


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


root = ctk.CTk()
root.title("Cricket Ball Detector")

root.attributes('-zoomed', True)

root.grid_columnconfigure((0), weight=0)
root.grid_columnconfigure((1), weight=1)
root.grid_columnconfigure((2), weight=1)
root.grid_rowconfigure((0), weight=1)


fram1=ctk.CTkFrame(root, corner_radius=0,fg_color="transparent")

fram1.grid(row=0, column=0, sticky="nsew")


fram2=ctk.CTkScrollableFrame(root,fg_color="white")

fram2.grid(row=0, column=1, sticky="nsew")

fram3=ctk.CTkScrollableFrame(root,fg_color="white")

fram3.grid(row=0, column=2, sticky="nsew")


myfont= ctk.CTkFont(family="Helvetica",size=14,weight="bold")

sample_image_label = ctk.CTkLabel(fram2, text="")
sample_image_label.pack(fill="both", expand=True)


result_image_label = ctk.CTkLabel(fram3, text="")
result_image_label.pack(fill="both", expand=True)


def load_placeholder_images():
    sample_image = Image.open("cricket1.jpg")
    sample_photo = ImageTk.PhotoImage(sample_image)
    sample_image_label.configure(image=sample_photo)
    sample_image_label.image = sample_photo

    result_image = Image.open("cricket2.jpg")
    result_photo = ImageTk.PhotoImage(result_image)
    result_image_label.configure(image=result_photo)
    result_image_label.image = result_photo
    
def clear_canvas():
    load_placeholder_images()

model = YOLO("runs/detect/my_custom_save/yolov8m_saved_weights.pt")



def get_latest_directory(base_path):
    """ Returns the most recently created directory in the given base path. """
    directories = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    latest_directory = max(directories, key=os.path.getmtime, default=None)
    return latest_directory

def apply_yolo_to_image():
    file_path = filedialog.askopenfilename(
        filetypes=[ ("All Files", "*.*"),("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        try:
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            sample_image_label.configure(image=photo)
            sample_image_label.image = photo

            
            results = model.predict(file_path, classes=32, save=True, project="output", name="predict")
            
            
            latest_results_dir = get_latest_directory("output/")
            if latest_results_dir is not None:
                result_image_path = os.path.join(latest_results_dir, os.path.basename(file_path))
                result_image = Image.open(result_image_path)
                result_photo = ImageTk.PhotoImage(result_image)
                result_image_label.configure(image=result_photo)
                result_image_label.image = result_photo
            else:
                print("No prediction results found.")
        except Exception as e:
            print("An error occurred while processing the image: ", e)


load_button = ctk.CTkButton(
    fram1, text="Load Image and Detect", font=myfont, command=apply_yolo_to_image
)
load_button.pack(pady=20, padx=10)

load_placeholder_images()  


clear_button = ctk.CTkButton(fram1, text="Reset", font=myfont,command=clear_canvas)
clear_button.pack(pady=20, padx=10)


def force_quit():
    root.destroy()

force_quit_button = ctk.CTkButton(fram1, text="Quit", font=myfont,command=force_quit)
force_quit_button.pack(pady=20, padx=10)

root.mainloop()
