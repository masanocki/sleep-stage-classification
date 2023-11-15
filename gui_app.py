import customtkinter as ctk
import tkinter as tk
from customtkinter import filedialog
from PIL import Image

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sleep Stage Classification")
        self.geometry("1280x720")

        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.start()

    def start(self):
        # side menu
        self.side_menu = ctk.CTkFrame(
            self, width=160, corner_radius=10, fg_color="transparent"
        )
        self.side_menu.grid(row=0, column=0, rowspan=4, sticky="news")
        self.side_menu.grid_rowconfigure(4, weight=1)
        menu_image = ctk.CTkImage(
            light_image=Image.open("./assets/zzz.png"),
            dark_image=Image.open("./assets/zzz.png"),
            size=(150, 150),
        )
        self.menu_title = ctk.CTkLabel(
            self.side_menu,
            image=menu_image,
            text="",
        )
        self.menu_title.grid(row=0, column=0, padx=30, pady=(30, 60))
        self.custom_pred_button = ctk.CTkButton(
            self.side_menu,
            text="Custom Train Prediction",
            command=self.custom_train_prediction,
            font=ctk.CTkFont(size=15, weight="bold"),
            border_spacing=10,
        )
        self.custom_pred_button.grid(row=1, column=0, padx=20, pady=10)
        self.theme_mode_label = ctk.CTkLabel(
            self.side_menu,
            text="Change Theme:",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.theme_mode_label.grid(row=6, column=0, padx=20, pady=5)
        self.theme_mode_options = ctk.CTkOptionMenu(
            self.side_menu,
            values=["Light", "Dark", "System"],
            command=self.theme_mode,
        )
        self.theme_mode_options.set(ctk.get_appearance_mode())
        self.theme_mode_options.grid(row=7, column=0, padx=20, pady=10)

        # main content
        self.main_content = ctk.CTkFrame(self, corner_radius=10)
        self.main_content.grid(
            row=0, column=2, padx=(7, 2), pady=2, rowspan=4, sticky="news"
        )
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.placeholder_content = ctk.CTkLabel(
            self.main_content, text="HAHAHA", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.placeholder_content.grid(row=0, column=0, padx=20, pady=20, sticky="news")

        self.credentials_label = ctk.CTkLabel(
            self.side_menu,
            text="Designed by Maciej Sanocki, Icons by Icons8",
            font=ctk.CTkFont(size=9),
        )
        self.credentials_label.grid(row=8, column=0, padx=(0, 10), sticky="wes")

    def theme_mode(self, new_theme_mode):
        ctk.set_appearance_mode(new_theme_mode)

    def back_to_start(self):
        self.start()

    def train_edf_file_insert(self):
        filePath = filedialog.askopenfilename()
        self.train_edf_file.insert(0, filePath)

    def custom_train_prediction(self):
        self.placeholder_content.destroy()
        self.main_content.grid_rowconfigure(3, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        self.custom_pred_button = ctk.CTkButton(
            self.side_menu,
            text="Back to Start",
            command=self.back_to_start,
            fg_color="#808080",
            font=ctk.CTkFont(weight="bold"),
        )
        self.custom_pred_button.grid(row=5, column=0, padx=20, pady=10)

        self.first_file = ctk.CTkFrame(self.main_content, corner_radius=10)
        self.first_file.grid_rowconfigure(3, weight=1)
        self.first_file.grid_columnconfigure(0, weight=1)
        self.first_file.grid(row=0, column=0, padx=6, pady=6, sticky="news")
        self.first_file_label = ctk.CTkLabel(
            self.first_file,
            text="Train Dataset",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.first_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        self.train_edf_file_label = ctk.CTkLabel(
            self.first_file, text="Raw Data:", font=ctk.CTkFont(weight="bold")
        )
        self.train_edf_file_label.grid(
            row=1,
            column=0,
            padx=10,
            sticky="w",
        )
        self.train_edf_file = ctk.CTkEntry(
            self.first_file, placeholder_text="PSG Raw data", width=700
        )
        self.train_edf_file.grid(row=1, column=0, padx=10, sticky="s")
        self.train_edf_button = ctk.CTkButton(
            self.first_file,
            text="Upload",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.train_edf_file_insert,
        )
        self.train_edf_button.grid(row=1, column=0, padx=(0, 10), sticky="e")

        self.second_file = ctk.CTkFrame(self.main_content, corner_radius=10)
        self.second_file.grid_rowconfigure(3, weight=1)
        self.second_file.grid_columnconfigure(0, weight=1)
        self.second_file.grid(row=3, column=0, padx=6, pady=6, sticky="news")
        self.file_2_button = ctk.CTkButton(
            self.second_file,
            text="Upload",
            width=250,
            height=90,
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.file_2_button.grid(row=2, column=0, pady=(5, 10), sticky="s")
