import random
import sys
import customtkinter as ctk
import tkinter as tk
import time
from tkinter import ttk
from customtkinter import filedialog
from PIL import Image
from custompred import CustomTrainPredict
from io import StringIO
from threading import Thread

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
            command=self.result_screen,
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
            fg_color="#282828",
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
        self.train_edf_file.configure(fg_color="#343638")
        self.train_edf_file.delete(0, "end")
        self.train_edf_file.insert(0, filePath)

    def train_annotation_file_insert(self):
        filePath = filedialog.askopenfilename()
        self.train_annotation_file.configure(fg_color="#343638")
        self.train_annotation_file.delete(0, "end")
        self.train_annotation_file.insert(0, filePath)

    def test_edf_file_insert(self):
        filePath = filedialog.askopenfilename()
        self.test_edf_file.configure(fg_color="#343638")
        self.test_edf_file.delete(0, "end")
        self.test_edf_file.insert(0, filePath)

    def test_annotation_file_insert(self):
        filePath = filedialog.askopenfilename()
        self.test_annotation_file.configure(fg_color="#343638")
        self.test_annotation_file.delete(0, "end")
        self.test_annotation_file.insert(0, filePath)

    def n_estimators_progressbar_on_change(self, value):
        self.n_estimators_value.delete(0, "end")
        self.n_estimators_value.insert(0, int(value))

    def n_estimators_value_on_change(self, value):
        if value.isdigit():
            if int(value) > 200:
                self.n_estimators_progressbar.set(1)
                self.n_estimators_value.configure(fg_color="#ff0033")
                return True

            if self.n_estimators_value.cget("fg_color") == "#ff0033":
                self.n_estimators_value.configure(fg_color="#343638")
            self.n_estimators_progressbar.set(int(value))
        return True

    def min_samples_leaf_progressbar_on_change(self, value):
        self.min_samples_leaf_value.delete(0, "end")
        self.min_samples_leaf_value.insert(0, int(value))

    def min_samples_leaf_value_on_change(self, value):
        if value.isdigit():
            if int(value) > 100:
                self.min_samples_leaf_progressbar.set(1)
                self.min_samples_leaf_value.configure(fg_color="#ff0033")
                return True

            if self.min_samples_leaf_value.cget("fg_color") == "#ff0033":
                self.min_samples_leaf_value.configure(fg_color="#343638")
            self.min_samples_leaf_progressbar.set(int(value))
        return True

    def generate_random_state(self):
        self.random_state_value.delete(0, "end")
        self.random_state_value.insert(0, random.randint(0, 4294967295))

    def make_custom_train_prediction(self):
        if self.max_features_optionmenu.get() == "None":
            clf = CustomTrainPredict(
                train_raw_data_path=self.train_edf_file.get(),
                train_annotations_path=self.train_annotation_file.get(),
                test_raw_data_path=self.test_edf_file.get(),
                test_annotations_path=self.test_annotation_file.get(),
                n_estimators=int(self.n_estimators_value.get()),
                min_samples_leaf=int(self.min_samples_leaf_value.get()),
                max_features=None,
                random_state=int(self.random_state_value.get()),
            )
        else:
            clf = CustomTrainPredict(
                train_raw_data_path=self.train_edf_file.get(),
                train_annotations_path=self.train_annotation_file.get(),
                test_raw_data_path=self.test_edf_file.get(),
                test_annotations_path=self.test_annotation_file.get(),
                n_estimators=int(self.n_estimators_value.get()),
                min_samples_leaf=int(self.min_samples_leaf_value.get()),
                max_features=self.max_features_optionmenu.get(),
                random_state=int(self.random_state_value.get()),
            )
        self.loading_screen()
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        self.after(1, lambda: self.console_output_textbox_insert(my_stdout, "0.0"))
        self.after(1, lambda: self.loading_screen_gif_update(0))
        clf.predict()

        sys.stdout = old_stdout

    def console_output_textbox_insert(self, my_stdout, progress_check):
        self.console_output_textbox.configure(state="normal")
        self.console_output_textbox.delete("0.0", "end")
        self.console_output_textbox.insert("end", my_stdout.getvalue())
        self.console_output_textbox.configure(state="disabled")
        self.console_output_textbox.see("end")
        if self.console_output_textbox.index("end") != progress_check:
            elements_done = int(float(self.console_output_textbox.index("end"))) - int(
                float(progress_check)
            )
            self.prediction_progress_bar.set(
                self.prediction_progress_bar.get() + 0.024 * elements_done
            )
        if self.t.is_alive():
            self.after(
                1,
                lambda: self.console_output_textbox_insert(
                    my_stdout, self.console_output_textbox.index("end")
                ),
            )

    def loading_screen_gif_update(self, i):
        if i > 96:
            i = 2
        self.loading_animation.configure(image=self.animation_gif_frames[i])
        if self.t.is_alive():
            self.after(50, lambda: self.loading_screen_gif_update(i + 1))

    def start_make_custom_train_prediction(self):
        if self.custom_train_prediction_validation():
            self.t = Thread(target=self.make_custom_train_prediction, daemon=True)
            self.t.start()

    def create_gif(self, frames, framescnt):
        result = []
        for i in range(framescnt):
            gif_frame = ctk.CTkImage(
                light_image=frames.copy(),
                dark_image=frames.copy(),
                size=(350, 350),
            )
            result.append(gif_frame)
            frames.seek(i)
        return result

    def result_screen(self):
        # container
        self.result_tab_view = ctk.CTkTabview(self.main_content)
        self.result_tab_view.grid(row=0, column=0, sticky="news")
        self.result_tab_view.add("Summary")
        self.result_tab_view.add("Classification Report")
        self.result_tab_view.set("Summary")

        # first result screen
        self.result_tab_view.tab("Summary").grid_rowconfigure(7, weight=1)
        self.result_tab_view.tab("Summary").grid_columnconfigure(0, weight=1)
        self.first_result_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="Summary",
            font=ctk.CTkFont(size=30, weight="bold"),
        )
        self.first_result_label.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
        self.train_result_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="Train dataset:",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.train_result_label.grid(
            row=2, column=0, padx=(10, 5), pady=10, sticky="nw"
        )
        self.train_result_info = ctk.CTkTextbox(
            self.result_tab_view.tab("Summary"),
            fg_color=["gray85", "gray16"],
            font=ctk.CTkFont(size=18),
            width=400,
            height=265,
        )
        self.train_result_info.grid(row=3, column=0, padx=(10, 5), sticky="nw")
        self.test_result_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="Test dataset:",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.test_result_label.grid(
            row=2, column=0, padx=(5, 280), pady=10, sticky="ne"
        )
        self.test_result_info = ctk.CTkTextbox(
            self.result_tab_view.tab("Summary"),
            fg_color=["gray85", "gray16"],
            font=ctk.CTkFont(size=18),
            width=400,
            height=265,
        )
        self.test_result_info.grid(row=3, column=0, padx=(5, 10), sticky="ne")
        self.infos_separator = ctk.CTkFrame(
            self.result_tab_view.tab("Summary"), fg_color="#181818", height=240, width=2
        )
        self.infos_separator.grid(row=3, column=0, sticky="n")
        self.result_config_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="Algorithm config:",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.result_config_label.grid(row=4, column=0, padx=(5, 10), sticky="nw")

        self.result_n_estimators_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="n_estimators:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_n_estimators_label.grid(
            row=5, column=0, padx=(5, 10), pady=10, sticky="nw"
        )
        self.result_n_estimators_value = ctk.CTkEntry(
            self.result_tab_view.tab("Summary"),
            font=ctk.CTkFont(size=18),
            width=50,
            fg_color=["gray85", "gray16"],
            border_color=["gray85", "gray16"],
        )
        self.result_n_estimators_value.grid(
            row=5, column=0, padx=(130, 10), pady=10, sticky="nw"
        )
        self.result_min_samples_leaf_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="min_samples_leaf:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_min_samples_leaf_label.grid(
            row=5, column=0, padx=(200, 10), pady=10, sticky="nw"
        )
        self.result_min_samples_leaf_value = ctk.CTkEntry(
            self.result_tab_view.tab("Summary"),
            font=ctk.CTkFont(size=18),
            width=50,
            fg_color=["gray85", "gray16"],
            border_color=["gray85", "gray16"],
        )
        self.result_min_samples_leaf_value.grid(
            row=5, column=0, padx=(370, 10), pady=10, sticky="nw"
        )
        self.result_max_features_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="max_features:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_max_features_label.grid(row=5, column=0, pady=10, sticky="n")
        self.result_max_features_value = ctk.CTkEntry(
            self.result_tab_view.tab("Summary"),
            font=ctk.CTkFont(size=18),
            width=60,
            fg_color=["gray85", "gray16"],
            border_color=["gray85", "gray16"],
        )
        self.result_max_features_value.grid(
            row=5, column=0, padx=(210, 10), pady=10, sticky="n"
        )
        self.result_random_state_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="random_state:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_random_state_label.grid(
            row=5, column=0, padx=(430, 10), pady=10, sticky="n"
        )
        self.result_random_state_value = ctk.CTkEntry(
            self.result_tab_view.tab("Summary"),
            font=ctk.CTkFont(size=18),
            width=200,
            fg_color=["gray85", "gray16"],
            border_color=["gray85", "gray16"],
        )
        self.result_random_state_value.grid(
            row=5, column=0, padx=(770, 10), pady=10, sticky="n"
        )
        self.result_accuracy_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="Accuracy:",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.result_accuracy_label.grid(
            row=6, column=0, padx=(5, 10), pady=10, sticky="nw"
        )
        self.result_accuracy_value = ctk.CTkEntry(
            self.result_tab_view.tab("Summary"),
            font=ctk.CTkFont(size=20),
            width=400,
            fg_color=["gray85", "gray16"],
            border_color=["gray85", "gray16"],
        )
        self.result_accuracy_value.grid(
            row=6, column=0, padx=(100, 10), pady=10, sticky="nw"
        )
        self.result_total_time_label = ctk.CTkLabel(
            self.result_tab_view.tab("Summary"),
            text="Total Time:",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.result_total_time_label.grid(
            row=7, column=0, padx=(5, 10), pady=10, sticky="nw"
        )
        self.result_total_time_value = ctk.CTkEntry(
            self.result_tab_view.tab("Summary"),
            font=ctk.CTkFont(size=20),
            width=400,
            fg_color=["gray85", "gray16"],
            border_color=["gray85", "gray16"],
        )
        self.result_total_time_value.grid(
            row=7, column=0, padx=(115, 10), pady=10, sticky="nw"
        )

        # second result screen
        self.result_tab_view.tab("Classification Report").grid_rowconfigure(0, weight=1)
        self.result_tab_view.tab("Classification Report").grid_columnconfigure(
            0, weight=1
        )
        self.classification_report_textbox = ctk.CTkTextbox(
            self.result_tab_view.tab("Classification Report"),
            fg_color=["gray85", "gray16"],
            font=ctk.CTkFont(size=40),
        )
        self.classification_report_textbox.grid(row=0, column=0, sticky="news")

        self.clf = CustomTrainPredict(
            train_raw_data_path="./datasets/recordings/SN001.edf",
            train_annotations_path="./datasets/recordings/SN001_sleepscoring.edf",
            test_raw_data_path="./datasets/recordings/SN002.edf",
            test_annotations_path="./datasets/recordings/SN002_sleepscoring.edf",
            n_estimators=100,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        )
        self.start_pred = time.time()
        (
            self.tr,
            self.ts,
            self.n_est,
            self.min_sampl,
            self.max_feat,
            self.rand_st,
            self.accuracy,
            self.rep,
        ) = self.clf.predict()
        self.stop_pred = time.time()
        self.total_time = self.stop_pred - self.start_pred

        # test setters
        ###
        self.train_result_info.insert("0.0", str(self.tr)[27:-1])
        self.test_result_info.insert("0.0", str(self.ts)[27:-1])
        self.result_n_estimators_value.insert(0, self.n_est)
        self.result_min_samples_leaf_value.insert(0, self.min_sampl)
        self.result_max_features_value.insert(0, self.max_feat)
        self.result_random_state_value.insert(0, self.rand_st)
        self.result_accuracy_value.insert(0, round(self.accuracy, 2))
        self.result_total_time_value.insert(0, round(self.total_time, 2))
        self.classification_report_textbox.insert("0.0", self.rep)

        self.classification_report_textbox.configure(state="disabled")
        self.result_total_time_value.configure(state="disabled")
        self.result_accuracy_value.configure(state="disabled")
        self.result_random_state_value.configure(state="disabled")
        self.result_max_features_value.configure(state="disabled")
        self.result_min_samples_leaf_value.configure(state="disabled")
        self.result_n_estimators_value.configure(state="disabled")
        self.train_result_info.configure(state="disabled")
        self.test_result_info.configure(state="disabled")

    def loading_screen(self):
        self.first_file.grid_forget()
        self.loading_screen_frame = ctk.CTkFrame(self.main_content, corner_radius=10)
        self.loading_screen_frame.grid(row=0, column=0, sticky="news")
        self.loading_screen_frame.grid_rowconfigure(0, weight=1)
        self.loading_screen_frame.grid_columnconfigure(0, weight=1)
        self.animation_gif_whole = Image.open("./assets/loading_screen_animation.gif")

        self.animation_gif_frames = self.create_gif(self.animation_gif_whole, 97)

        self.loading_animation = ctk.CTkLabel(
            self.loading_screen_frame, text="", image=self.animation_gif_frames[2]
        )
        self.loading_animation.grid(row=0, column=0, pady=(0, 250))

        self.prediction_progress_bar = ctk.CTkProgressBar(
            self.loading_screen_frame,
            orientation="horizontal",
            mode="determinate",
            height=25,
        )
        self.prediction_progress_bar.grid(row=0, column=0, padx=10, sticky="we")
        self.prediction_progress_bar.set(0)
        self.console_output_textbox = ctk.CTkTextbox(
            self.loading_screen_frame, height=310, state="disabled"
        )
        self.console_output_textbox.grid(
            row=0, column=0, padx=10, pady=(0, 10), sticky="wes"
        )

    def custom_train_prediction_validation(self):
        flag = True
        if self.train_edf_file.get() == "":
            self.train_edf_file.configure(fg_color="#ff0033")
            flag = False
        if self.train_annotation_file.get() == "":
            self.train_annotation_file.configure(fg_color="#ff0033")
            flag = False
        if self.test_edf_file.get() == "":
            self.test_edf_file.configure(fg_color="#ff0033")
            flag = False
        if self.test_annotation_file.get() == "":
            self.test_annotation_file.configure(fg_color="#ff0033")
            flag = False
        return flag

    def custom_train_prediction_screen(self):
        self.placeholder_content.destroy()
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.custom_pred_button.configure(fg_color="#282828")

        self.custom_pred_back_to_start_button = ctk.CTkButton(
            self.side_menu,
            text="Back to Start",
            command=self.back_to_start,
            font=ctk.CTkFont(weight="bold"),
        )
        self.custom_pred_back_to_start_button.grid(row=5, column=0, padx=20, pady=10)

        self.first_file = ctk.CTkFrame(self.main_content, corner_radius=10)
        self.first_file.grid_rowconfigure(12, weight=1)
        self.first_file.grid_columnconfigure(0, weight=1)
        self.first_file.grid(row=0, column=0, sticky="news")
        self.first_file_label = ctk.CTkLabel(
            self.first_file,
            text="Train Datasets",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.first_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        self.train_edf_file_label = ctk.CTkLabel(
            self.first_file, text="Raw Data:", font=ctk.CTkFont(size=15, weight="bold")
        )
        self.train_edf_file_label.grid(
            row=1,
            column=0,
            padx=10,
            pady=(10, 0),
            sticky="w",
        )
        self.train_edf_file = ctk.CTkEntry(
            self.first_file, placeholder_text="PSG Raw data", width=700
        )
        self.train_edf_file.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="s")
        self.train_edf_button = ctk.CTkButton(
            self.first_file,
            text="Upload",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.train_edf_file_insert,
        )
        self.train_edf_button.grid(
            row=1, column=0, padx=(0, 10), pady=(10, 0), sticky="e"
        )

        self.train_annotation_file_label = ctk.CTkLabel(
            self.first_file,
            text="Annotations:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.train_annotation_file_label.grid(
            row=2,
            column=0,
            padx=10,
            pady=(20, 0),
            sticky="w",
        )
        self.train_annotation_file = ctk.CTkEntry(
            self.first_file, placeholder_text="Data for Annotations", width=700
        )
        self.train_annotation_file.grid(
            row=2, column=0, pady=(20, 0), padx=10, sticky="s"
        )
        self.train_annotation_button = ctk.CTkButton(
            self.first_file,
            text="Upload",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.train_annotation_file_insert,
        )
        self.train_annotation_button.grid(
            row=2, column=0, padx=(0, 10), pady=(20, 0), sticky="e"
        )

        self.second_file_label = ctk.CTkLabel(
            self.first_file,
            text="Test Datasets",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.second_file_label.grid(
            row=4, column=0, padx=10, pady=(50, 10), sticky="nw"
        )

        self.test_edf_file_label = ctk.CTkLabel(
            self.first_file, text="Raw Data:", font=ctk.CTkFont(size=15, weight="bold")
        )
        self.test_edf_file_label.grid(
            row=5,
            column=0,
            padx=10,
            pady=(10, 0),
            sticky="w",
        )
        self.test_edf_file = ctk.CTkEntry(
            self.first_file, placeholder_text="PSG Raw data", width=700
        )
        self.test_edf_file.grid(row=5, column=0, padx=10, pady=(10, 0), sticky="s")
        self.test_edf_button = ctk.CTkButton(
            self.first_file,
            text="Upload",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.test_edf_file_insert,
        )
        self.test_edf_button.grid(
            row=5, column=0, padx=(0, 10), pady=(10, 0), sticky="e"
        )

        self.test_annotation_file_label = ctk.CTkLabel(
            self.first_file,
            text="Annotations:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.test_annotation_file_label.grid(
            row=6,
            column=0,
            padx=10,
            pady=(20, 0),
            sticky="w",
        )
        self.test_annotation_file = ctk.CTkEntry(
            self.first_file, placeholder_text="Data for Annotations", width=700
        )
        self.test_annotation_file.grid(
            row=6, column=0, padx=10, pady=(20, 0), sticky="s"
        )
        self.test_annotation_button = ctk.CTkButton(
            self.first_file,
            text="Upload",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.test_annotation_file_insert,
        )
        self.test_annotation_button.grid(
            row=6, column=0, padx=(0, 10), pady=(20, 0), sticky="e"
        )

        self.adjustment_label = ctk.CTkLabel(
            self.first_file,
            text="Parameters Tuning",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.adjustment_label.grid(row=7, column=0, padx=10, pady=(50, 10), sticky="nw")

        self.n_estimators_label = ctk.CTkLabel(
            self.first_file,
            text="n_estimators:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.n_estimators_label.grid(
            row=8, column=0, padx=10, pady=(20, 0), sticky="nw"
        )

        self.n_estimators_progressbar = ctk.CTkSlider(
            self.first_file,
            from_=1,
            to=200,
            number_of_steps=200,
            width=700,
            command=self.n_estimators_progressbar_on_change,
        )
        self.n_estimators_progressbar.grid(row=8, column=0, pady=(20, 0))

        self.n_estimators_reg = self.register(self.n_estimators_value_on_change)

        self.n_estimators_value = ctk.CTkEntry(
            self.first_file,
            validate="key",
            validatecommand=(self.n_estimators_reg, "%P"),
            justify="center",
        )
        self.n_estimators_value.grid(
            row=8, column=0, padx=(0, 10), pady=(20, 0), sticky="e"
        )
        test = ctk.IntVar()
        self.min_samples_leaf_label = ctk.CTkLabel(
            self.first_file,
            text="min_samples_leaf:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.min_samples_leaf_label.grid(
            row=9, column=0, padx=10, pady=(20, 0), sticky="nw"
        )

        self.min_samples_leaf_progressbar = ctk.CTkSlider(
            self.first_file,
            from_=1,
            to=100,
            number_of_steps=100,
            width=700,
            command=self.min_samples_leaf_progressbar_on_change,
        )
        self.min_samples_leaf_progressbar.grid(row=9, column=0, pady=(20, 0))

        self.min_samples_leaf_reg = self.register(self.min_samples_leaf_value_on_change)

        self.min_samples_leaf_value = ctk.CTkEntry(
            self.first_file,
            validate="key",
            validatecommand=(self.min_samples_leaf_reg, "%P"),
            justify="center",
        )
        self.min_samples_leaf_value.grid(
            row=9, column=0, padx=(0, 10), pady=(20, 0), sticky="e"
        )

        self.max_features_label = ctk.CTkLabel(
            self.first_file,
            text="max_features:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.max_features_label.grid(
            row=10, column=0, padx=10, pady=(20, 0), sticky="nw"
        )

        self.max_features_optionmenu = ctk.CTkOptionMenu(
            self.first_file,
            values=["sqrt", "log2", "None"],
            anchor="center",
            font=ctk.CTkFont(size=13, weight="bold"),
            dropdown_font=ctk.CTkFont(size=13, weight="bold"),
        )

        self.max_features_optionmenu.grid(
            row=10, column=0, padx=(130, 0), pady=(20, 0), sticky="nw"
        )

        self.random_state_label = ctk.CTkLabel(
            self.first_file,
            text="random_state:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.random_state_label.grid(
            row=11, column=0, padx=10, pady=(20, 0), sticky="nw"
        )

        self.random_state_value = ctk.CTkEntry(
            self.first_file, width=200, justify="center"
        )

        self.random_state_value.grid(
            row=11, column=0, padx=(130, 0), pady=(20, 0), sticky="nw"
        )
        self.random_state_generate_button = ctk.CTkButton(
            self.first_file,
            text="Generate",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.generate_random_state,
        )
        self.random_state_generate_button.grid(
            row=11, column=0, padx=(340, 0), pady=(20, 0), sticky="nw"
        )

        self.custom_predict_clear_button = ctk.CTkButton(
            self.first_file,
            text="Clear",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.custom_train_prediction_screen,
        )

        self.custom_predict_clear_button.grid(
            row=13, column=0, padx=(0, 170), pady=(0, 20), sticky="e"
        )

        self.custom_predict_start_button = ctk.CTkButton(
            self.first_file,
            text="Start",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.start_make_custom_train_prediction,
        )

        self.custom_predict_start_button.grid(
            row=13, column=0, padx=(0, 10), pady=(0, 20), sticky="e"
        )

        # default values setters
        self.n_estimators_progressbar.set(100)
        self.n_estimators_value.insert(0, 100)
        self.min_samples_leaf_progressbar.set(1)
        self.min_samples_leaf_value.insert(0, 1)
        self.random_state_value.insert(0, 42)
