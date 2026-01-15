import tkinter as tk
from tkinter import font, messagebox
import cv2
from PIL import Image, ImageTk

# --- Configuration ---
# Colors for the UI elements
BG_COLOR = "#121212"
RIGHT_PANEL_COLOR = "#1e1e1e"
FACE_BOX_COLOR = "#03DAC6"
TEXT_COLOR = "#FFFFFF"
MODE_INACTIVE_COLOR = "#555555"
MODE_ACTIVE_COLOR = "#03DAC6"
FONT_FAMILY = "Segoe UI"

class MemoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AR Memory Support (Live Mock-up)")
        # Set a default size instead of forcing fullscreen, making it resizable
        self.root.geometry("1280x720")
        self.root.configure(bg=BG_COLOR)

        # --- Face Detection Setup ---
        # A more robust way to load the cascade file.
        # This will look in the default OpenCV data path.
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier()
        
        if not self.face_cascade.load(cascade_path):
            # If loading fails, show a user-friendly error and exit.
            messagebox.showerror(
                "Error", 
                "Could not load face cascade classifier.\n"
                "Please ensure OpenCV is installed correctly and the file\n"
                "'haarcascade_frontalface_default.xml' is accessible."
            )
            self.root.destroy()
            return
            
        self.detected_faces = []
        self.current_name = ""

        # --- Main Layout ---
        main_frame = tk.Frame(root, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel (Camera Feed) ---
        self.camera_canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.camera_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Bind the click event for toggling modes
        self.camera_canvas.bind("<Button-1>", self.on_canvas_click)

        # --- Right Panel (Memory & Controls) ---
        # Reduced width to give more space to the camera feed
        right_panel = tk.Frame(main_frame, bg=RIGHT_PANEL_COLOR, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        right_panel.pack_propagate(False)

        # --- Memory Display Widgets ---
        tk.Label(right_panel, text="Memory & Context", font=(FONT_FAMILY, 16, "bold"), bg=RIGHT_PANEL_COLOR, fg=TEXT_COLOR).pack(pady=20, padx=10)
        self.memory_label = tk.Label(right_panel, text="", font=(FONT_FAMILY, 11), bg=RIGHT_PANEL_COLOR, fg=TEXT_COLOR, wraplength=280, justify="left")
        self.memory_label.pack(pady=(20, 10), padx=10, fill=tk.X)

        # --- Control Widgets ---
        control_frame = tk.Frame(right_panel, bg=RIGHT_PANEL_COLOR)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20, padx=10)
        
        open_input_btn = tk.Button(control_frame, text="Open Memory Input", command=self.open_input_window, bg="#333333", fg=TEXT_COLOR, font=(FONT_FAMILY, 10, "bold"))
        open_input_btn.pack(fill=tk.X, pady=5)
        
        exit_btn = tk.Button(control_frame, text="Exit Application", command=self.close_app, bg="#552222", fg=TEXT_COLOR, font=(FONT_FAMILY, 10, "bold"))
        exit_btn.pack(fill=tk.X, pady=5)
        
        # --- Mode Status Variables ---
        self.modes = {
            'L': tk.BooleanVar(value=False),
            'F': tk.BooleanVar(value=False),
            'S': tk.BooleanVar(value=False)
        }
        # Dictionary to store clickable areas for modes
        self.mode_positions = {}

        # --- Initialize Camera ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            self.root.destroy()
            return
            
        self.update_camera_feed()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.input_window = None

    def on_canvas_click(self, event):
        """Handles clicks on the camera canvas to toggle modes."""
        for letter, (x0, y0, x1, y1) in self.mode_positions.items():
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                # Click is inside this mode's bounding box
                current_state = self.modes[letter].get()
                self.modes[letter].set(not current_state)
                # The main loop will handle redrawing, no need to call it here
                break

    def draw_overlays(self):
        """Draws all static and dynamic overlays on the camera canvas."""
        self.camera_canvas.delete("overlay")
        self.mode_positions.clear() # Clear old positions before redrawing

        # 1. Mode Indicators
        mode_font = font.Font(family=FONT_FAMILY, size=16, weight="bold")
        for i, (letter, var) in enumerate(self.modes.items()):
            color = MODE_ACTIVE_COLOR if var.get() else MODE_INACTIVE_COLOR
            x, y = 30 + (i * 30), 30
            self.camera_canvas.create_text(x, y, text=letter, font=mode_font, fill=color, tags="overlay")

            # Calculate and store bounding box for click detection
            width = mode_font.measure(letter)
            height = mode_font.metrics('linespace')
            x0 = x - (width / 2)
            y0 = y - (height / 2)
            x1 = x + (width / 2)
            y1 = y + (height / 2)
            self.mode_positions[letter] = (x0, y0, x1, y1)

        # 2. Dynamic Face Detection Box and Name
        for (x, y, w, h) in self.detected_faces:
            # Draw the rectangle around the face
            self.camera_canvas.create_rectangle(x, y, x+w, y+h, outline=FACE_BOX_COLOR, width=3, tags="overlay")
            # Draw the name on top of the box if a name has been set
            if self.current_name:
                name_font = font.Font(family=FONT_FAMILY, size=14, weight="bold")
                self.camera_canvas.create_text(x + w//2, y - 15, text=self.current_name, font=name_font, fill=FACE_BOX_COLOR, tags="overlay")


    def update_camera_feed(self):
        """Reads a frame, performs face detection, and displays the feed."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # --- Face Detection Logic ---
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use self.face_cascade only if it was loaded correctly
            if not self.face_cascade.empty():
                self.detected_faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            else:
                self.detected_faces = []

            # --- Image Display Logic ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            canvas_w, canvas_h = self.camera_canvas.winfo_width(), self.camera_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                img_pil.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

            # Scale face coordinates to the resized image
            x_scale = img_pil.width / frame.shape[1]
            y_scale = img_pil.height / frame.shape[0]
            
            # Center the image on the canvas
            img_x = (canvas_w - img_pil.width) // 2
            img_y = (canvas_h - img_pil.height) // 2

            scaled_faces = []
            for (x, y, w, h) in self.detected_faces:
                scaled_x = int(x * x_scale) + img_x
                scaled_y = int(y * y_scale) + img_y
                scaled_w = int(w * x_scale)
                scaled_h = int(h * y_scale)
                scaled_faces.append((scaled_x, scaled_y, scaled_w, scaled_h))
            self.detected_faces = scaled_faces

            self.photo = ImageTk.PhotoImage(image=img_pil)
            self.camera_canvas.create_image(img_x, img_y, image=self.photo, anchor=tk.NW)

            self.draw_overlays()

        self.root.after(15, self.update_camera_feed)

    def open_input_window(self):
        """Opens the secondary window for data input."""
        if self.input_window and self.input_window.winfo_exists():
            self.input_window.lift()
            return

        self.input_window = tk.Toplevel(self.root)
        self.input_window.title("Manual Input")
        self.input_window.geometry("400x250")
        self.input_window.configure(bg=BG_COLOR)

        tk.Label(self.input_window, text="Name:", bg=BG_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, 10)).pack(pady=(10, 0))
        name_entry = tk.Entry(self.input_window, width=50, bg="#333333", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
        name_entry.pack()

        tk.Label(self.input_window, text="Memory:", bg=BG_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, 10)).pack(pady=(10, 0))
        memory_text = tk.Text(self.input_window, width=45, height=5, bg="#333333", fg=TEXT_COLOR, insertbackground=TEXT_COLOR, wrap=tk.WORD)
        memory_text.pack()

        show_btn = tk.Button(self.input_window, text="Show on Display", command=lambda: self.update_display(name_entry.get(), memory_text.get("1.0", "end-1c")), bg=FACE_BOX_COLOR, fg=BG_COLOR, font=(FONT_FAMILY, 10, "bold"))
        show_btn.pack(pady=20)

    def update_display(self, name, memory):
        """Updates the name on the canvas and memory in the right-hand panel."""
        self.current_name = name
        self.memory_label.config(text=memory)
        
        self.modes['F'].set(True)
        self.modes['S'].set(True)
        
        if self.input_window:
            self.input_window.destroy()

    def close_app(self):
        """Releases camera resources and closes the application."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app_root = tk.Tk()
    app = MemoryApp(app_root)
    app_root.mainloop()