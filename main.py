"""
Main entry point for the Alzheimer AR Memory Support System
"""

import sys

def run_ui():
    from ui.ui_interface import MemoryApp
    import tkinter as tk

    root = tk.Tk()
    app = MemoryApp(root)
    root.mainloop()


def run_ml_test():
    from ml.test_recognition import main as test_main
    test_main()


if __name__ == "__main__":
    """
    Usage:
    python main.py        -> runs full UI
    python main.py test   -> runs ML recognition test
    """

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("[INFO] Running ML recognition test...")
        run_ml_test()
    else:
        print("[INFO] Starting AR Memory UI...")
        run_ui()
