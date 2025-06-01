#!/usr/bin/env python3
"""
Empathy Chatbot UI

A modern, user-friendly chat interface for interacting with empathy models.
Features:
- Beautiful, responsive UI with dark/light mode
- Support for multiple models (OpenAI, Simple, Local)
- Chat history and session management
- Response analysis and feedback
"""

import os
import sys
import json
import time
import uuid
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import requests
from datetime import datetime
from dotenv import load_dotenv
import webbrowser

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import the simple empathy model
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple_empathy_model import SimpleEmpathyModel
    SIMPLE_MODEL_AVAILABLE = True
except ImportError:
    SIMPLE_MODEL_AVAILABLE = False

# Try to import the response evaluator
try:
    from response_quality_evaluator import ResponseEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    # Create a minimal version for standalone use
    class ResponseEvaluator:
        def __init__(self):
            # Define empathy indicators
            self.validation_phrases = [
                "valid", "understand", "makes sense", "natural", "reasonable", 
                "understandable", "normal", "okay to feel", "it's okay", "it is okay",
                "you're not alone", "you are not alone", "many people", "common",
                "i hear you", "that sounds", "that must be", "that's really",
                "that is really", "i can imagine", "it's difficult", "it is difficult"
            ]
            
            self.reflection_phrases = [
                "you feel", "you're feeling", "you are feeling", "you seem", 
                "you sound", "you mentioned", "you've been", "you have been",
                "you're going through", "you are going through", "you've experienced",
                "you have experienced", "it sounds like you", "it seems like you",
                "you're saying", "you are saying"
            ]
            
            self.support_phrases = [
                "here for you", "support", "help", "listen", "there for you",
                "by your side", "alongside you", "with you", "together", "through this",
                "i'm here", "i am here", "we can", "let's", "let us", "take care",
                "be gentle", "be kind", "self-care", "self care"
            ]
        
        def count_empathy_indicators(self, text):
            """Count empathy indicators in a response"""
            text = text.lower()
            
            validation_count = sum(1 for phrase in self.validation_phrases if phrase in text)
            reflection_count = sum(1 for phrase in self.reflection_phrases if phrase in text)
            support_count = sum(1 for phrase in self.support_phrases if phrase in text)
            
            return {
                "validation": validation_count,
                "reflection": reflection_count,
                "support": support_count,
                "total": validation_count + reflection_count + support_count
            }

# Load environment variables
load_dotenv()

class ScrollableFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

class ChatMessage(ctk.CTkFrame):
    def __init__(self, master, message, is_user=True, metrics=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        
        # Set colors based on sender
        if is_user:
            bg_color = "#DCF8C6"  # Light green for user
            text_color = "#000000"
        else:
            bg_color = "#FFFFFF"  # White for assistant
            text_color = "#000000"
        
        # Message frame
        self.message_frame = ctk.CTkFrame(self, fg_color=bg_color, corner_radius=10)
        self.message_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.message_frame.grid_columnconfigure(0, weight=1)
        
        # Sender label
        sender = "You" if is_user else "Assistant"
        sender_label = ctk.CTkLabel(
            self.message_frame, 
            text=sender, 
            text_color="#555555", 
            font=("Arial", 10, "bold")
        )
        sender_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
        
        # Message text
        message_text = ctk.CTkLabel(
            self.message_frame, 
            text=message, 
            text_color=text_color,
            font=("Arial", 12),
            wraplength=500,
            justify="left"
        )
        message_text.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))
        
        # Add metrics if provided and not user message
        if metrics and not is_user:
            # Metrics frame
            metrics_frame = ctk.CTkFrame(self.message_frame, fg_color="transparent")
            metrics_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
            
            # Validation indicator
            validation_label = ctk.CTkLabel(
                metrics_frame, 
                text=f"Validation: {metrics['validation']}", 
                text_color="#FF6B6B",
                font=("Arial", 10)
            )
            validation_label.grid(row=0, column=0, padx=5)
            
            # Reflection indicator
            reflection_label = ctk.CTkLabel(
                metrics_frame, 
                text=f"Reflection: {metrics['reflection']}", 
                text_color="#4ECDC4",
                font=("Arial", 10)
            )
            reflection_label.grid(row=0, column=1, padx=5)
            
            # Support indicator
            support_label = ctk.CTkLabel(
                metrics_frame, 
                text=f"Support: {metrics['support']}", 
                text_color="#6A0572",
                font=("Arial", 10)
            )
            support_label.grid(row=0, column=2, padx=5)
            
            # Total indicator
            total_label = ctk.CTkLabel(
                metrics_frame, 
                text=f"Total: {metrics['total']}", 
                text_color="#1A535C",
                font=("Arial", 10, "bold")
            )
            total_label.grid(row=0, column=3, padx=5)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        time_label = ctk.CTkLabel(
            self.message_frame, 
            text=timestamp, 
            text_color="#999999", 
            font=("Arial", 8)
        )
        time_label.grid(row=3 if metrics and not is_user else 2, column=0, sticky="e", padx=10, pady=(0, 5))

class EmpathyChatbotUI:
    def __init__(self, root):
        """Initialize the chatbot UI"""
        self.root = root
        self.root.title("Empathy Chatbot")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)
        
        # Set appearance mode
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize models
        self.init_models()
        
        # Initialize evaluator
        self.evaluator = ResponseEvaluator()
        
        # Chat history
        self.chat_history = []
        
        # Create the main layout
        self.create_layout()
        
        # Bind events
        self.user_input.bind("<Return>", self.on_enter_pressed)
        
        # Welcome message
        self.add_message("Hello! I'm here to listen and support you. How are you feeling today?", is_user=False)
    
    def init_models(self):
        """Initialize available models"""
        # OpenAI model
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
        
        # Simple model
        self.simple_model = None
        if SIMPLE_MODEL_AVAILABLE:
            self.simple_model = SimpleEmpathyModel()
        
        # Available models
        self.models = {}
        
        if self.openai_client:
            self.models["OpenAI Fine-tuned"] = {
                "type": "openai",
                "id": "ft:gpt-3.5-turbo-0125:valis::BYfKr10K",
                "system_message": "You are an empathetic assistant that provides supportive and compassionate responses."
            }
            
            self.models["OpenAI Base"] = {
                "type": "openai",
                "id": "gpt-3.5-turbo",
                "system_message": "You are an empathetic assistant that provides supportive and compassionate responses."
            }
        
        if self.simple_model:
            self.models["Simple Model"] = {
                "type": "simple"
            }
        
        # Default model
        self.current_model = next(iter(self.models.keys())) if self.models else None
    
    def create_layout(self):
        """Create the application layout"""
        # Main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Header
        self.create_header()
        
        # Chat area
        self.create_chat_area()
        
        # Input area
        self.create_input_area()
    
    def create_header(self):
        """Create the header area"""
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Empathy Chatbot", 
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Model selection
        if self.models:
            model_label = ctk.CTkLabel(
                header_frame, 
                text="Model:", 
                font=("Arial", 12)
            )
            model_label.grid(row=0, column=1, padx=(20, 5), pady=5, sticky="e")
            
            self.model_var = ctk.StringVar(value=self.current_model)
            model_dropdown = ctk.CTkOptionMenu(
                header_frame, 
                values=list(self.models.keys()),
                variable=self.model_var,
                command=self.on_model_change
            )
            model_dropdown.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        
        # Settings button
        settings_button = ctk.CTkButton(
            header_frame, 
            text="⚙️", 
            width=30, 
            command=self.show_settings
        )
        settings_button.grid(row=0, column=3, padx=5, pady=5, sticky="e")
        
        # Theme toggle
        self.theme_var = ctk.StringVar(value="dark")
        theme_switch = ctk.CTkSwitch(
            header_frame, 
            text="Light Mode", 
            variable=self.theme_var, 
            onvalue="light", 
            offvalue="dark",
            command=self.toggle_theme
        )
        theme_switch.grid(row=0, column=4, padx=10, pady=5, sticky="e")
    
    def create_chat_area(self):
        """Create the chat area"""
        # Chat frame
        self.chat_frame = ScrollableFrame(
            self.main_container, 
            width=200, 
            height=300,
            fg_color=("gray90", "gray13")
        )
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    
    def create_input_area(self):
        """Create the input area"""
        input_frame = ctk.CTkFrame(self.main_container)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        # User input
        self.user_input = ctk.CTkTextbox(
            input_frame, 
            height=60, 
            wrap="word",
            font=("Arial", 12)
        )
        self.user_input.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=5)
        
        # Send button
        send_button = ctk.CTkButton(
            input_frame, 
            text="Send", 
            width=80, 
            command=self.send_message
        )
        send_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Focus on input
        self.user_input.focus_set()
    
    def add_message(self, message, is_user=True, metrics=None):
        """Add a message to the chat"""
        # Create message widget
        message_widget = ChatMessage(
            self.chat_frame, 
            message, 
            is_user=is_user, 
            metrics=metrics
        )
        message_widget.pack(fill="x", padx=5, pady=5)
        
        # Add to history
        self.chat_history.append({
            "role": "user" if is_user else "assistant",
            "content": message,
            "metrics": metrics
        })
        
        # Scroll to bottom
        self.chat_frame._parent_canvas.yview_moveto(1.0)
    
    def send_message(self):
        """Send a user message and get a response"""
        # Get message
        message = self.user_input.get("0.0", "end").strip()
        
        if not message:
            return
        
        # Add user message to chat
        self.add_message(message, is_user=True)
        
        # Clear input
        self.user_input.delete("0.0", "end")
        
        # Get response in a separate thread
        threading.Thread(target=self.get_response, args=(message,), daemon=True).start()
    
    def get_response(self, message):
        """Get a response from the selected model"""
        try:
            response = ""
            
            # Get current model
            model_name = self.model_var.get()
            model_config = self.models.get(model_name, {})
            model_type = model_config.get("type", "")
            
            # Get response based on model type
            if model_type == "openai" and self.openai_client:
                # Use OpenAI API
                system_message = model_config.get("system_message", "You are an empathetic assistant that provides supportive and compassionate responses.")
                
                api_response = self.openai_client.chat.completions.create(
                    model=model_config.get("id", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                response = api_response.choices[0].message.content
            
            elif model_type == "simple" and self.simple_model:
                # Use simple model
                response = self.simple_model.generate_response(message)
            
            else:
                # Fallback response
                response = "I'm sorry, but I'm having trouble connecting to my response system. Please check the application configuration."
            
            # Evaluate response
            metrics = self.evaluator.count_empathy_indicators(response)
            
            # Add response to chat (in main thread)
            self.root.after(0, lambda: self.add_message(response, is_user=False, metrics=metrics))
            
            # Save chat history
            self.save_chat_history()
        
        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.root.after(0, lambda: self.add_message(error_message, is_user=False))
    
    def on_enter_pressed(self, event):
        """Handle Enter key press"""
        # Send message on Enter without Shift
        if not event.state & 0x1:  # Check if Shift is not pressed
            self.send_message()
            return "break"  # Prevent default behavior
    
    def on_model_change(self, model_name):
        """Handle model change"""
        self.current_model = model_name
    
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        theme = self.theme_var.get()
        ctk.set_appearance_mode(theme)
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create tabs
        tabview = ctk.CTkTabview(settings_window)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # API tab
        api_tab = tabview.add("API Settings")
        api_tab.grid_columnconfigure(1, weight=1)
        
        # OpenAI API key
        api_key_label = ctk.CTkLabel(
            api_tab, 
            text="OpenAI API Key:", 
            font=("Arial", 12)
        )
        api_key_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        api_key_var = ctk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))
        api_key_entry = ctk.CTkEntry(
            api_tab, 
            width=300, 
            show="*", 
            textvariable=api_key_var
        )
        api_key_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # System message
        system_msg_label = ctk.CTkLabel(
            api_tab, 
            text="System Message:", 
            font=("Arial", 12)
        )
        system_msg_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        system_msg = ""
        if self.current_model and self.current_model in self.models:
            system_msg = self.models[self.current_model].get("system_message", "")
        
        system_msg_text = ctk.CTkTextbox(
            api_tab, 
            width=300, 
            height=100
        )
        system_msg_text.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        system_msg_text.insert("0.0", system_msg)
        
        # Save button
        def save_api_settings():
            # Save API key to .env file
            api_key = api_key_var.get().strip()
            if api_key:
                with open(".env", "w") as f:
                    f.write(f"OPENAI_API_KEY={api_key}\n")
                
                # Update client
                if OPENAI_AVAILABLE:
                    self.openai_client = OpenAI(api_key=api_key)
                    
                    # Update models
                    if "OpenAI Fine-tuned" not in self.models and self.openai_client:
                        self.models["OpenAI Fine-tuned"] = {
                            "type": "openai",
                            "id": "ft:gpt-3.5-turbo-0125:valis::BYfKr10K",
                            "system_message": "You are an empathetic assistant that provides supportive and compassionate responses."
                        }
                    
                    if "OpenAI Base" not in self.models and self.openai_client:
                        self.models["OpenAI Base"] = {
                            "type": "openai",
                            "id": "gpt-3.5-turbo",
                            "system_message": "You are an empathetic assistant that provides supportive and compassionate responses."
                        }
            
            # Update system message
            if self.current_model and self.current_model in self.models:
                new_system_msg = system_msg_text.get("0.0", "end").strip()
                if new_system_msg:
                    self.models[self.current_model]["system_message"] = new_system_msg
            
            settings_window.destroy()
            messagebox.showinfo("Settings", "Settings saved successfully!")
        
        save_button = ctk.CTkButton(
            api_tab, 
            text="Save", 
            command=save_api_settings
        )
        save_button.grid(row=2, column=1, padx=10, pady=20, sticky="e")
        
        # History tab
        history_tab = tabview.add("Chat History")
        history_tab.grid_columnconfigure(0, weight=1)
        
        # Save history button
        def save_history():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.chat_history, f, indent=2)
                messagebox.showinfo("History", "Chat history saved successfully!")
        
        save_history_button = ctk.CTkButton(
            history_tab, 
            text="Save History", 
            command=save_history
        )
        save_history_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Load history button
        def load_history():
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        history = json.load(f)
                    
                    # Clear current chat
                    for widget in self.chat_frame.winfo_children():
                        widget.destroy()
                    
                    # Load history
                    self.chat_history = []
                    for message in history:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        metrics = message.get("metrics")
                        
                        self.add_message(
                            content, 
                            is_user=(role == "user"), 
                            metrics=metrics
                        )
                    
                    messagebox.showinfo("History", "Chat history loaded successfully!")
                
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load history: {str(e)}")
        
        load_history_button = ctk.CTkButton(
            history_tab, 
            text="Load History", 
            command=load_history
        )
        load_history_button.grid(row=1, column=0, padx=10, pady=10)
        
        # Clear history button
        def clear_history():
            if messagebox.askyesno("Clear History", "Are you sure you want to clear the chat history?"):
                # Clear current chat
                for widget in self.chat_frame.winfo_children():
                    widget.destroy()
                
                # Reset history
                self.chat_history = []
                
                # Add welcome message
                self.add_message("Hello! I'm here to listen and support you. How are you feeling today?", is_user=False)
                
                messagebox.showinfo("History", "Chat history cleared successfully!")
        
        clear_history_button = ctk.CTkButton(
            history_tab, 
            text="Clear History", 
            command=clear_history
        )
        clear_history_button.grid(row=2, column=0, padx=10, pady=10)
        
        # About tab
        about_tab = tabview.add("About")
        
        about_text = """
        Empathy Chatbot UI
        
        A modern, user-friendly chat interface for interacting with empathy models.
        
        Features:
        - Beautiful, responsive UI with dark/light mode
        - Support for multiple models (OpenAI, Simple, Local)
        - Chat history and session management
        - Response analysis and feedback
        
        Version: 1.0
        """
        
        about_label = ctk.CTkLabel(
            about_tab, 
            text=about_text, 
            font=("Arial", 12),
            justify="left"
        )
        about_label.pack(padx=20, pady=20)
        
        # GitHub link
        def open_github():
            webbrowser.open("https://github.com/yourusername/empathy-chatbot")
        
        github_button = ctk.CTkButton(
            about_tab, 
            text="View on GitHub", 
            command=open_github
        )
        github_button.pack(pady=10)
    
    def save_chat_history(self):
        """Save chat history to file"""
        # Create directory if it doesn't exist
        os.makedirs("chat_history", exist_ok=True)
        
        # Save to file
        with open("chat_history/latest_chat.json", "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=2)

def main():
    # Install customtkinter if not available
    try:
        import customtkinter
    except ImportError:
        import subprocess
        import sys
        
        print("Installing customtkinter...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
        
        # Try import again
        import customtkinter
    
    # Create root window
    root = ctk.CTk()
    app = EmpathyChatbotUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
