from tkinter import Label, Tk
import time 

app_window = Tk()
app_window.title("Dijital Saat")
app_window.geometry("400x200")
app_window.resizable(0,0)
app_window.configure(bg="black")

text_font = ("Boulder", 36, 'bold')
background = "black"
foreground = "white"
border_width = 20

#saat etiketi
label = Label(app_window, font=text_font, bg=background, fg=foreground)
label.grid(row=0, column=1, padx=border_width, pady=10)

#tarih etiketi
date_label = Label(app_window, font=text_font, bg=background, fg=foreground)
date_label.grid(row=1, column=1, padx=border_width, pady=10)


def digital_clock():
    time_live = time.strftime("%H:%M:%S")
    label.config(text=time_live)
    date_info = time.strftime("%d %B %Y")
    date_label.config(text=date_info)
    label.after(200, digital_clock)

digital_clock()


app_window.mainloop()