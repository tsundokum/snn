#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *
import tkMessageBox
import tkFont
from createpropositions import read_all_propositions
import random

BUTTON_NEXT_TEXT = 'Дальше'
MAIN_WINDOW_TITLE = " "
LABEL_DEFAULT_TEXT = " "
BUTTON_START_TEXT = "Начинаем!"
FINISH_MESSAGE = "Эксперимент окончен. Спасибо!"

random.seed(113)

proposition_words_file = 'resources/RogersMcClelland08_ru.yaml'
output_file = 'resources/test-distances.csv'
SCALER_DEFAULT_VALUE = 0


def get_list_from_file(file_name):
    rc = []
    with open(file_name) as f:
        for line in f:
            line_item = line.strip()
            if not line_item:
                continue
            rc.append(line_item)
    return rc


def generate_and_shuffle_propositions_generator():
    props = read_all_propositions(proposition_words_file)
    random.shuffle(props)

    for i, prop in enumerate(props):
        yield prop
        print i + 1


def proposition_text_view(proposition_as_tuple):
    words_to_show = filter(lambda x: '$' not in x, proposition_as_tuple)
    return ' '.join(words_to_show)


def write_out_answer(proposition, scale_value):
    with open(output_file, "a") as f:
        write_line = u','.join(proposition + (unicode(scale_value),)) + u'\n'
        print write_line,
        f.write(write_line.encode('utf8'))


class ScalerApp:
    def __init__(self, master, props_iterator):
        self.props = props_iterator
        self.current_proposition = None

        self.master = master
        self.master.title(MAIN_WINDOW_TITLE)
        self.master.minsize(320, 0)

        frame = Frame(self.master)
        frame.pack()

        self.label_var = StringVar(value=LABEL_DEFAULT_TEXT)
        self.label = Label(frame, textvariable=self.label_var, font=tkFont.Font(size=14))
        self.label.grid(row=0, column=1, columnspan=3)

        self.scale_var = IntVar(value=SCALER_DEFAULT_VALUE)

        self.scale = Scale(frame, from_=0, to=7, orient="horizontal", variable=self.scale_var, state=DISABLED, length=300)
        self.scale.grid(row=1, column=1, columnspan=3)

        self.button = Button(frame, text=BUTTON_START_TEXT, command=self.next_scale)
        self.button.grid(row=2, column=2)

    def next_scale(self):
        try:
            if self.current_proposition:
                write_out_answer(self.current_proposition, self.scale_var.get())
            self.current_proposition = self.props.next()
            self.label_var.set(proposition_text_view(self.current_proposition))
            self.scale_var.set(SCALER_DEFAULT_VALUE)
            self.button['text'] = BUTTON_NEXT_TEXT
            self.scale['state'] = NORMAL
        except StopIteration:
            tkMessageBox.showinfo("", FINISH_MESSAGE)
            self.master.destroy()


def center_window(tk_root):
    tk_root.update_idletasks()
    width = tk_root.winfo_width()
    height = tk_root.winfo_height()
    x = (tk_root.winfo_screenwidth() / 2) - (width / 2)
    y = (tk_root.winfo_screenheight() / 2) - (height / 2)
    tk_root.geometry('{0}x{1}+{2}+{3}'.format(width, height, x, y))


if __name__ == "__main__":
    root = Tk()
    app = ScalerApp(root, generate_and_shuffle_propositions_generator())
    center_window(root)
    root.mainloop()