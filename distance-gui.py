#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *
from rope.base.builtins import get_list


def get_list_from_file(file_name):
    rc = []
    with open(file_name) as f:
        for line in f:
            line_item = line.strip()
            if not line_item:
                continue
            rc.append(line_item)
    return rc


def generate_stimuli():
    attributes = get_list_from_file('resources/attributes.txt')
    items = get_list_from_file('resources/items.txt')
    relations = get_list_from_file('resources/items.txt')

    props = []
    for attr in attributes:
        for item in items:
            for relations in relations:
                props.append((item, relations, attr))
    return len(props)



class ScalerApp:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.scale_var = IntVar()
        self.scale = Scale(frame, from_=0, to=7, orient="horizontal", variable=self.scale_var)
        self.scale.pack(anchor=CENTER)

        self.button = Button(frame, text="Ok!", command=self.next_scale)
        self.button.pack(side=RIGHT)

    def next_scale(self):
        print "new value is", self.scale_var.get()




if __name__ == "__main__":
    print generate_stimuli()
    root = Tk()
    app = ScalerApp(root)
    root.mainloop()