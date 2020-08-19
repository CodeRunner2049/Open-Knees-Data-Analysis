import tkinter as tk

def make_GUI ():
    root= tk.Tk()

    canvas1 = tk.Canvas(root, width = 500, height = 300)
    canvas1.pack()

    label1 = tk.Label(root, text="Enter the filepath of an Open Knee database: ")
    label1.config(font=('calibri', 14))
    canvas1.create_window(100, 100, window=label1)

    entry1 = tk.Entry (root) # create 1st entry box
    canvas1.create_window(300, 100, window=entry1)

    # New_Unemployment_Rate label and input box
    label2 = tk.Label(root, text=' Type Unemployment Rate: ')
    label1.config(font=('calibri', 14))
    canvas1.create_window(120, 120, window=label2)

    entry2 = tk.Entry (root) # create 2nd entry box
    canvas1.create_window(270, 120, window=entry2)

    def values():
        global New_Interest_Rate #our 1st input variable
        New_Interest_Rate = float(entry1.get())

        global New_Unemployment_Rate #our 2nd input variable
        New_Unemployment_Rate = float(entry2.get())

        Prediction_result  = ('Predicted Stock Index Price: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))
        label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
        canvas1.create_window(260, 280, window=label_Prediction)

    button1 = tk.Button (root, text='Predict Stock Index Price',command=values, bg='orange') # button to call the 'values' command above
    canvas1.create_window(270, 150, window=button1)

    #root.mainloop()
