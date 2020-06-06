import tkinter as tk
window = tk.Tk()
window.title('My first TK program!')
window.geometry('400x400')
# try1
# def click_b1():
#     global on_click1
#     if on_click1 == False:
#         on_click1 = True
#         var1.set("you click me")
#     else:
#         on_click1 =False
#         var1.set('')
#
# # 字符串变量
# var1 = tk.StringVar()
# # 监测点击次数
# on_click1 = False
#
# label1 = tk.Label(window, textvariable = var1, bg='red',width = 15, height =2)
# label1.pack()
# b1 = tk.Button(window, text = 'button 1', bg = 'green', width = 12,height = 2, command = click_b1)
# b1.pack()

# try2
e = tk.Entry(window,show=None)# 输入密码show = '*'
e.pack()
t = tk.Text(window, height=2)
# 用e.get()即可获取entry的内容
def insert_point():
    var = e.get()
    t.insert('insert',var)
def insert_end():
    var = e.get()
    t.insert('end',var)
    # 插入到指定位置，1.0表示第一行第一个位置，1.1表示第一行第二个位置
    # t.insert(1.0,var)

b1 = tk.Button(window,text = 'insert point', width = 12,height = 2, command = insert_point)
b2 = tk.Button(window,text = 'insert end', width = 12,height = 2, command = insert_end)
b1.pack()
b2.pack()
t.pack()






window.mainloop()
