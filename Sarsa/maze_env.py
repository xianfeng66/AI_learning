import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  #pixels
MAZE_H = 4 # grid height
MAZE_W = 4 # grid width


class Maze(tk.Tk,object):
    def __init__(self):
        super(Maze,self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H*UNIT,MAZE_W * ))
        self._build_maze()
    def _build_maze(self):
        self.canvas = tk.Canvas(self,bg='white',
                                height=MAZE_H*UNIT,
                                width=MAZE_W * UNIT)

        #create grids
        for c in range(0, MAZE_W*UNIT,UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        #create origin
        origin = np.array([20,20])

        #hell
        hell_center = origin + np.array([UNIT*2,UNIT])
        self.hell = self.canvas.create_rectangle(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15,hell_center[1] + 15,
            fill = 'black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] -15,oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill = 'red'
        )

        #create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15,origin[1] - 15,
            origin[0] + 15, origin[i] + 15,
            fill='red'
        )

        #pack all
        self.canvas.pack()


    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20,20])
        self.rect = self.canvas.create_rectangle(
            origin[0] -15, origin[1] - 15,
            origin[0] + 15,origin[1] + 15,
            fill = 'red'
        )
        #return observation
        return self.canvas.coords(self.rect)
    def step(self,action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0,0])
        if action == 0:   #up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  #down
            if s[1]< (MAZE_H - 1)*UNIT:
                base_action[1] += UNIT
        elif action == 2:  #right
            if s[0] < (MAZE_W - 1)*UNIT:
                base_action[0] += UNIT
        elif action == 3:   #left
             if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect,base_action[0],base_action[1])  #move

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward =1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell),self.canvas.coords(self.hell2)]:
            reward = -1
            done =True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done
    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


# # 导入Python的数值计算库numpy
# import numpy as np
#
# # 导入time库，它可以让我们使用时间相关的函数，例如获取当前时间或进行时间计算
# import time
#
# # 导入sys库，它提供对Python解释器使用或维护的一些变量的访问
# import sys
#
# # 检查当前Python的版本信息，如果版本是2.x，则导入Tkinter库为tk，否则为tkinter库
# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
#     import tkinter as tk
#
# # 定义一个常量UNIT，表示每个网格单元的像素大小，这里设定为40像素
# UNIT = 40  # pixels
#
# # 定义MAZE_H和MAZE_W两个变量，分别表示网格的高度和宽度，这里设定为4x4的网格
# MAZE_H = 4  # grid height
# MAZE_W = 4  # grid width
#
#
# # 定义一个名为Maze的类，继承自tkinter库的Tk类和object类，object类是所有Python类的基类
# class Maze(tk.Tk, object):
#     # 类的初始化方法，创建Maze类的实例时会自动调用
#     def __init__(self):
#         # 使用super函数调用父类的初始化方法，这样可以在初始化过程中设置一些父类初始化中定义的属性或方法
#         super(Maze, self).__init__()
#         # 定义一个名为action_space的属性，它是一个字符串列表，包含了所有可能的动作，例如'u'表示上，'d'表示下，'l'表示左，'r'表示右
#         self.action_space = ['u', 'd', 'l', 'r']
#         # 获取动作空间列表的长度，也就是可能的行为数量，并赋值给n_actions属性
#         self.n_actions = len(self.action_space)
#         # 使用self.title方法设置窗口的标题为'maze'
#         self.title('maze')
#         # 使用self.geometry方法设置窗口的大小为{'height': MAZE_H*UNIT, 'width': MAZE_W * UNIT}，其中MAZE_H和MAZE_W是之前定义的常量，UNIT是40像素
#         self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
#         # 调用_build_maze方法来构建迷宫
#         self._build_maze()
#
#         # _build_maze方法用于构建迷宫的图形界面
#
#     def _build_maze(self):
#         # 创建一个Canvas对象，它是在窗口中绘制图形的主要工具，参数包括self（指向Maze对象的引用）、bg（背景颜色）、height（高度）、width（宽度）等
#         self.canvas = tk.Canvas(self, bg='white',
#                                 height=MAZE_H * UNIT,
#                                 width=MAZE_W * UNIT)
#
#         # 通过循环在Canvas上绘制迷宫的格子线，其中c是循环变量，从0开始以UNIT为步长递增到MAZE_W*UNIT，每次循环都在Canvas上绘制一条从(c,0)到(c,MAZE_H*UNIT)的直线
#         for c in range(0, MAZE_W * UNIT, UNIT):
#             x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
#             self.canvas.create_line(x0, y0, x1, y1)
#
#             # 创建代表起始位置的圆形标记，"hell"可能是代表"起点"的意思，但这取决于具体的使用环境或意图
#         # 通过np库的数组加法将origin（一个二维坐标点）分别加上UNIT*2（二维坐标点平移），得到"hell"中心的位置坐标
#         hell_center = origin + np.array([UNIT * 2, UNIT])
#         # 在Canvas上创建一个矩形，"hell"可能代表"起点"的意思，但这取决于具体的使用环境或意图。矩形的位置和大小通过参数传递给create_rectangle方法确定，颜色被设置为黑色（fill参数）
#         self.hell = self.canvas.create_rectangle(
#             hell_center[0] - 15, hell_center[1] - 15,
#             hell_center[0] + 15, hell_center[1] + 15,
#             fill='black')
#
#         # 在Canvas上创建一个椭圆形标记，"oval"可能是代表"目标"的意思，但这取决于具体的使用环境或意图。位置和大小通过参数传递给create_oval方法确定，颜色被设置为红色（fill参数）
#         oval_center = origin + UNIT * 2
#         self.oval = self.canvas.create_oval(
#             oval_center[0] - 15, oval_center[1] - 15,
#             oval_center[0] + 15, oval_center[1] + 15,
#             fill='red'
#         )
#
#         # 在Canvas上创建一个红色矩形标记，"rect"可能是代表"障碍"或"终点"的意思，但这取决于具体的使用环境或意图。位置和大小通过参数传递给create_rectangle方法确定，颜色被设置为红色（fill参数）
#         self.rect = self.canvas.create_rectangle(
#             origin[0] - 15, origin[1] - 15,
#             origin
#         创建一个名为rect的属性，它是一个Canvas上的矩形对象，位置和大小通过参数传递给create_rectangle方法确定，颜色被设置为红色（fill参数）
#         self.rect = self.canvas.create_rectangle(
#             origin[0] - 15, origin[1] - 15,
#             origin[0] + 15, origin[i] + 15,
#             fill='red'
#         )
#
#         将Canvas添加到窗口中，这样我们就可以在窗口中看到它了
#         # pack方法是一种布局管理器，它把一个窗口或控件安排到另一个窗口或控件中，以占据尽可能小的空间
#         # pack all是把所有的控件都放到窗口中，并按照一定的顺序排列
#         self.canvas.pack()
#
#         # 定义一个名为reset的方法，该方法属于某个类的实例方法，具体的类名在此代码片段中未给出
#
#         def reset(self):
#             # 调用update方法，此方法可能用于更新或刷新canvas的显示
#             self.update()
#             # 使程序暂停0.5秒，这可能是为了给用户一些时间来查看矩形的位置变化
#             time.sleep(0.5)
#             # 从canvas中删除当前的矩形对象（self.rect），即删除旧的矩形
#             self.canvas.delete(self.rect)
#             # 创建一个名为origin的numpy数组，其值为[20,20]，表示矩形的初始位置
#             origin = np.array([20, 20])
#             # 在canvas中创建一个新的矩形对象，其左上角位置为(20-15, 20-15)，即(5,5)，右下角位置为(20+15, 20+15)，即(35,35)，填充颜色为红色
#             self.rect = self.canvas.create_rectangle(
#                 origin[0] - 15, origin[1] - 15,
#                 origin[0] + 15, origin[1] + 15,
#                 fill='red'
#             )
#             # 返回矩形的当前位置，即返回一个表示矩形位置的四元组或类似对象
#             # return observation
#             return self.canvas.coords(self.rect)
#
#         # 定义一个名为step的方法，该方法接受一个动作作为输入，并返回下一步的状态、奖励和完成标志
#         def step(self, action):
#             # 获取矩形的当前位置
#             s = self.canvas.coords(self.rect)
#             # 初始化基础动作，初始为(0,0)，表示没有移动
#             base_action = np.array([0, 0])
#             # 如果动作是向上（0）
#             if action == 0:  # up
#                 # 如果矩形上方还有空间（s[1] > UNIT）
#                 if s[1] > UNIT:
#                     # 则将基础动作的y坐标减去一个单位
#                     base_action[1] -= UNIT
#                     # 如果动作是向下（1）
#             elif action == 1:  # down
#                 # 如果矩形下方还有空间（s[1]< (MAZE_H - 1)*UNIT）
#                 if s[1] < (MAZE_H - 1) * UNIT:
#                     # 则将基础动作的y坐标加上一个单位
#                     base_action[1] += UNIT
#                     # 如果动作是向右（2）
#             elif action == 2:  # right
#                 # 如果矩形右边还有空间（s[0] < (MAZE_W - 1)*UNIT）
#                 if s[0] < (MAZE_W - 1) * UNIT:
#                     # 则将基础动作的x坐标加上一个单位
#                     base_action[0] += UNIT
#                     # 如果动作是向左（3）
#             elif action == 3:  # left
#                 # 如果矩形左边还有空间（s[0] > UNIT）
#                 if s[0] > UNIT:
#                     # 则将基础动作的x坐标减去一个单位
#                     base_action[0] -= UNIT
#
#                     # 使用基础动作移动矩形
#             self.canvas.move(self.rect, base_action[0], base_action[1])  # move
#
#             # 获取矩形的新位置
#             s_ = self.canvas.coords(self.rect)  # next state
#
#             # 定义奖励函数
#             # 如果新位置与目标位置相同，则奖励为1，游戏结束并返回'terminal'状态
#             if s_ == self.canvas.coords(self.oval):
#                 reward = 1
#                 done = True
#                 s_ = 'terminal'
#                 # 如果新位置与"hell"或"hell2"相同，则奖励为-1，游戏结束并返回'terminal'状态
#             elif s_ in [self.canvas.coords(self.hell), self.canvas.coords(self.hell2)]:
#                 reward = -1
#                 done = True
#                 s_ = 'terminal'
#             else:
#                 reward = 0
#                 done = False
#
#             return s_, reward, done
#
#











































