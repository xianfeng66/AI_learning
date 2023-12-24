
from maze_env import Maze
from Sarsa.RL_train import SarsaTable
from Sarsa.RL_train import SarsaLambdaTable

def update():
    for episode in range(100):  #跑了多少个回合
        #获取初始位置坐标信息（1，1）
        observation = env.reset()
        # RL 选择基于观测的下一个行为

        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # # RL  选择基于观测的下一个行为
            # action = RL.choose_action(str(observation))
            #RL 根据采取的行为获取下一个观测和当前的奖励
            observation_, reward, done = env.step(action)
            #RL 基于刚刚得到的观测选择下一个行为
            action_ = RL.choose_action(str(observation))

            # RL 更新Q-table
            RL.learn(str(observation),action,reward,str(observation_),action_)

            observation = observation_
            action = action_

            if done:
                break

print('end of game')
env.destory()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    #RL = SaraLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

# # 导入迷宫环境模块
# from maze_env import Maze
# # 导入基于表格的强化学习算法模块
# from Sarsa.RL_train import SarsaTable
# # 导入另一个基于表格的强化学习算法模块（与Sarsa略有不同）
# from Sarsa.RL_train import SarsaLambdaTable
#
#
# # 定义一个名为update的函数，用于更新强化学习模型
# def update():
#     # 循环100次，表示进行了100个回合的游戏
#     for episode in range(100):  # 跑了多少个回合
#         # 获取环境的初始状态，这里假设env.reset()返回的是一个代表初始状态的整数
#         observation = env.reset()  # 获取初始位置坐标信息
#         # 重置强化学习状态，为新的回合做准备
#         RL.reset()  # 重置RL状态
#         # 使用强化学习算法选择一个行动，行动取决于当前状态（observation）
#         action = RL.choose_action(observation)  # RL 选择基于观测的下一个行为
#
#         # 进入循环，直到游戏结束
#         while True:
#             # 显示当前环境的状态，给玩家一个可视化的提示
#             env.render()  # 刷新环境
#             # 通过环境的行动函数env.step(action)，获取下一个状态、奖励和游戏是否结束的信息
#             observation_, reward, done = env.step(action)  # RL 根据采取的行为获取下一个观测和当前的奖励
#             # 如果游戏结束，跳出循环
#             if done:  # 如果回合结束
#                 break
#                 # 使用强化学习算法选择下一个行动，行动取决于下一个状态（observation_）
#             action_ = RL.choose_action(observation_)  # RL 基于刚刚得到的观测选择下一个行为
#             # 使用强化学习算法更新Q-table，增加新的观测、行动、奖励和下一个观测、下一个行动的信息
#             RL.learn(observation, action, reward, observation_, action_)  # RL 更新Q-table
#             # 更新当前状态为下一个状态，为下一步做准备
#             observation = observation_  # 更新观测
#             action = action_  # 更新行动
#
#     print('end of game')  # 游戏结束后打印消息，告知玩家游戏结束
#     env.destroy()  # 销毁环境对象，释放内存空间
#
#
# if __name__ == '__main__':  # 只有当该文件被直接运行时，下面的代码才会执行
#     env = Maze()  # 创建一个迷宫环境对象
#     RL = SarsaTable(actions=list(range(env.n_actions)))  # 初始化一个Sarsa表格强化学习对象，假设env.n_actions表示环境的所有可能行动数量
#     # 以下行代码与上面注释的代码功能相同，只是用了不同的强化学习算法（SarsaLambdaTable），你可能需要根据实际需要选择合适的算法
#     # RL = SaraLambdaTable(actions=list(range(env.n_actions)))  # 初始化一个Sarsa Lambda表格强化学习对象，假设env.n_actions表示环境的所有可能行动数量
#     env.after(100, update)  # 当游戏进行100个回合后，执行update函数，更新强化学习模型
#     env.mainloop()  # 开始主循环，运行环境，直到游戏结束为止
#
#
