import threading
from maze import *
import sys




def main():
    size_list = [5,10,30,60,100,200,300,500]
    for i in size_list:
        maze = Maze(i,i,recursion_limit=300000)
        with open('../layouts/maze_gen_'+str(i)+'.lay', 'w') as f:
            f.write(maze.text_maze())

threading.stack_size(67108864) # 64MB stack
thread = threading.Thread(target=main)
thread.start()