import math
from simpleai.search import SearchProblem, astar
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import time

# Define cost of moving around the map
cost_regular = 1.0
cost_diagonal = 1.7

# Create the cost dictionary
COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
    "up left": cost_diagonal,
    "up right": cost_diagonal,
    "down left": cost_diagonal,
    "down right": cost_diagonal,
}

# Define the map
MAP = """
                                          
 ######################################## 
 #    #         #             #   #     # 
 #   ####      ##    ###          #  #  # 
 #         #       ###    #   #   ####  # 
 #         #              #   #         # 
 # ####    ########       #   #     ### # 
 #    #    #              #        ##   # 
 #    # #     #####  ######   #    #    # 
 #      #   ###   #           #         # 
 #      #     #   #    ##  ####   ####  # 
 #     #####            #     #    #    # 
 #  #           #       #     ###  #    # 
 #######    #   ##      #          #  ### 
 #     #   ##    #         #####        # 
 #     #   #     ########          ##   # 
 #  #      #            #   #      #    #  
 #   #     #  #      #      #           # 
 ######################################## 
"""

# Convert map to a list
MAP = [list(x) for x in MAP.split("\n") if x]
M = 19
N = 41
W = 25
mau_den = np.zeros((W, W, 3), np.uint8) + (np.uint8(0), np.uint8(0), np.uint8(0))
mau_trang = np.zeros((W, W, 3), np.uint8) + (np.uint8(255), np.uint8(255), np.uint8(255))
image = np.ones((M * W, N * W, 3), np.uint8) * 255

# Vẽ tọa độ x
for x in range(N + 1):
    cv2.putText(image, str(x), (x * W + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Vẽ tọa độ y
for y in range(M + 1):
    cv2.putText(image, str(y), (5, y * W + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

for x in range(1, M):
    for y in range(1, N):
        if MAP[x][y] == '#':
            image[x * W:(x + 1) * W, y * W:(y + 1) * W] = mau_den
        elif MAP[x][y] == ' ':
            image[x * W:(x + 1) * W, y * W:(y + 1) * W] = mau_trang

color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_coverted)


# Class containing the methods to solve the maze
class MazeSolver(SearchProblem):
    # Initialize the class 
    def __init__(self, board):
        self.board = board
        self.exits = []

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.exits.append((x, y))

        super(MazeSolver, self).__init__(initial_state=self.initial)

    # Define the method that takes actions
    # to arrive at the solution
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)

        return actions

    # Update the state based on the action
    def result(self, state, action):
        x, y = state

        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1

        new_state = (x, y)

        return new_state

    # Check if we have reached the goal
    def is_goal(self, state):
        return state in self.exits

    # Compute the cost of taking an action
    def cost(self, state, action, state2):
        return COSTS[action]

    # Heuristic that we use to arrive at the solution
    def heuristic(self, state):
        x, y = state
        return min(math.sqrt((x - ex) ** 2 + (y - ey) ** 2) for ex, ey in self.exits)


st.title("Maze Solver with A* Algorithm")

st.sidebar.title("Control Panel")
start_point = st.sidebar.text_input("Start Point (x,y)", "1,1")
end_point = st.sidebar.text_input("End Point (x,y)", "39,17")

start = tuple(map(int, start_point.split(',')))
end = tuple(map(int, end_point.split(',')))

# Modify the map with start and end points
MAP[start[1]][start[0]] = 'o'
MAP[end[1]][end[0]] = 'x'

# Highlight start and end points on the image
image[start[1] * W:(start[1] + 1) * W, start[0] * W:(start[0] + 1) * W] = (255, 0, 0)  # Red color for start
image[end[1] * W:(end[1] + 1) * W, end[0] * W:(end[0] + 1) * W] = (0, 0, 255)  # Blue color for end

problem = MazeSolver(MAP)

if st.sidebar.button('Solve Maze'):
    result = astar(problem, graph_search=True)
    path = [x[1] for x in result.path()]

    placeholder = st.empty()

    for i in range(1, len(path)):
        x = path[i][0]
        y = path[i][1]
        image[y * W:(y + 1) * W, x * W:(x + 1) * W] = (0, 255, 0)
        # Convert image for display
        img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_display)
        placeholder.image(pil_img)
        time.sleep(0.1)
else:
    st.image(pil_image)
