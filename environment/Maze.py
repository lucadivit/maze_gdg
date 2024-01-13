from pyamaze import maze, COLOR, agent
import random
from copy import deepcopy


class Maze:
    # NB: Robot start position is always in (MAZE_HEIGHT, MAZE_WIDTH) and end position is always in (1, 1) (in pyamaze lib)
    DICT_POSITION_ENC_FROM_INT = {1: "UP", 2: "LEFT", 3: "DOWN", 4: "RIGHT"}
    DICT_POSITION_ENC_FROM_STRING = {"UP": 1, "LEFT": 2, "DOWN": 3, "RIGHT": 4}
    END_POS = (1, 1)

    def __init__(self, height, width):
        self.__height = height
        self.__width = width

    def create_maze(self, seed: int = None) -> (maze, agent, dict):
        if seed is not None: print(f"Set Maze Seed: {seed}")
        random.seed(seed)
        m = maze(self.__height, self.__width)
        m.CreateMaze(loopPercent=30, theme=COLOR.light, pattern='v')
        a = agent(m, filled=False, footprints=True)
        # m.path contains the shortest path to exit
        # Path is a dictionary like {(old_pos: new_pos)} with old_pos = (h, w)
        # Ex {(6, 5): (5, 5), (6, 6): (6, 5)} --> robot will start from (6, 6) and with next step it will go in (6, 5), and so on...
        solution_path = m.path
        random.seed(None)
        return m, a, solution_path

    def convert_path_to_int_list(self, solution_path: dict) -> list:
        position_encode = deepcopy(self.DICT_POSITION_ENC_FROM_STRING)
        print(f"Solution path to encode is {solution_path}")
        step_pos_h = self.__height
        step_pos_w = self.__width
        converted_path_str = []
        is_completed = False
        while not is_completed:
            current_pos = (step_pos_h, step_pos_w)
            next_pos = solution_path.get(current_pos)
            h_current = current_pos[0]
            w_current = current_pos[1]
            h_next = next_pos[0]
            w_next = next_pos[1]
            if h_current == h_next and w_current == w_next:
                # This situation is not possible
                raise Exception("An error occurred during solution path conversion")
            elif h_current == h_next and w_current != w_next:
                if w_current == w_next + 1:
                    movement = "LEFT"
                elif w_current == w_next - 1:
                    movement = "RIGHT"
                else:
                    # This situation is not possible
                    raise Exception("An error occurred during solution path conversion")
            elif h_current != h_next and w_current == w_next:
                if h_current == h_next + 1:
                    movement = "UP"
                elif h_current == h_next - 1:
                    movement = "DOWN"
                else:
                    # This situation is not possible
                    raise Exception("An error occurred during solution path conversion")
            elif h_current != h_next and w_current != w_next:
                # This situation is not possible
                raise Exception("An error occurred during solution path conversion")
            else:
                # This situation is not possible
                raise Exception("An error occurred during solution path conversion")
            print(
                f"Current position is {current_pos}, end position {next_pos} hence action is {movement} ({position_encode.get(movement)})")
            converted_path_str.append(movement)
            step_pos_h = next_pos[0]
            step_pos_w = next_pos[1]
            if next_pos == self.END_POS:
                is_completed = True
        converted_path = [position_encode.get(pos) for pos in converted_path_str]
        print(f"Converted path is: {converted_path_str} --> {converted_path}")
        return converted_path

    def convert_path_to_tuple_list(self, solution_path_int: list) -> dict:
        solution_path = dict()
        converted_path = []
        step_pos_h = self.__height
        step_pos_w = self.__width
        start_pos = (step_pos_h, step_pos_w)
        for element in solution_path_int:
            action = self.DICT_POSITION_ENC_FROM_INT.get(element)
            if action == "UP":
                step_pos_h -= 1
            elif action == "LEFT":
                step_pos_w -= 1
            elif action == "DOWN":
                step_pos_h += 1
            elif action == "RIGHT":
                step_pos_w += 1
            else:
                # This situation is not possible
                raise Exception(f"An error occurred during chromosome to path conversion. Element is {element}")
            new_pos = (step_pos_h, step_pos_w)
            converted_path.insert(0, [start_pos, new_pos])
            start_pos = new_pos
        for element in converted_path:
            pos = element[0]
            new_pos = element[1]
            solution_path[pos] = new_pos
        print(f"Converted solution is: {solution_path}")
        return solution_path
