from feature_extractor import FeatureExtractor
import numpy as np

class Actions:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class Layout:
    goal = [3, 3]
    start = [0, 0]
    hole = [[3,0], [1,1], [1,3], [2,3]]
    column_min = 0
    column_max = 3
    row_min = 0
    row_max = 3
    
class FrozenLakeFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:   [1,0,0,0], 
        Actions.DOWN:   [0,1,0,0], 
        Actions.RIGHT:  [0,0,1,0], 
        Actions.UP:     [0,0,0,1]
    }
    
    def __init__(self, env):
        '''
        Initializes the FrozenLakeFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)
        self.features_list.append(self.f3)  
        self.features_list.append(self.f4)  
        self.features_list.append(self.f_penalty_for_loop)
        self.features_list.append(self.f_is_next_position_hole)


    def get_num_features(self):
        '''
        Returns the number of features extracted by the feature extractor.
        '''
        return len(self.features_list) + self.get_num_actions()
    
    def get_num_actions(self):
        '''
        Returns the number of actions available in the environment.
        '''
        return len(self.get_actions())
    
    def get_action_one_hot_encoded(self, action):
        '''
        Returns the one-hot encoded representation of an action.
        '''
        return self.__actions_one_hot_encoding[action]
    
    def is_terminal_state(self, state):
        '''
        Verifica se o estado é terminal.
        Um estado terminal só é considerado um sucesso se o agente alcançar o objetivo (não cair no buraco).
        '''
        goal_state = Layout.goal[0] * 4 + Layout.goal[1]
        hole_states = [hole[0] * 4 + hole[1] for hole in Layout.hole]

        if state == goal_state:
            return True
        
        if state in hole_states:
            return True
        
        return False
    
    def get_actions(self):
        '''
        Returns a list of available actions in the environment.
        '''
        return [Actions.LEFT, Actions.DOWN, Actions.RIGHT, Actions.UP]
    
    def get_features(self, state, action):
        '''
        Takes a state and an action as input and returns the feature vector for that state-action pair. 
        It calls the feature extraction methods and constructs the feature vector.
        '''
        feature_vector = np.zeros(len(self.features_list))
        
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)
        
        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])
        
        return feature_vector
    
    @staticmethod
    def __manhattanDistance(xy1, xy2):
        '''
        Computes the Manhattan distance between two points.
        '''
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    
    @staticmethod
    def __euclideanDistance(xy1, xy2):
        '''
        Computes the Euclidean distance between two points.
        '''
        return np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)
    
    def _get_agent_position(self, state):
        '''
        Gets agent's position based on a 4x4 grid.
        '''
        return [state // 4, state % 4]
    
    def _get_next_position(self, position, action):
        '''
        Returns the agent's next position based on the current position and action.
        '''
        row, col = position
        if action == Actions.LEFT:
            col = max(Layout.column_min, col - 1)
        elif action == Actions.DOWN:
            row = min(Layout.row_max, row + 1)
        elif action == Actions.RIGHT:
            col = min(Layout.column_max, col + 1)
        elif action == Actions.UP:
            row = max(Layout.row_min, row - 1)
        return [row, col]

    def f0(self, state, action):
        '''
        Bias term.
        '''
        return 1.0

    def f1(self, state, action):
        '''
        Distância normalizada ao objetivo.
        '''
        agent_position = self._get_agent_position(state)
        goal_position = Layout.goal
        distance = self.__euclideanDistance(agent_position, goal_position)
        return 1 / (1 + distance)
    
    def f2(self, state, action):
        '''
        Distância normalizada ao buraco mais próximo.
        '''
        agent_position = self._get_agent_position(state)
        hole_positions = Layout.hole
        distance = min([self.__euclideanDistance(agent_position, hole) for hole in hole_positions])
        return 1 / (1 + distance)
    
    def f3(self, state, action):
        '''
        Verifica se o agente bateu em uma parede.
        '''
        agent_position = self._get_agent_position(state)
        row, column = agent_position
        bump = (
            (action == Actions.LEFT and column == Layout.column_min) or
            (action == Actions.RIGHT and column == Layout.column_max) or
            (action == Actions.UP and row == Layout.row_min) or
            (action == Actions.DOWN and row == Layout.row_max)
        )
        return int(bump)
    
    def f4(self, state, action):
        '''
        Verifica se o agente está adjacente a um buraco.
        '''
        agent_position = self._get_agent_position(state)
        adjacent_positions = [
            (agent_position[0] - 1, agent_position[1]),
            (agent_position[0] + 1, agent_position[1]),
            (agent_position[0], agent_position[1] - 1),
            (agent_position[0], agent_position[1] + 1)
        ]
        
        return int(any(pos in Layout.hole for pos in adjacent_positions))
    
    def f_penalty_for_loop(self, state, action):
        '''
        Penaliza o agente se ele estiver se movendo na direção oposta ao objetivo.
        Esta é a feature para evitar que o agente tente "dar a volta" no mapa.
        '''
        agent_position = self._get_agent_position(state)
        goal_position = Layout.goal

        direction_to_goal = [goal_position[0] - agent_position[0], goal_position[1] - agent_position[1]]

        action_directions = {
            Actions.LEFT:  [-1, 0],
            Actions.DOWN:  [0, -1],
            Actions.RIGHT: [1, 0],
            Actions.UP:    [0, 1]
        }
        
        action_direction = action_directions[action]
        dot_product = direction_to_goal[0] * action_direction[0] + direction_to_goal[1] * action_direction[1]
        
        if dot_product < 0:
            return -1.0
        
        return 0.0
    
    def f_is_next_position_hole(self, state, action):
        '''
        Verifica se a próxima posição do agente é um buraco.
        Retorna 0 se a próxima posição for um buraco, caso contrário retorna 1.
        '''
        agent_position = self._get_agent_position(state)
        next_position = self._get_next_position(agent_position, action)

        if next_position in Layout.hole:
            return 0
        
        return 1