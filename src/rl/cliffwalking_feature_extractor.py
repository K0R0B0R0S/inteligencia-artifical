from feature_extractor import FeatureExtractor
import numpy as np

class Actions:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class Layout:
    goal = [3, 11]
    start = [3, 0]
    cliff = [[3, i] for i in range(1, 10)]
    column_min = 0
    column_max = 11
    row_min = 0
    row_max = 3
    
class CliffWalkingFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:   [1,0,0,0], 
        Actions.DOWN:   [0,1,0,0], 
        Actions.RIGHT:  [0,0,1,0], 
        Actions.UP:     [0,0,0,1]
    }
    
    def __init__(self, env):
        '''
        Initializes the CliffWalkingFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)
        self.features_list.append(self.f3)

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
        Checks if the state is terminal (either goal or cliff).
        '''
        column, row = state % 12, state // 12
        is_goal = [row, column] == Layout.goal
        is_cliff = [row, column] in Layout.cliff
        return is_goal or is_cliff
    
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
    
    def _get_agent_position(self, state):
        '''
        Gets agent's position based on a 4x12 grid.
        '''
        return [state // 12, state % 12]
    
    def f0(self, state, action):
        '''
        This is just the bias term.
        '''
        return 1.0
    
    def f1(self, state, action):
        '''
        Calcula a distância de Manhattan do agente ao objetivo.
        '''
        agent_position = self._get_agent_position(state)
        if agent_position in Layout.cliff:
            return 0
        distance_to_goal = self.__manhattanDistance(agent_position, Layout.goal)
        return 1.0 / (distance_to_goal + 1)
    
    def f2(self, state, action):
        '''
        Calculo a distância de Manhattan do agente ao penhasco.
        '''
        agent_position = self._get_agent_position(state)
        distance_to_cliff = min([self.__manhattanDistance(agent_position, cliff) for cliff in Layout.cliff])
        return distance_to_cliff + 1    
    
    def f3(self, state, action):
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
        
        return int(any(pos in Layout.cliff for pos in adjacent_positions))

    def f4(self, state, action):
        '''
        Não deixa o agente ir muito longe do objetivo no eixo y.
        '''
        agent_position = self._get_agent_position(state)
        distance_y_axis = abs(agent_position[0] - Layout.goal[0])
        return 1.0 / (distance_y_axis + 1)
    
    