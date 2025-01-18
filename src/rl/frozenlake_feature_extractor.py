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
    hole = [[1,1], [1,3], [2,3], [3,1]]
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
        Initializes the CliffWalkingFeatureExtractor object. 
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)
        self.features_list.append(self.f3)
        self.features_list.append(self.f4)
        self.features_list.append(self.f5)

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
        column, row = state % 4, state // 4
        is_goal = [row, column] == Layout.goal
        is_hole = [row, column] in Layout.hole
        return is_goal or is_hole
    
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
        This is just the bias term.
        '''
        return 1.0
    
    def f1(self, state, action):
        '''
        Calcula a distância de Manhattan do agente ao objetivo.	
        '''
        agent_position = self._get_agent_position(state)
        if agent_position in Layout.hole:
            return 0
        distance_to_goal = self.__manhattanDistance(agent_position, Layout.goal)
        return 1.0 / (distance_to_goal + 1)
    
    def f2(self, state, action):
        '''
        CCalcula a distância de Manhattan do agente ao buraco mais próximo.
        '''
        agent_position = self._get_agent_position(state)
        distance_to_hole = min(self.__manhattanDistance(agent_position, hole) for hole in Layout.hole)
        return 1.0 / (distance_to_hole + 1)
    
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
            (agent_position[0] - 1, agent_position[1]),  # Up
            (agent_position[0] + 1, agent_position[1]),  # Down
            (agent_position[0], agent_position[1] - 1),  # Left
            (agent_position[0], agent_position[1] + 1)   # Right
        ]
        
        return int(any(pos in Layout.hole for pos in adjacent_positions))
    
    def f5(self, state, action, is_slippery=True):
        '''
        Verifica a possibilidade do chão estar escorregadio.
        '''
        agent_position = self._get_agent_position(state)
        row, column = agent_position
        
        if not is_slippery:
            slip_probability = 0.0
        else:
            slip_probability = 1.0 / 3.0
            
        initial_effectiveness = 1.0 - slip_probability
        slip_effectiveness = 0.0
        
        if action in [Actions.LEFT, Actions.RIGHT]:
            if row > Layout.row_min:
                slip_effectiveness += slip_probability
            if row < Layout.row_max:
                slip_effectiveness = slip_probability
        
        elif action in [Actions.UP, Actions.DOWN]:
            if column > Layout.column_min:
                slip_effectiveness += slip_probability
            if column < Layout.column_max:
                slip_effectiveness += slip_probability
                
        return initial_effectiveness + slip_effectiveness
