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
        Manhattan distance from the agent's position to the goal.
        '''
        agent_pos = self._get_agent_position(state)
        return self.__manhattanDistance(agent_pos, Layout.goal)
    
    def f2(self, state, action):
        '''
        Returns 1 if the agent is adjacent to the cliff, 0 otherwise.
        '''
        agent_pos = self._get_agent_position(state)
        for cliff_cell in Layout.cliff:
            if self.__manhattanDistance(agent_pos, cliff_cell) == 1:
                return 1.0
        return 0.0
    
    def f3(self, state, action):
        '''
        Returns 1 if the action moves the agent closer to the goal, 0 otherwise.
        '''
        agent_pos = self._get_agent_position(state)
        next_pos = agent_pos.copy()
        
        if action == Actions.LEFT:
            next_pos[1] = max(Layout.column_min, agent_pos[1] - 1)
        elif action == Actions.RIGHT:
            next_pos[1] = min(Layout.column_max, agent_pos[1] + 1)
        elif action == Actions.UP:
            next_pos[0] = max(Layout.row_min, agent_pos[0] - 1)
        elif action == Actions.DOWN:
            next_pos[0] = min(Layout.row_max, agent_pos[0] + 1)
        
        current_distance = self.__manhattanDistance(agent_pos, Layout.goal)
        next_distance = self.__manhattanDistance(next_pos, Layout.goal)
        return 1.0 if next_distance < current_distance else 0.0
    
    def f4(self, state, action):
        '''
        Manhattan distance to the nearest cliff cell.
        '''
        agent_pos = self._get_agent_position(state)
        return min(self.__manhattanDistance(agent_pos, cliff_cell) for cliff_cell in Layout.cliff)
    
    def f5(self, state, action):
        '''
        Checks if the action aligns with the row or column direction of the goal.
        '''
        agent_pos = self._get_agent_position(state)
        if action == Actions.LEFT and Layout.goal[1] < agent_pos[1]:
            return 1.0
        elif action == Actions.RIGHT and Layout.goal[1] > agent_pos[1]:
            return 1.0
        elif action == Actions.UP and Layout.goal[0] < agent_pos[0]:
            return 1.0
        elif action == Actions.DOWN and Layout.goal[0] > agent_pos[0]:
            return 1.0
        return 0.0