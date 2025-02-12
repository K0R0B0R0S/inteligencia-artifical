from feature_extractor import FeatureExtractor
import numpy as np

class Actions:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class Layout:
    goal = [3, 11]
    # start = [3, 0]
    start = [2, 6] #Para CliffWalking
    cliff = [[3, i] for i in range(1, 11)]
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
        self.features_list.append(self.f6)
        self.features_list.append(self.f7)
        self.features_list.append(self.f8)
        self.features_list.append(self.f_penalty_for_loop)

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
        goal_state = 47
        return state == goal_state
    
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
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    
    @staticmethod
    def __eucledianDistance(xy1, xy2):
        return np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)
    
    def _get_agent_position(self, state):
        '''
        Retorna a posição do agente
        '''
        return [state // 12, state % 12]
    
    def f0(self, state, action):
        '''
        This is just the bias term.
        '''
        return 1.0
    
    def f1(self, state, action):
        '''
        Distância normalizada entre o agente e o objetivo.
        '''
        agent_position = self._get_agent_position(state)
        goal_position = Layout.goal
        distance = self.__eucledianDistance(agent_position, goal_position)
        return 1 / (1 + distance)
    
    def f2(self, state, action):
        '''
        Distância normalizada entre o agente e o penhasco.
        '''
        agent_position = self._get_agent_position(state)
        cliff_positions = Layout.cliff
        distance = min([self.__eucledianDistance(agent_position, cliff) for cliff in cliff_positions])
        return 1 / (1 + distance)
    
    def f3(self, state, action):
        '''
        Verifica se o agente está nas proximidades do penhasco.
        '''
        agent_position = self._get_agent_position(state)
        cliff_positions = Layout.cliff

        for cliff in cliff_positions:
            if self.__eucledianDistance(agent_position, cliff) == 1:
                return 1.0
        return 0.0
    
    def f4(self, state, action):
        '''
        Distância normalizada entre o agente e a posição inicial.
        '''
        agent_position = self._get_agent_position(state)
        start_position = Layout.start
        distance = self.__eucledianDistance(agent_position, start_position)
        return 1 / (1 + distance)
    
    def f5(self, state, action):
        '''
        Posição do agente na coluna (indicando onde ele está na direção horizontal).
        '''
        agent_position = self._get_agent_position(state)
        return agent_position[1] / Layout.column_max
    
    def f6(self, state, action):
        '''
        Penalidade se o agente estiver em uma coluna do penhasco.
        '''
        agent_position = self._get_agent_position(state)
        if agent_position[0] == 3 and 1 <= agent_position[1] <= 10:
            return 1.0
        return 0.0
    
    def f7(self, state, action):
        '''
        Posição do agente na linha.
        '''
        agent_position = self._get_agent_position(state)
        return agent_position[0] / Layout.row_max
    
    def f8(self, state, action):
        '''
        Indica se o agente está na linha final.
        '''
        agent_position = self._get_agent_position(state)
        return 1.0 if agent_position[0] == Layout.row_max else 0.0
    
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