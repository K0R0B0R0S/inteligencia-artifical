import numpy as np
from feature_extractor import FeatureExtractor

class Actions:
	STICK = 0
	HIT = 1

class BlackjackFeatureExtractor(FeatureExtractor):
	__actions_one_hot_encoding = {
	Actions.STICK:   np.array([1,0]), 
	Actions.HIT:     np.array([0,1]) 
	}

	def __init__(self, env):
		'''
		Initializes the TaxiFeatureExtractor object. 
		It adds feature extraction methods to the features_list attribute.
		'''
		self.env = env
		self.features_list = []
		self.features_list.append(self.f0)
		self.features_list.append(self.f1_player_hand_sum)
		self.features_list.append(self.f2_dealer_visible_card)
		self.features_list.append(self.f3_player_aces_count)
		self.features_list.append(self.f4_is_bust)
		self.features_list.append(self.f5_distance_to_21)
		self.features_list.append(self.f6_probability_bust)

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
		if state[2] == True:
			return True
		elif state[0] > 21:
			return True
		return False

	def get_actions(self):
		'''
		Returns a list of available actions in the environment.
		'''
		return [Actions.STICK, Actions.HIT]

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
	
	def f0(self, state, action):
		'''
		This is just the bias term.
		'''
		return 1.0

	def f1_player_hand_sum(self, state, action):
		'''
		Returns the sum of the player's hand.
		'''
		return state[0]

	def f2_dealer_visible_card(self, state, action):
		'''
		Returns the value of the dealer's visible card.
		'''
		return state[1]

	def f3_player_aces_count(self, state, action):
		'''
		Returns 1 if the player holds a usable ace, otherwise 0.
		'''
		return state[2]  # Usable ace is directly given in state[2]

	def f4_is_bust(self, state, action):
		'''
		Returns 1 if the player's hand sum exceeds 21, otherwise 0.
		'''
		return 1 if state[0] > 21 else 0

	def f5_distance_to_21(self, state, action):
		'''
		Returns the difference between the player's hand sum and 21.
		'''
		return max(21 - state[0], 0)
	
	def f6_probability_bust(self, state, action):
		'''
		Calculates the probability of busting if another card is drawn.
		'''
		player_sum = state[0]
		cards_remaining = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Values 2-10 and Ace (11)
		probabilities = [1/13 for _ in cards_remaining]  # Assuming uniform deck distribution

		prob_bust = 0
		for card, prob in zip(cards_remaining, probabilities):
			if player_sum + card > 21:
				prob_bust += prob

		return prob_bust