class NEUR_CONSTANTS:
	
	def __init__(self):
		self.NETWORK_MODELS = ('sequential 1', 'par net relu', 'par net sig', 'funnel net','scaling funnel net','noisy parallel sequential','models_mixture','leaky tansig')

		self.COST_FUNCTION_NAMES = ('Binary cross entropy',
							   'Mean Absolute Error',
								'Mean Squared Error',
								'Root Mean Squared Error',
								'Mean Squared Logarithmic Error',
								'Root Mean Squared Logarithmic Error',
								'Categorical cross entropy',
								'Binary hinge entropy',
								'Categorical hinge entropy'
							   )

		self.COST_FUNCTION_VALUES = {
				'Mean Absolute Error':'mae',
				'Mean Squared Error':'mse',
				'Root Mean Squared Error':'rmse',
				'Mean Squared Logarithmic Error':'msle',
				'Root Mean Squared Logarithmic Error':'rmsle',
				'Categorical cross entropy':'categorical_crossentropy',
				'Binary cross entropy':'binary_crossentropy',
				'Binary hinge entropy':'binary_hinge',
				'Categorical hinge entropy':'categorical_hinge'
		}
		
	def get_constants_tuple(self):
		return (self.NETWORK_MODELS, self.COST_FUNCTION_NAMES, self.COST_FUNCTION_VALUES)
	

class DEEPWATCH_CONSTANTS():

	def __init__(self):
		self.DEEPWATCH_CHARACTER_ROSTER = [
			'Ana', 'Ashe', 'Baptiste', 'Bastion', 'Brigitte', 'D.va',
			'Doomfist', 'Genji', 'Hanzo', 'Junkrat', 'Lucio', 'McCree',
			'Mei', 'Mercy', 'Moira', 'Orisa', 'Pharah', 'Reaper', 'Reinhardt',
			'Roadhog', 'Soldier', 'Sombra', 'Symmetra', 'Torbjorn', 'Tracer',
			'Widowmaker', 'Winston', 'Hammond', 'Zarya', 'Zenyatta'
		]

		self.DEEPWATCH_LEVEL_OF_PLAY = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'Grandmaster']

		self.DEEPWATCH_MAPS = [
			'Hanamura', 'Horizon', 'Paris', 'Anubis', 'Volskaya', 'Dorado', 'Havana', 'Junkertown',
			'Rialto', 'Route 66', 'Watchpoint Gibraltar', 'Blizzard World', 'Eichenwalde', 'Hollywood'
			'King\'s Row', 'Numbani', 'Busan', 'Ilios', 'Lijiang Tower', 'Nepal', 'Oasis'
		]	

	def get_constants_tuple(self):
		return (self.DEEPWATCH_CHARACTER_ROSTER, self.DEEPWATCH_LEVEL_OF_PLAY, self.DEEPWATCH_MAPS)