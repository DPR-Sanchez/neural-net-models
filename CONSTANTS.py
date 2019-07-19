def constants():
	constants.NETWORK_MODELS = ('sequential 1', 'par net relu', 'par net sig', 'funnel net')

	constants.COST_FUNCTION_NAMES = ('Binary cross entropy',
						   'Mean Absolute Error',
							'Mean Squared Error',
							'Root Mean Squared Error',
							'Mean Squared Logarithmic Error',
							'Root Mean Squared Logarithmic Error',
							'Categorical cross entropy',
							'Binary hinge entropy',
							'Categorical hinge entropy'
						   )

	constants.COST_FUNCTION_VALUES = {
	'Mean Absolute Error':'mae','Mean Squared Error':'mse',
			'Root Mean Squared Error':'rmse',
			'Mean Squared Logarithmic Error':'msle',
			'Root Mean Squared Logarithmic Error':'rmsle',
			'Categorical cross entropy':'categorical_crossentropy',
			'Binary cross entropy':'binary_crossentropy',
			'Binary hinge entropy':'binary_hinge',
			'Categorical hinge entropy':'categorical_hinge'
	}

constants()