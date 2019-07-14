import PySimpleGUI as sg
import dill
import numpy as np
		
network_models = ('sequential 1', 'par net relu', 'par net sig', 'funnel net')

sg.ChangeLookAndFeel('Black')

main_menu_layout = [
						[sg.Button('General Training'),sg.Button('DeepWatch')],
						[sg.T(' '*10)],
						[sg.Button('Exit')]
					]

general_training_layout = 	[
								[sg.T(' ')],
								[sg.T(' ' * 10), sg.Button('Predict'), sg.T(' ' * 30), sg.Text('Label Data for Training:')],
								[sg.T(' ' * 60), sg.InputCombo(('Loss'), size=(20, 3))],
								[sg.T(' ' * 60), sg.Text('net model:')],
								[sg.T(' ' * 60), sg.InputCombo(network_models, size=(20, 3))],
								[sg.T(' ' * 60), sg.Text('epcch training period:')],
								[sg.T(' ' * 60), sg.Input(size=(20, 3))],
								[sg.T(' ' * 60), sg.Text('uploaded dataset: ')],
								[sg.Text('_' * 80)],
								[sg.Text('Choose your desired dataset', size=(35, 1))],
								[sg.Text('Dataset:', size=(15, 1), auto_size_text=False, justification='right'),
								 sg.InputText('Select dataset >>'), sg.FileBrowse()],
								[sg.Text('Save trained network:', size=(35, 1))],
								[sg.Text('save location:', size=(15, 1), auto_size_text=False, justification='right'),
								 sg.InputText('Select save location >>'), sg.FileBrowse()],
								[sg.Submit(), sg.Button('Save Net'),sg.Button('Exit')]
							]


main_menu_window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(main_menu_layout)

exit_value = False

while not exit_value:
	event, values = main_menu_window.Read()

	if event is None or event == 'Exit':
		exit_value = True
	elif event == 'DeepWatch':
		with open('deepwatch.dill', 'rb') as file:
			network = dill.load(file)
			
		character_roster = ['Ana', 'Ashe', 'Baptiste', 'Bastion', 'Brigitte', 'D.va',
					'Doomfist', 'Genji', 'Hanzo', 'Junkrat', 'Lucio', 'McCree',
					'Mei', 'Mercy', 'Moira', 'Orisa', 'Pharah', 'Reaper', 'Reinhardt',
					'Roadhog', 'Soldier', 'Sombra', 'Symmetra', 'Torbjorn', 'Tracer',
					'Widowmaker', 'Winston', 'Hammond', 'Zarya', 'Zenyatta']

		level_of_play = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'Grandmaster']

		map = ['Hanamura', 'Horizon', 'Paris', 'Anubis', 'Volskaya', 'Dorado', 'Havana', 'Junkertown',
				'Rialto', 'Route 66', 'Watchpoint Gibraltar', 'Blizzard World', 'Eichenwalde', 'Hollywood'
				'King\'s Row', 'Numbani', 'Busan', 'Ilios', 'Lijiang Tower', 'Nepal', 'Oasis']
		deepwatch_layout = [
			[sg.Text('DeepWatch Interface', size=(30, 1), font=("Helvetica", 25))],
			[sg.Text('Your team\t\tEnemy Team')],
			[sg.InputCombo(character_roster, size=(20, 3)),sg.InputCombo(character_roster, size=(20, 3)),sg.T(' '  * 10),sg.Text('Map:')],
			[sg.InputCombo(character_roster, size=(20, 3)),sg.InputCombo(character_roster, size=(20, 3)),sg.T(' '  * 10),sg.InputCombo(map, size=(20, 3))],
			[sg.InputCombo(character_roster, size=(20, 3)),sg.InputCombo(character_roster, size=(20, 3)),sg.T(' '  * 10),sg.Text('Level of Play:')],
			[sg.InputCombo(character_roster, size=(20, 3)),sg.InputCombo(character_roster, size=(20, 3)),sg.T(' '  * 10),sg.InputCombo(level_of_play, size=(20, 3))],
			[sg.InputCombo(character_roster, size=(20, 3)),sg.InputCombo(character_roster, size=(20, 3))],
			[sg.InputCombo(character_roster, size=(20, 3)),sg.InputCombo(character_roster, size=(20, 3))],
			[sg.T(' ')],
			[sg.T(' '*10),sg.Button('Predict'),sg.T(' '*30),sg.Text('Label Data for Training:')],
			[sg.T(' ' * 10), sg.Text('Result:'), sg.T(' '*33),sg.InputCombo(('Loss'), size=(20, 3))],
			[sg.T(' '*8), sg.Text('', size=(10, 1), font=("Helvetica", 25), key='change'),sg.Button('Add example')],
			[sg.Text('_'  * 80)],
			[sg.Text('Choose A Folder', size=(35, 1))],
			[sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Selet file >>'), sg.FileBrowse()],
			[sg.Submit(), sg.Button('Exit')]
		]
		
		main_menu_window.Close()
		deepwatch_window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(deepwatch_layout)

				
		selection_tuple = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)
		hero_index = {character_roster[x]:x for x in range(0,len(character_roster))}
		map_index = {map[x]:x for x in range(len(map))}
		level_index = {level_of_play[x]:x for x in range(len(level_of_play))}
		
		while not exit_value:
			event, values = deepwatch_window.Read()
			if event is None or event == 'Exit':
				exit_value = True
				deepwatch_window.Close()
			elif event == 'Predict':
				map_slection_index = 4
				level_selection_index = 9
				example = [hero_index[hero] for hero in [values[x] for x in selection_tuple]]
				example.append(map_index[values[map_slection_index]])
				example.append(level_index[values[level_selection_index]])
				pred = network.predict(example)
				deepwatch_window.Element('change').Update('Win' if pred[0][0] > 0.0 else 'Loss')
	elif event == 'General Training':
		main_menu_window.Close()
		general_training_window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(general_training_layout)

		while not exit_value:
			event, values = general_training_window.Read()
			if event is None or event == 'Exit':
				exit_value = True
				general_training_window.Close()
			elif event == 'Predict':
				pass