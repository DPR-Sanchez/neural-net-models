import CONSTANTS

import datetime
import dill
import models_training as neural_training
import PySimpleGUI as sg
import asyncio		


async def gui_layouts(layout):
	sg.ChangeLookAndFeel('Black')

	if layout == 'main menu':

		main_menu_layout = [
			[sg.Button('General Training'), sg.Button('DeepWatch')],
			[sg.T(' ' * 10)],
			[sg.Button('Exit')]
		]
		return main_menu_layout
	if layout == 'general':
		NETWORK_MODELS, COST_FUNCTION_NAMES, COST_FUNCTION_VALUES  = CONSTANTS.NEUR_CONSTANTS().get_constants_tuple()
		general_training_layout = [
			[sg.T(' ')],
			[sg.Button('Predict'), sg.Button('Train'),sg.Button('Show training history'),
			 sg.Text('Cost function for Training:')],
			[sg.T(' ' * 60), sg.InputCombo(COST_FUNCTION_NAMES, size=(20, 3))],
			[sg.T(' ' * 60), sg.Text('net model:')],
			[sg.T(' ' * 60), sg.InputCombo(NETWORK_MODELS, size=(20, 3))],
			[sg.T(' ' * 60), sg.Text('epoch training period:')],
			[sg.T(' ' * 60), sg.Input(default_text='100', size=(20, 3))],
			[sg.T(' ' * 60), sg.Text('numpy seed:')],
			[sg.T(' ' * 60), sg.Input(default_text='614', size=(20, 3))],
			[sg.T(' ' * 60), sg.Text('tensorflow  seed:')],
			[sg.T(' ' * 60), sg.Input(default_text='1234', size=(20, 3))],
			[sg.T(' ' * 60), sg.Text('rand seed:')],
			[sg.T(' ' * 60), sg.Input(default_text='2', size=(20, 3))],
			[sg.T(' ' * 60), sg.Text('uploaded dataset: ')],
			[sg.Text('_' * 80)],
			[sg.Text('Choose your desired dataset that you would like to predict or train on', size=(60, 1))],
			[sg.Text('Dataset:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select dataset >>'), sg.FileBrowse()],
			[sg.Text('select a trained neural network you would like to load', size=(60, 1))],
			[sg.Text('trained net:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select trained neural net >>'), sg.FileBrowse()],
			[sg.Text('Save trained network:', size=(35, 1))],
			[sg.Text('save location:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select save location >>'), sg.FileSaveAs()],
			[sg.Submit(), sg.Button('Load Net'), sg.Button('Save Net'), sg.Button('Exit')]
		]
		return general_training_layout

	if layout == 'deep watch':
		character_roster, level_of_play, map = CONSTANTS.DEEPWATCH_CONSTANTS().get_constants_tuple()

		deepwatch_layout = [
			[sg.Text('DeepWatch Interface', size=(30, 1), font=("Helvetica", 25))],
			[sg.Text('Your team\t\tEnemy Team')],
			[sg.InputCombo(character_roster, size=(20, 3)), sg.InputCombo(character_roster, size=(20, 3)),
			 sg.T(' ' * 10), sg.Text('Map:')],
			[sg.InputCombo(character_roster, size=(20, 3)), sg.InputCombo(character_roster, size=(20, 3)),
			 sg.T(' ' * 10), sg.InputCombo(map, size=(20, 3))],
			[sg.InputCombo(character_roster, size=(20, 3)), sg.InputCombo(character_roster, size=(20, 3)),
			 sg.T(' ' * 10), sg.Text('Level of Play:')],
			[sg.InputCombo(character_roster, size=(20, 3)), sg.InputCombo(character_roster, size=(20, 3)),
			 sg.T(' ' * 10), sg.InputCombo(level_of_play, size=(20, 3))],
			[sg.InputCombo(character_roster, size=(20, 3)), sg.InputCombo(character_roster, size=(20, 3))],
			[sg.InputCombo(character_roster, size=(20, 3)), sg.InputCombo(character_roster, size=(20, 3))],
			[sg.T(' ')],
			[sg.T(' ' * 10), sg.Button('Predict'), sg.T(' ' * 30), sg.Text('Label Data for Training:')],
			[sg.T(' ' * 10), sg.Text('Result:'), sg.T(' ' * 33), sg.InputCombo(('Loss'), size=(20, 3))],
			[sg.T(' ' * 8), sg.Text('', size=(10, 1), font=("Helvetica", 25), key='change'),
			 sg.Button('Add example')],
			[sg.Text('_' * 80)],
			[sg.Text('Choose A Folder', size=(35, 1))],
			[sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Selet file >>'), sg.FileBrowse()],
			[sg.Submit(), sg.Button('Exit')]
		]

		return deepwatch_layout

	return None



async def events_loop(layouts_list):
	exit_value = False
	main_menu_window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(await gui_layouts('main menu'))

	while not exit_value:
		event, values = main_menu_window.Read()

		if event is None or event == 'Exit':
			exit_value = True
		elif event == 'DeepWatch':
			with open('deepwatch.dill', 'rb') as file:
				network = dill.load(file)




			main_menu_window.Close()
			deepwatch_window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(await gui_layouts('deep watch'))

			character_roster, level_of_play, map = CONSTANTS.DEEPWATCH_CONSTANTS().get_constants_tuple()
			
			selection_tuple = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)
			hero_index = {character_roster[x]: x for x in range(0, len(character_roster))}
			map_index = {map[x]: x for x in range(len(map))}
			level_index = {level_of_play[x]: x for x in range(len(level_of_play))}

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
			general_training_window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(
				await gui_layouts('general'))

			trained_net = None

			while not exit_value:
				event, values = general_training_window.Read()
				print(values)
				if event is None or event == 'Exit':
					exit_value = True
					general_training_window.Close()
				elif event == 'Train':
					if values[6] != 'Select dataset >>' and values[6][-3:] == 'csv':
						# verify all integer inputs are actually integers
						try:
							values[2], values[3], values[4] = (int(x) for x in [values[2], values[3], values[4]])
						except Exception as e:
							sg.Popup("Please verify that all seed value inputs are integers")

						if all([type(1) == type(x) for x in [values[2], values[3], values[4]]]):
							# numpy tensor rand
							training_results = neural_training.train_model(numpy_seed=values[2], tensor_seed=values[3],
																			ran_seed=values[4], datasource=values[6],
																			network_select=values[1])
							trained_net = training_results

						else:
							sg.Popup("Please verify that all seed value inputs are integers")
					else:
						sg.Popup("Please select a csv based dataset")
				elif event == 'Save Net':
					if trained_net != None:
						print(event, "\n", values, "\n\n")
						file_save_name = f'{values[7]}.dill' if \
							values[7] != 'Select trained neural net >>' else \
							f"Neur trained net {str(datetime.datetime.now()).replace(':','').replace('-','').replace(' ','').replace('.','')}.dill"

						with open(file_save_name, 'wb') as f:
							dill.dump(trained_net, f)

					else:
						sg.Popup("Please train a Neural Network before attempting to save.")
				elif event == 'Show training history':
					pass
				elif event == 'Load Net':
					if trained_net is not None:
						nothing = None
					else:
						sg.Popup("Please either train a netowrk or load a prior trained network")
				elif event == 'Predict':
					pass

async def gui_launch():
	layouts = await gui_layouts('main menu')
	await events_loop(layouts)
	
if __name__ == '__main__':
	loop = asyncio.get_event_loop()
	task = loop.create_task(gui_launch())
	loop.run_until_complete(task)
	loop.close()