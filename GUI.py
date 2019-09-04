import CONSTANTS
import models_training

import asyncio
import datetime
import dill
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import os
import re
import base64


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
			[sg.Button('Predict'), sg.Button('Train'),sg.Button('Show training history'),sg.Button('Aux. Test')],
			[sg.Text('Cost function for Training:'), sg.T(' ' * 2), sg.Text('net model:'), sg.T(' ' * 22),sg.Text('epoch training period:')],
			[sg.InputCombo(COST_FUNCTION_NAMES, size=(20, 3)),sg.InputCombo(NETWORK_MODELS, size=(20, 3)), sg.Input(default_text='100', size=(20, 3))],
			[sg.Text('numpy seed:'), sg.T(' ' * 12), sg.Text('tensorflow  seed:'), sg.T(' ' * 8),sg.Text('rand seed:')],
			[sg.Input(default_text='614', size=(20, 3)),sg.Input(default_text='1234', size=(20, 3)), sg.Input(default_text='2', size=(20, 3))],
			[sg.Text('training accuracy: ', size=(21, 1), key='accuracy'),
			 	sg.Text('training dataset: ', size=(30, 1),key='training dataset')],
			[sg.Text('_' * 80)],
			[sg.Text('Choose your desired dataset that you would like to predict or train on', size=(60, 1))],
			[sg.Text('Dataset:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select dataset >>'), sg.FileBrowse()],
			[sg.Checkbox(': dataset first column (left) is index ', default=True),sg.Checkbox(': dataset has headers ', default=False)],
			[sg.Text('select a trained neural network you would like to load', size=(60, 1))],
			[sg.Text('trained net:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select trained neural net >>'), sg.FileBrowse()],
			[sg.Text('Save trained network:', size=(35, 1))],
			[sg.Text('save location:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select save location >>'), sg.FileSaveAs()],
			[sg.Text('Save prediction:', size=(35, 1))],
			[sg.Text('save location:', size=(15, 1), auto_size_text=False, justification='right'),
			 sg.InputText('Select save location >>'), sg.FileSaveAs()],

			[sg.Submit(), sg.Button('Load Net'), sg.Button('Save Net'), sg.Button('Back'), sg.Button('Exit')]
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
			[sg.Submit(), sg.Button('Back'), sg.Button('Exit')]
		]

		return deepwatch_layout

	return None


async def windows(win='main'):

	if False == any([x for x in ['main', 'general', 'deepwatch']]):
		win = 'main'
	if win=='main':
		try:
			#check for existance of windows.icon
			type(windows.icon)
		except Exception as e:
			#makes windows.icon exist if it did not prior
			with open('Neur_Icon_256.png', 'rb') as raw:
				windows.icon = base64.b64encode(raw.read())
		return sg.Window(
							title='Neur - A Net Creation Tool',
							default_element_size=(40, 1),
							layout=await gui_layouts('main menu'),
							icon=windows.icon
						)
	elif win == 'general':
		return sg.Window(
							title='Neur - A Net Creation Tool',
							default_element_size=(40, 1),
							layout=await gui_layouts('general'),
							icon=windows.icon
						)

	elif win == 'deepwatch':
		return sg.Window(
							title='Neur - A Net Creation Tool',
							default_element_size=(40, 1),
							layout=await gui_layouts('deep watch'),
							icon=windows.icon
						)

async def events_loop(layouts_list):
	exit_value = False

	main_menu_window = await windows('main')

	while not exit_value:
		event, values = main_menu_window.Read()

		if event is None or event == 'Exit':
			exit_value = True
		elif event == 'DeepWatch':
			with open('deepwatch.dill', 'rb') as file:
				network = dill.load(file)

			main_menu_window.Close()
			deepwatch_window = await windows('deepwatch')

			character_roster, level_of_play, map = CONSTANTS.DEEPWATCH_CONSTANTS().get_constants_tuple()

			selection_tuple = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)
			hero_index = {character_roster[x]: x for x in range(0, len(character_roster))}
			map_index = {map[x]: x for x in range(len(map))}
			level_index = {level_of_play[x]: x for x in range(len(level_of_play))}

			deepwatch_window_close = False
			while not exit_value | deepwatch_window_close:
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
				elif event == 'Back':

					main_menu_window = await windows('main')
					deepwatch_window.Close()
					deepwatch_window_close = True

		elif event == 'General Training':
			main_menu_window.Close()
			general_training_window = await windows('general')

			trained_net = None
			optimizer = None
			general_window_close = False
			while not exit_value | general_window_close:

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

							neur_constants = CONSTANTS.NEUR_CONSTANTS()

							selected_loss_function = neur_constants.get_constants_tuple()[2][values[0]]
							optimizer,trained_net,accuracy = models_training.train_model(numpy_seed=values[2], tensor_seed=values[3],
																				ran_seed=values[4], data_source=values[6],
																				network_select=values[1], loss_function=selected_loss_function,
																				epochs_count=values[2], index=values[7],headers=values[8]
																				)
							general_training_window.Element('accuracy').Update(f'training accuracy: {accuracy}')

							file_name = values[6].split('/')[-1]

							general_training_window.Element('training dataset').Update(f'training dataset: {file_name}')
							general_training_window.Refresh()


						else:
							sg.Popup("Please verify that all seed value inputs are integers")
					else:
						sg.Popup("Please select a csv based dataset")

				elif event == 'Predict':
					if values[6] != 'Select dataset >>' and values[6][-3:] == 'csv':
						if trained_net is not None:
							if values[11] != 'Select save location >>':
								models_training.prediction(trained_net, data_source=values[6], index=values[7],save_location=values[11],headers=values[8])
							else:
								sg.Popup("Please select a location to save the prediction to.")
						else:
							sg.Popup("Please either load or train a Neural Network")
					else:
						sg.Popup("Please select a csv based dataset")

				elif event == 'Show training history':
					if optimizer != None:

						# try:
						# optimizer.plot_errors()
						optimizer.plot_errors(show=False)
						plt.savefig('training_history.png')
						image_elem = sg.Image(filename='training_history.png')

						display_training_layout = [
							[image_elem],
							[sg.Text(' ' * 26), sg.Button('Back')]
						]
						display_training_window = sg.Window(
							'Network Training history',
							default_element_size=(50, 1)
						).Layout(display_training_layout)

						event, _ = display_training_window.Read()

						if event == 'Back':
							display_training_window.Close()

					else:
						sg.Popup("Please train or load a neural net.")

				elif event == 'Aux. Test':
					if values[6] != 'Select dataset >>' and values[6][-3:] == 'csv':
						if trained_net is not None:
							accuracy = models_training.prediction(
																	trained_net,
																	mode='accuracy',
																	data_source=values[6],
																	index=values[7],
																	headers=values[8]
																)
							general_training_window.Element('accuracy').Update(f'training accuracy: {accuracy}')
							general_training_window.Refresh()
						else:
							sg.Popup("Please either load or train a Neural Network")
					else:
						sg.Popup("Please select a csv based dataset")

				elif event == 'Save Net':
					if trained_net != None:
						if values[10] != 'Select save location >>':
							file_save_name = f'{values[10]}.dill'
						else:
							time_stamp = re.sub('([,\.:\-\s])', '', str(datetime.datetime.now()))
							file_save_name = f'Neur trained net {time_stamp}.dill'

						with open(file_save_name, 'wb') as f:
							dill.dump(trained_net, f)

					else:
						sg.Popup("Please train a Neural Network before attempting to save.")

				elif event == 'Load Net':
					if values[9]!= 'Select trained neural net >>':

						load_pickle_confirm_layout = [
							[sg.Text('Loading pickle based objects can pose a security risk.', size=(40, 1), auto_size_text=False, justification='center')],
							[sg.Text('Are you sure you trust the origins of this file?', size=(40, 1),
									 auto_size_text=False, justification='center')],
							[sg.Text(' '*26),sg.Button('Yes'), sg.Button('No')]
						]
						confirm_window = sg.Window(
														'Load File Confirmation',
														default_element_size=(50, 1)
													).Layout(load_pickle_confirm_layout)

						event, _ = confirm_window.Read()

						if event == 'Yes':
							file_load_name = values[9]
							try:
								with open(file_load_name, 'rb') as f:
									trained_net = dill.load(f)
							except Exception as e:
								sg.Popup(str(e))
							finally:
								confirm_window.Close()
						else:
							confirm_window.Close()
					else:
						sg.Popup("Please select trained neural net to load")

				elif event == 'Back':

					main_menu_window = await windows('main')

					general_training_window.Close()
					general_window_close = True


async def gui_launch():
	layouts = await gui_layouts('main menu')
	await events_loop(layouts)

if __name__ == '__main__':
	loop = asyncio.get_event_loop()
	task = loop.create_task(gui_launch())
	loop.run_until_complete(task)
	loop.close()
