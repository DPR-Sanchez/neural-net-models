import PySimpleGUI as sg

character_roster = ('Tracer','Bastion','Hanzo','Genji','McCree','Reaper','D.va','Soldier76','DoomFist','Hammond','Rheinhart','Winston',
					'Baptiste','Brigette','Torjorn','Anna','Mei','Mercy','Lucio','Orisa','Junkrat','Roadhog','Ashe','Moira','Phara',
					'Sombra','Widow Maker','Symmetra','Zarya','Zenyatta')
level_of_play = ['Bronze','Silver','Gold','Platinum','Diamond','Master','Grand Master']
map = ['Hanamura']
sg.ChangeLookAndFeel('Black')

column1 = [[sg.Text('Column 1', background_color='#d3dfda', justification='center', size=(10, 1))],
           [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],
           [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],
           [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]]
layout = [
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
	[sg.T(' '*60),sg.InputCombo(('Loss'), size=(20, 3))],
	[sg.T(' '*60),sg.Button('Add example')],
    [sg.Text('_'  * 80)],
    [sg.Text('Choose A Folder', size=(35, 1))],
    [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
     sg.InputText('Default Folder'), sg.FolderBrowse()],
    [sg.Submit(), sg.Cancel()]
]

selection_tuple = ()
hero_index = {character_roster(x):x for x in range(0,len(character_roster))}

window = sg.Window('Neur - A Net Creation Tool', default_element_size=(40, 1)).Layout(layout)

while True:
	event, values = window.Read()

	if event is None or event == 'Exit':
		break
	elif event == 'Predict':
		heros_selected = [hero_index[hero] for hero in [values[x] for x in selection_tuple]]

	print(event, values)

window.Close()