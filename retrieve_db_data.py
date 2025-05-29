"""
	Used to retrieve player data from the database
"""

import pandas as pd
import pymysql
from pymysql.connections import Connection

from settings import SETTINGS

class DB_Retriever:
	def __init__(self, _host: str, _database: str, _username: str, _password: str):
		self.host = _host
		self.database = _database
		self.db_username = _username
		self.db_password = _password

	def _get_connection(self) -> Connection | None:
		"""Helper method to establish database connection"""
		try:
			connection = pymysql.connect(
				host=self.host,
				port=3306,
				user=self.db_username,
				password=self.db_password,
				db=self.database
			)
			return connection
		except Exception as e:
			print(f'Failed to connect: {e}')
			return

	def get_archer_scores(self, _firstname: str, _lastname: str, _birthyear: int) -> pd.DataFrame | None:
		"""
		Retrieve archer scores from the database

		Args:
			_firstname: Archer's first name
			_lastname: Archer's last name
			_birthyear: Archer's birth year

		Returns:
			pandas.DataFrame with columns: ArcherID, Date, TotalScore, Round
			Returns None if no archers found or connection fails
		"""
		connection = self._get_connection()
		if connection is None:
			return

		cursor = connection.cursor()

		# Find archer IDs matching the criteria
		id_query_text = f'SELECT `archerID` FROM `archers` WHERE LOWER(`firstname`) LIKE LOWER(\'{_firstname}%\') AND LOWER(`lastname`) LIKE LOWER(\'{_lastname}%\') AND `yearOfBirth` = {_birthyear};'

		cursor.execute(id_query_text)
		ids = cursor.fetchall()
		if len(ids) == 0:
			print('Did not find any archers matching the parameters')
			connection.close()
			return

		list_ids = [id_tuple[0] for id_tuple in ids if id_tuple is not None]
		ids_textformatted = ','.join([str(x) for x in list_ids])

		# Get scores for the found archers
		total_scores_query_text = f'SELECT `archerID`, `date`, `totalScore`, `round` FROM `allRoundScores` WHERE `archerID` in ({ids_textformatted});'

		cursor.execute(total_scores_query_text)
		total_scores = cursor.fetchall()

		# Format output as DataFrame
		output = {
			'ArcherID': [entry[0] for entry in total_scores],
			'Date': [entry[1] for entry in total_scores],
			'TotalScore': [entry[2] for entry in total_scores],
			'Round': [entry[3] for entry in total_scores]
		}

		connection.close()
		return pd.DataFrame(output)

	def get_round_info(self) -> pd.DataFrame | None:
		"""
		Retrieve round information including maximum possible scores

		Returns:
			pandas.DataFrame with columns: RoundName, MaxScore
			Returns None if connection fails
		"""
		connection = self._get_connection()
		if connection is None:
			return

		cursor = connection.cursor()

		# Get maximum scores for all rounds
		max_scores_query = f'SELECT `round_definition`.`name`, SUM(`range_definition`.`ends` * `range_definition`.`arrowsPerEnd` * 10) AS maxScore FROM `round_definition` NATURAL JOIN `range_order` NATURAL JOIN `range_definition` GROUP BY `round_definition`.`roundDefID`;'

		cursor.execute(max_scores_query)
		max_scores = cursor.fetchall()

		# Format output as DataFrame
		output = {
			'RoundName': [entry[0] for entry in max_scores],
			'MaxScore': [int(entry[1]) for entry in max_scores]
		}

		connection.close()
		return pd.DataFrame(output)

	def get_scores_as_fraction(
		self,
		_firstname: str,
		_lastname: str,
		_birthyear: int
	) -> pd.DataFrame | None:
		"""
		Get archer scores with score fractions (score/max_score)

		Args:
			_firstname: Archer's first name
			_lastname: Archer's last name
			_birthyear: Archer's birth year

		Returns:
			tuple of (scores_df, rounds_df) where:
			- scores_df has columns: ArcherID, Date, ScoreFraction
			- rounds_df has columns: RoundName, MaxScore
			Returns None if no data found
		"""
		# Get archer scores and round info
		scores_df = self.get_archer_scores(_firstname, _lastname, _birthyear)
		rounds_df = self.get_round_info()

		if scores_df is None or rounds_df is None or scores_df.empty:
			return

		# Create max scores dictionary for lookup
		max_scores_dict = dict(zip(rounds_df['RoundName'], rounds_df['MaxScore']))

		# Calculate score fractions
		output = {
			'ArcherID': [],
			'Date': [],
			'ScoreFraction': []
		}

		for _, row in scores_df.iterrows():
			round_name = row['Round']
			if round_name in max_scores_dict:
				output['ArcherID'].append(row['ArcherID'])
				output['Date'].append(row['Date'])
				output['ScoreFraction'].append(float(row['TotalScore'] / max_scores_dict[round_name]))

		return pd.DataFrame(output)


def main():
	retriever = DB_Retriever(
		SETTINGS.database_host,
		SETTINGS.database,
		SETTINGS.database_username,
		SETTINGS.database_password
	)

	firstname = 'Moselle'
	lastname = 'Speachley'
	yob = 1986

	archer_scores = retriever.get_archer_scores(firstname, lastname, yob)
	if archer_scores is not None:
		print('Archer scores:')
		print(archer_scores)

	round_info = retriever.get_round_info()
	if round_info is not None:
		print('\nRound info:')
		print(round_info)

	scores_df = retriever.get_scores_as_fraction(firstname, lastname, yob)
	if scores_df is not None:
		print('\nScores with fractions:')
		print(scores_df)

if __name__ == '__main__':
	main()
