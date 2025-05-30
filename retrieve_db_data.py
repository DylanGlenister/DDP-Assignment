"""
	Used to retrieve player data from the database
"""

import pandas as pd
import pymysql
from pymysql.connections import Connection

import shared
from settings import SETTINGS


class DB_Retriever:
	def __init__(self):
		self.host = SETTINGS.database_host
		self.database = SETTINGS.database_name
		self.db_username = SETTINGS.database_username
		self.db_password = SETTINGS.database_password

	def _get_connection(self) -> Connection:
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
			raise ConnectionError(f'Could not connect to the database: {e}')

	def get_archer_scores(
		self,
		_firstname: str,
		_lastname: str,
		_birthyear: int,
		_round: str | None = None
	) -> pd.DataFrame:
		"""
		Retrieve archer scores from the database

		Args:
			_firstname: Archer's first name
			_lastname: Archer's last name
			_birthyear: Archer's birth year
			_round: Optional round filter to apply to results

		Returns:
			pandas.DataFrame with columns: ArcherID, Date, TotalScore, Round
			Returns empty DataFrame if no archers found or connection fails
		"""
		# Define the empty DataFrame structure once
		empty_df = pd.DataFrame(columns=[
			shared.COLUMN_ARCHER_ID,
			shared.COLUMN_DATE,
			shared.COLUMN_TOTAL_SCORE,
			shared.COLUMN_ROUND
		])

		try:
			connection = self._get_connection()
			cursor = connection.cursor()

			# Find archer IDs matching the criteria using parameterized query
			id_query_text = '''
				SELECT `archerID`
				FROM `archers`
				WHERE LOWER(`firstname`) LIKE LOWER(%s)
				AND LOWER(`lastname`) LIKE LOWER(%s)
				AND `yearOfBirth` = %s
			'''

			cursor.execute(id_query_text, (f'{_firstname}%', f'{_lastname}%', _birthyear))
			ids = cursor.fetchall()

			if not ids:
				print('Did not find any archers matching the parameters')
				return empty_df

			# Extract archer IDs safely
			list_ids = [id_tuple[0] for id_tuple in ids if id_tuple and id_tuple[0] is not None]

			if not list_ids:
				return empty_df

			# Get scores for the found archers using parameterized query
			placeholders = ','.join(['%s' for _ in list_ids])
			total_scores_query_text = f'''
				SELECT `archerID`, `date`, `totalScore`, `round`
				FROM `allRoundScores`
				WHERE `archerID` IN ({placeholders})
			'''

			cursor.execute(total_scores_query_text, list_ids)
			total_scores = cursor.fetchall()

			# Create DataFrame directly from query results
			if not total_scores:
				return empty_df

			output = pd.DataFrame(total_scores, columns=[
				shared.COLUMN_ARCHER_ID,
				shared.COLUMN_DATE,
				shared.COLUMN_TOTAL_SCORE,
				shared.COLUMN_ROUND
			])

			# Apply round filter if specified
			if _round is not None:
				output = output[output[shared.COLUMN_ROUND] == _round]

			return output

		except Exception as e:
			print(f'Error retrieving archer scores: {e}')
			return empty_df

		finally:
			# Ensure connection is closed even if an exception occurs
			if 'connection' in locals():
				connection.close()  # type: ignore

	def get_round_info(self) -> pd.DataFrame:
		"""
		Retrieve round information including maximum possible scores

		Returns:
			pandas.DataFrame with columns: RoundName, MaxScore
			Returns empty DataFrame if connection fails
		"""
		# Define the empty DataFrame structure for connection failures
		try:
			connection = self._get_connection()
			cursor = connection.cursor()

			# Get maximum scores for all rounds
			max_scores_query = '''
				SELECT `round_definition`.`name`,
						SUM(`range_definition`.`ends` * `range_definition`.`arrowsPerEnd` * 10) AS maxScore
				FROM `round_definition`
				NATURAL JOIN `range_order`
				NATURAL JOIN `range_definition`
				GROUP BY `round_definition`.`roundDefID`
			'''

			cursor.execute(max_scores_query)
			max_scores = cursor.fetchall()

			# Create DataFrame directly from query results
			output = pd.DataFrame(max_scores, columns=[
				shared.COLUMN_ROUND_NAME,
				shared.COLUMN_MAX_SCORE
			])

			# Convert MaxScore to int
			output[shared.COLUMN_MAX_SCORE] = output[shared.COLUMN_MAX_SCORE].astype(int)

			return output

		except Exception as e:
			print(f'Error connecting to database: {e}')
			return pd.DataFrame(columns=[
				shared.COLUMN_ROUND_NAME,
				shared.COLUMN_MAX_SCORE
			])

		finally:
			# Ensure connection is closed even if an exception occurs
			if 'connection' in locals():
				connection.close()  # type: ignore

	def get_scores_as_fraction(
		self,
		_firstname: str,
		_lastname: str,
		_birthyear: int,
		_round: str | None = None
	) -> pd.DataFrame:
		"""
		Get archer scores with score fractions (score/max_score)

		Args:
			_firstname: Archer's first name
			_lastname: Archer's last name
			_birthyear: Archer's birth year
			_round: Optional round filter to apply to results

		Returns:
			pandas.DataFrame with columns: ArcherID, Date, ScoreFraction
			Returns empty DataFrame if no data found or error occurs
		"""
		# Define the empty DataFrame structure
		empty_df = pd.DataFrame(columns=[
			shared.COLUMN_ARCHER_ID,
			shared.COLUMN_DATE,
			shared.COLUMN_SCORE_FRACTION
		])

		try:
			# Get archer scores and round info
			scores_df = self.get_archer_scores(
				_firstname,
				_lastname,
				_birthyear,
				_round
			)

			if scores_df.empty:
				return empty_df

			rounds_df = self.get_round_info()

			if rounds_df.empty:
				print('Warning: No round information available')
				return empty_df

			# Create max scores dictionary for lookup
			max_scores_dict = dict(zip(rounds_df[shared.COLUMN_ROUND_NAME], rounds_df[shared.COLUMN_MAX_SCORE]))

			# Calculate score fractions using vectorized operations
			valid_scores = scores_df[scores_df[shared.COLUMN_ROUND].isin(max_scores_dict.keys())].copy()

			if valid_scores.empty:
				print('Warning: No scores found for known rounds')
				return empty_df

			# Map max scores and calculate fractions
			valid_scores['max_score'] = valid_scores[shared.COLUMN_ROUND].map(max_scores_dict)
			valid_scores[shared.COLUMN_SCORE_FRACTION] = (
				valid_scores[shared.COLUMN_TOTAL_SCORE] / valid_scores['max_score']
			).astype(float)

			# Return only the required columns
			output = valid_scores[[
				shared.COLUMN_ARCHER_ID,
				shared.COLUMN_DATE,
				shared.COLUMN_SCORE_FRACTION
			]].copy()

			return output

		except Exception as e:
			print(f'Error calculating score fractions: {e}')
			return empty_df


def main():
	retriever = DB_Retriever()

	firstname = 'Devi'
	lastname = 'Cropton'
	yob = 2012

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

""" === Output reference ===

	Archer scores:
	   ArcherID        Date TotalScore       Round
	0         7  2024-01-10        569     Samford
	1         7  2025-04-08        265      Darwin
	2         7  2025-01-01        392  Wollongong
	3         7  2025-02-14        452    Canberra

	Round info:
	          RoundName  MaxScore
	0         WA90/1440      1440
	1         WA70/1440      1440
	2         WA60/1440      1440
	3         AA50/1440      1440
	4         AA40/1440      1080
	5          WA70/720       720
	6          WA60/720       720
	7          WA50/720       720
	8          Canberra       900
	9       Long Sydney      1200
	10           Sydney      1200
	11    Long Brisbane      1200
	12         Brisbane      1200
	13         Adelaide      1200
	14   Short Adelaide      1200
	15           Hobart       900
	16            Perth       900
	17   Short Canberra       900
	18  Junior Canberra       900
	19    Mini Canberra       900
	20           Grange       900
	21        Melbourne       900
	22           Darwin       900
	23          Geelong       900
	24        Newcastle       900
	25             Holt       900
	26          Samford       900
	27            Drake       900
	28       Wollongong       720
	29       Townsville       720
	30       Launceston       720
	31      Full Spread      1200

	Scores with fractions:
	   ArcherID        Date  ScoreFraction
	0         7  2024-01-10       0.632222
	1         7  2025-04-08       0.294444
	2         7  2025-01-01       0.544444
	3         7  2025-02-14       0.502222

"""
