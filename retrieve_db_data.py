"""
    Used to retrieve player data from the database
"""

import pandas as pd
import numpy as np

import pymysql


class DB_Retriever:
    def __init__(self, host:str, database:str, username:str, password:str):
        self.host = host
        self.database = database
        self.db_username = username
        self.db_password = password


    def get_scores(self, firstname:str, lastname:str, birthyear:int):
        connection = pymysql.connect(
            host=self.host,
            port=3306,
            user=self.db_username,
            password=self.db_password,
            db=self.database
        )

        cursor = connection.cursor()

        id_query_text = f"SELECT `archerID` FROM `archers` WHERE LOWER(`firstname`) LIKE LOWER(\'{firstname}%\') AND LOWER(`lastname`) LIKE LOWER(\'{lastname}%\') AND `yearOfBirth` = {birthyear};"

        cursor.execute(id_query_text)
        ids = cursor.fetchall()
        if len(ids) == 0:
            return None

        list_ids = [id_tuple[0] for id_tuple in ids if id_tuple is not None]

        ids_textformatted = ",".join([str(x) for x in list_ids])

        total_scores_query_text = f"SELECT `archerID`, `date`, `totalScore`, `round` FROM `allRoundScores` WHERE `archerID` in ({ids_textformatted});"

        cursor.execute(total_scores_query_text)
        total_scores = cursor.fetchall()

        round_names = [entry[3] for entry in total_scores]
        round_names = list(tuple(round_names))
        formatted_round_names = "\',\'".join(round_name for round_name in round_names)

        max_scores_query = f"SELECT `round_definition`.`name`, SUM(`range_definition`.`ends` * `range_definition`.`arrowsPerEnd` * 10) AS maxScore FROM `round_definition` NATURAL JOIN `range_order` NATURAL JOIN `range_definition` GROUP BY `round_definition`.`roundDefID`;"

        cursor.execute(max_scores_query)
        max_scores = cursor.fetchall()

        max_scores_dict = {}
        for max_score in max_scores:
            max_scores_dict[max_score[0]] = int(max_score[1])

        output_max = {
            'RoundName': max_scores_dict.keys(),
            'MaxScore': max_scores_dict.values()
        }
        print(output_max)

        output = {
            "ArcherID":         [],
            "Date":             [],
            "ScoreFraction":    []
        }
        for entry in total_scores:
            output["ArcherID"].append(entry[0])
            output["Date"].append(entry[1])
            output["ScoreFraction"].append(float(entry[2]/max_scores_dict[entry[3]]))

        connection.close()
        return (pd.DataFrame(output), pd.DataFrame(output_max))
