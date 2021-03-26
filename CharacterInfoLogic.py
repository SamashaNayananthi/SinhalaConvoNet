import pandas as pd
import json
import model.CharacterInfo as characterInfo

df = pd.read_csv('CharacterData.csv', header=0)


def get_info(characterClass):
    print(characterClass)
    characterInfoRow = df[(df["Class"] == characterClass)]
    print(characterInfoRow)
    print(str(characterInfoRow.iloc[0]["Character"]))

    info = characterInfo.CharacterInfo(str(characterInfoRow.iloc[0]["Character"]),
                                       str(characterInfoRow.iloc[0]["Name"]),
                                       str(characterInfoRow.iloc[0]["Unicode"]),
                                       str(characterInfoRow.iloc[0]["Phonetic"]),
                                       str(characterInfoRow.iloc[0]["Group"]),
                                       str(characterInfoRow.iloc[0]["Description"]))
    return json.dumps(info.__dict__)
