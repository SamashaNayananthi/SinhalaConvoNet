import pandas as pd
import json
import model.CharacterInfo as characterInfo

df = pd.read_csv('CharacterData.csv', header=0)


def get_info(characterClass):
    print(characterClass)
    characterInfoRow = df[(df["Class"] == characterClass)]
    print(str(characterInfoRow["Character"]))

    info = characterInfo.CharacterInfo(str(characterInfoRow["Character"]), str(characterInfoRow["Name"]),
                                       str(characterInfoRow["Unicode"]), str(characterInfoRow["Phonetic"]),
                                       str(characterInfoRow["Group"]), str(characterInfoRow["Description"]))
    return json.dumps(info.__dict__)
