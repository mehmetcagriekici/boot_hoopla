import json

def load_json(path):
    try:
        obj = open(path)
        return json.load(obj)
    except OSError as err:
        print("Couldn't open the file at provided path!", err)
    except json.DecodeError as err:
        print("Couldn't deoce the file. Invalid document!", err)
    except UnicodeDecodeError as err:
        print("Couldn't decode the file!", err)
        
