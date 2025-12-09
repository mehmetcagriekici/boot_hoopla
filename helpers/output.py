import re

def print_movies(movies, query):
    matches = []
    # get matches
    for i in range(len(movies)):
        if find_match(movies[i]["title"], query):
            matches.append(movies[i])
    # sort matches by id
    sorted(matches, key=lambda movie: movie["id"])
    # print matches
    for i in range(len(matches[:4])):
        title = matches[i]["title"]
        print(f"{i + 1}. {title}")

# wide search
def find_match(title, query):
    for tq in tokenize(query):
        for tt in tokenize(title):
            if tq in tt:
                return True
    return False

# create token strings
def tokenize(string):
    return list(filter(lambda w: w != "", re.sub(r'[^\w\s]+', "", string).lower().split()))
