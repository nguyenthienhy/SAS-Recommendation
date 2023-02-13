from tqdm import tqdm

# f = open("data/ml-10M100K/ratings.dat", "r")
# lines = f.readlines()

# f = open("data/ml-1m.txt", "r")
# lines = f.readlines()

f = open("data/ml-20m/ratings.csv", "r")
lines = f.readlines()[1:]

with open("data/ml-20m.txt", "w") as fw:
    for i, line in enumerate(tqdm(lines)):
        line = line.replace("\n", "")
        parts = line.split(",")
        fw.write(parts[0] + " " + parts[1] + '\n')

# user_id = []

# for line in tqdm(lines):
#     line = line.replace("\n", "")
#     parts = line.split("::")
#     user_id.append(parts[0])

# def myFunc(e):
#   return int(e)

# user_id_to_sort = list(set(user_id))
# user_id_to_sort.sort(reverse=False, key=myFunc)

# for i, id in enumerate(tqdm(user_id)):
#     if id in user_id_to_sort:
#         index = user_id_to_sort.index(id)
#         user_id[i] = index + 1

# with open("data/ml-10m.txt", "w") as fw:
#     for i, line in enumerate(tqdm(lines)):
#         line = line.replace("\n", "")
#         parts = line.split("::")
#         fw.write(str(user_id[i]) + " " + parts[1] + '\n')


        