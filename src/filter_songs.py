# import os
#
# """
# This script is outdated and not relevant.  The Tagtraum labeling was inadequate. Tagged pop songs as metal.
# """
#
# tagraum_labels_dir = "data\\tagtraum\\id_list_"
#
# def find_songs_genre(genre: str, write_to_file=None) -> list:
#     """
#     Finds songs of a specific genre in the lmd_matched dataset. Returns filepaths to a specific songs matching the filepath
#
#     :param genre: Target Genre.  Case sensitive.
#     :param write_to_file: If given a string, write to the file the paths.  If given None, no file is writen to
#     :return: List of filepaths to target songs
#     """
#     matches = None
#     try:
#         p = tagraum_labels_dir + genre + ".txt"
#         with open(p, 'r') as file:
#             matches = set(file.read().split('\n'))
#     except FileNotFoundError:
#         raise ValueError("Invalid genre entered.  No list found")
#
#     def DFS_helper(path: str, current_list: list):
#         print(path)
#         if os.path.isdir(path):
#             for filename in os.listdir(path):
#                 DFS_helper(path + "\\" + filename, current_list)
#         else:
#             song_id = path.split("\\")[-2]
#             if song_id in matches:
#                 current_list.append(path)
#
#     paths = []
#     DFS_helper("data\\lmd_matched", paths)
#
#     if write_to_file:
#         with open("data\\" + write_to_file, 'w') as file:
#             file.write(''.join(paths).replace(' ', '\n'))
#
#     return paths
#
# find_songs_genre("Rock", "Rock Paths")
