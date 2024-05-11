import os
import numpy as np

def count_subdirectories(directory_path):
    try:
        subdirectories = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
        return len(subdirectories)
    except OSError:
        return 0
    

def check_dir_existence(dir):
    if not os.path.exists(dir):
        # create dir if not exist
        os.mkdir(dir)
        print(f"Dir '{dir}' created.")
    else:
        print(f"Dir '{dir}' already exist.")


def save_graph_or_decay(dir_path, file_name, data_type, data_mat):
    check_dir_existence(dir_path)

    id = 1
    for filename in os.listdir(dir_path):
        if filename.endswith('.npy') and filename.startswith(file_name):
            id += 1
    file_name = file_name + '_' + str(id)
    print('Save {} as {}'.format(data_type, file_name))
    np.save(dir_path+file_name, data_mat)