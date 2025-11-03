import os

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True) 

def save_args(args, filename='config.txt'):
    args_dict = vars(args)

    with open(filename, 'a') as f:
        for key, val in args_dict.items():
            f.write(f'{key}: {val}\n')
        f.write('\n')
    f.close()