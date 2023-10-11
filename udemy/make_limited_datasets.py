import os
import shutil

def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def copy_directory(src, dst):
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

mkdir(r'C:\Users\anja.kovacevic\kod\udemy\large_files\fruits-360-small')

classes = [
    #'Apple Golden 1',
    #'Avocado',
    #'Lemon',
    #'Mango',
    #'Kiwi',
    'Banana',
    'Strawberry',
    'Raspberry'
]

train_path_from = os.path.abspath(r'C:\Users\anja.kovacevic\kod\udemy\large_files\fruits-360\Training')
valid_path_from = os.path.abspath(r'C:\Users\anja.kovacevic\kod\udemy\large_files\fruits-360\Test')

train_path_to = os.path.abspath(r'C:\Users\anja.kovacevic\kod\udemy\large_files\fruits-360-small\Training')
valid_path_to = os.path.abspath(r'C:\Users\anja.kovacevic\kod\udemy\large_files\fruits-360-small\Validation')

mkdir(train_path_to)
mkdir(valid_path_to)

for c in classes:
    copy_directory(os.path.join(train_path_from, c), os.path.join(train_path_to, c))
    copy_directory(os.path.join(valid_path_from, c), os.path.join(valid_path_to, c))
