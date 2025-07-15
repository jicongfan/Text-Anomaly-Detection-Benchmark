__author__ = 'XF'
__date__ = '2024/07/27'


'''
Login setting of Huggingface.
'''


import os


def hf_login(token, mirror):

    command = f'export HF_ENDPOINT={mirror}'
    os.system(command=command)

    command = f'huggingface-cli login --token {token} --add-to-git-credential=True'
    print(command)
    os.system(command=command)