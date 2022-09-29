Training:
- Go to the root of the project ./slot_attention
- Type in "python3 -m venv ./"
- Type in "source bin/activate"
- Go to this website: https://pytorch.org/ and choose your pytorch version.
- Copy the command for, it should look something like this: 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

- Execute this command in the root path: "python3 -m pip install -r requirements.txt"
- Go to the google drive folder https://drive.google.com/drive/folders/1oZPbTxWVzzqyYLPRaHjQiR2bLsG4VBwA?usp=sharing
- Download all datasets into the folder root/datasets/NAME and NAME is CHAIRS_EASY, CLEVR4, MDS, ...
- Each valid dataset has following subfolders: 'test', 'test_mask', 'train', 'val', 'val_mask'
- got to root/tmp and create folders CHAIRS, CLEVR4, MDS, ...

-Setting the parameters:
If you want to know which parameters to set, go to root/train.py, at the top you will see what arguments the parser accepts.

- run python3 -m train.py --param1 --param2 ...

Evaluation:
-Setting the parameters:
If you want to know which parameters to set, go to root/eval.py, at the top you will see what arguments the parser accepts.
- run python3 -m eval.py --param1 --param2 ...
