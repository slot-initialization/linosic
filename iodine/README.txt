Training:
- Go to the root of the project ./iodine
- Type in "python3 -m venv ./"
- Type in "source bin/activate"
- Go to this website: https://pytorch.org/ and choose your pytorch version.
- Copy the command for the download, it should look something like this:
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

- Execute this command in the root path: "python3 -m pip install -r requirements.txt"
- Go to the google drive folder https://drive.google.com/drive/folders/1oZPbTxWVzzqyYLPRaHjQiR2bLsG4VBwA?usp=sharing
- Download all datasets into the folder root/datasets/NAME and NAME is CHAIRS_EASY, CLEVR4, MDS, ...
- Each valid dataset has following subfolders: 'test', 'test_mask', 'train', 'val', 'val_mask'

-Setting the parameters:
If you want to know which parameters to set, go to root/arg_parser.py, at the top you will see what arguments the parser accepts.

-Run training:
Go to root/scripts/train.sh and adapt+run the shell with:
/bin/bash scripts/train.sh in your root directory.

Evaluation:
Go to root/scripts/eval.sh and adapt+run the shell with:
/bin/bash scripts/eval.sh in your root directory.
