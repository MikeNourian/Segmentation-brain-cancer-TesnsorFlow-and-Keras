patch:
	python extract_patches.py

train:
	python train.py
test:
	python predict.py

all:
	python extract_patches.py
	python train.py
	python predict.py
