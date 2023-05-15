# Adapting-OCR
Pytorch implementation of our paper [Adapting OCR with limited labels](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2020/AdaptingOCR_Deepayan_DAS2020_final.pdf)

![Qualitative Result of our Base, self-trained and hybrid model for English
(left) and Hindi (right) datasets. Here ST+FT refers to the model trained using
the proposed hybrid approach.](images/QualResults.png)

## Dependency

* This work was tested with PyTorch 1.2.0, CUDA 9.0, python 3.6 and Ubuntu 16.04.
* requirements can be found in the file.
* Also, please do a `pip install pytorch-pretrained-bert` as one of our kind contributors pointed out :)
* command to create environment from the file is `conda create -n pytorch1.4 --file env.txt`
* To activate the environment use: `source activate pytorch1.4`

## Training

* Supervised training 

`python -m train --name exp1 --path path/to/data `

* Main arguments
	* `--name`: creates a directory where checkpoints will be stored
	* `--path`: path to dataset. 
	* `--imgdir`: dir name of dataset


* Semi-supervised training

`python -m train_semi_supervised --name exp1 --path path --source_dir src_dirname --target_dir tgt_dirname --schedule --noise --alpha=1`

* Main arguments
	* `--name`: creates a directory where checkpoints will be stored
	* `--path`: path to datasets
	* `--source_dir`: labelled data directory on which ocr was trained
	* `--target_dir`: unlabeled data directory on which we want to adapt ocr
	* `--percent`: percentage of unlabeled data to include in self-training
	* `--schedule`: will include STLR scheduler while training
	* `--train_on_pred`: will treat top-predictions as targets
	* `--noise`: will add gaussian noise to images while training
	* `--alpha`: set to 1 to include the mixup criterion
	* `--combine_scoring`: will also take into account the scores outputted by a language model

**Note**: `--combine_scoring` works only with line images not word images

* Data 
	* Use [trdg](https://github.com/Belval/TextRecognitionDataGenerator) to generate synthetic data. The script for data generation is included `scrips/generate_data.sh`.
	* Download two different fonts and keep the data pertaining to each font in source and target dirs.
	* Use one of the fonts to train data from scratch in a supervised manner.
	* Then finetune the trained model on target data using semi-supervised learning
	* A sample lexicon is provided in `words.txt`. Download different lexicon as per need.


## References

* The OCR architecture is a CNN-LSTM model borrowed from [here](https://github.com/meijieru/crnn.pytorch)
* The mixup criterion code is borrowed from [here](https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L119)
* STLR is borrowed from this [paper](https://arxiv.org/abs/1801.06146)


