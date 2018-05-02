I don't have documented all of the dependacies but these are a few

### Dependances

numpy  
opencv 3  
pytorch  
cffi  
editdistance

Install this from the repo:
https://github.com/SeanNaren/warp-ctc


### Running

#### Prepare Font Data

git clone https://github.com/cwig/prepare_font_data  
cd prepare_font_data  
bash run.sh  
cd ..  
python character_set.py prepare_font_data/training.json prepare_font_data/char_set.json

#### Prepare IAM Data

You will need to sign up for an IAM Database access account

git clone https://github.com/cwig/prepare_IAM_Lines   
cd prepare_IAM_Lines
sh download_IAM_data.sh  
python extract_all_words_lines.py  
cd ..  
python character_set.py prepare_IAM_Lines/raw_gts/lines/txt/training.json prepare_IAM_Lines/char_set.json  

### Train

python train.py sample_config.json

or 

python train.py sample_config_iam.json

This will run way too many epochs, just kill whenever

### Perform Recogition

python recognize.py sample_config.json prepare_font_data/output/0.png

or 

python recognize.py sample_config_iam.json prepare_IAM_Lines/lines/r06/r06-000/r06-000-00.png
