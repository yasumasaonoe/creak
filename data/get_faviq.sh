wget https://nlp.cs.washington.edu/ambigqa/data/faviq_r_set.zip
wget https://nlp.cs.washington.edu/ambigqa/data/faviq_a_set.zip
unzip faviq_a_set.zip
unzip faviq_r_set.zip
python convert_faviq.py
