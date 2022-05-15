1. VarIDE Description
    - This system participated in edition 1.1 of the PARSEME shared task on automatic identification of verbal multiword expressions (VMWEs). 
Our system focuses on the task of VMWE variant identification by using morphosyntactic information in the training data to predict if candidates extracted from the test corpus could be idiomatic, thanks to a naive Bayes classifier.
Candidate identification also includes their categorization according to the [PARSEME guidelines] (/http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1//)

2. VarIDE utilisation
    - Requirements:
        - Package requirements:
            * **numpy** (pip install numpy) 
            * **tqdm** (pip install tqdm)
            * python Natural Language Toolkit (**NLTK**) (pip install nltk)
        - Input files are in the INPUT folder: 
            * 3 corpora *train*, *dev* and *test* provided for 19 languages: BG,DE,EL,EN,ES,EU,FA,FR,HE,HI,HR,HU,IT,LT,PL,PT,RO,SL,TR
            * *train* and *dev* provide manual annotation for verbal multiword Expressions (VMWEs) while 2 versions of *test* are available (gold + blind)
            * The corpora are provided in the [.cupt format] (/http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018&subpage=CONF_45_Format_specification/)
		- Parameters : see config file (code/config.cfg) to choose the language corpora, the max insertion filter and the languages for which is applied this filter and whether you want to use the classifier for many files (in this case, you need to set importClassifier to True).
    - To launch varIDE:
        - Clone or download the repository
        - Go into the "code" folder then run into the shell `python3 varIDE.py`
        - *dev* and *test* corpora annotated by VarIDE are in the OUTPUT folder for each language 
        
        
3. VarIDE evaluation 
    - [Evaluation script] (/http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018&subpage=CONF_50_Evaluation_metrics/) 
