{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clheight166 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadl100 \clpadr100 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\i\fs24 \cf0 Description. \
\

\i0 The classification model has been created with the objective to maximise the accuracy of model prediction. For the current project, we decided to stick to simpler models rather than neural networks. We used rule-based functions, which generated more informative features for different statistical learning algorithms. The ideas for generating features were partially borrowed from the paper by Mohammad, S., Kiritchenko, S. and Zhu, X. (2013). All of the features are binary (True/False), and correspond to the requirements of NLTK classifier. There were employed several models, and the best one (Linear SVC by scikit-learn) has been chosen as main classifier. \
\
In addition to unigrams and terms, among main features were:\
- emotionality of text\
- presence/absence of uppercase letters\
- presence/absence of elongated words\
- polarity score from lexicons\
- neutrality score if none of the lexicons words are in sentence\
- informative bigrams\
\
The only resource employed from external files is Bing Liu Lexicon(*), which consist of two lists of positive and negative words. This lexicon was used as general corpus that helps to assign some polarity scores for most frequent words and improve the model viability for testsets much different from the training set we utilised. \
\
* The sentiment lexicon of positive and negative words was composed by: \
Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."        \
    Proceedings of the ACM SIGKDD International Conference on Knowledge \
    Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, \
    Washington, USA, \cell \lastrow\row
}