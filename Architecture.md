*************************************

SBERT / Classification heads

SBERT (Sentence Bidirectional Encoder Representation Transformers) is a sentence-bert-encoder only model which is a foundational NLP model. 

SBERT encodes sentence embeddings by (Mean Pooling) averaging the embeddings of all output tokens to create a fixed vector size/representation. There are several methods to go about encoding sentence embeddings but this is generally the most optimal path.

I am going to finetune the ‘all-mpnet-base-v2’ SBERT model by using BABE, a media bias dataset, on the last three attention layers of our model (will freeze the other layers). The all-mpnet-base-v2 model is good enough on its own at generalizing basic biased language but I am fine tuning on media bias so that the model can better encode differences and nuance between bias levels that may otherwise be embedded as semantically similar.

Going to add two classification heads including one bias classification head and an opinion-label classification head which serves as an auxiliary loss function to the primary loss of bias. The classification heads (1 hidden layer MLP) will be trained whilst SBERT is finetuned as one backward pass updates both the main encoder and classification heads.

		L{total} = bias_loss + (alpha) * opinion_loss + regularization terms(L2 & Dropout)

(alpha serves as a weighting parameter. Setting at 0.3 for first run)
(Dropout at 0.2 and L2 weight decay 0.01)
