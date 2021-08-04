# distillNER

KD_dataprep.py : read news data and conllu data, use flair-ner-large to tag data with soft-label, and save as training set as pickle file.

trainer_KD_NER.py: train the model 

KD_eval.ipynb: evaluation

KD_plots.ipynb: visualization of training process(training loss, dev loss and dev scores)

flask_server.py: for empirical sentence test 

*example: curl -X POST "172.30.0.140:9002" -H 'content-type: application/json' --data '{ "message":"Die Mitarbeiter der Internetriesen Google und Facebook in den USA müssen sich vor einer Rückkehr in die Büros gegen das Coronavirus impfen lassen. Das teilten die Unternehmen unabhängig voneinander mit.." }'*
