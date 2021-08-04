
#%%
from flask import abort, Flask, jsonify, request
from waitress import serve
from sequence_tagger_model_KD import SequenceTagger
from flair.data import Sentence
import sys
#%%
if __name__ == '__main__':
   path = './models/'
   snapshot = 'ner_KD_char_25k_30ep' 

   app = Flask(__name__)
   model = SequenceTagger.load(path+snapshot+'.pt')

   @app.route('/', methods = ['POST'])

   def ner_service():
      if not request.json or not 'message' in request.json:
         abort(400)
      message = request.json['message']
      
      from flair.data import Sentence
      message = Sentence(message)
      model.predict(message)
      entities = list(message.get_spans('ner'))
      
      result = {}
      for span in entities:
         result[span.to_original_text()] = str(span.labels[0])
      return result

   app.run(host='0.0.0.0', port= 9002)
   # serve(app, host='0.0.0.0', port= 9002)
