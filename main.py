
import os
import numpy as np
from flask import Flask, request, jsonify
from logging import FileHandler,WARNING
import os
from PIL import Image
import json
import urllib.request
import pyrebase
import shutil

def number_to_label(number):
  number_list = ['0', '1', '101', '102', '103', '104', '11', '12', '13', '14', '15', '16', '18', '21', '22', '23', '24', '25', '26', '27', '3', '37', '38', '39', '45', '46', '47', '48', '49', '50', '51', '58', '66', '67', '69', '70', '76', '77', '78', '86', 'pest']

  label_list = ['rice leaf roller', 
    'rice leaf caterpillar', 
    'paddy stem maggot', 
    'asiatic rice borer', 
    'yellow rice borer', 
    'rice gall midge', 
    'Rice Stemfly', 
    'brown plant hopper', 
    'white backed plant hopper', 
    'small brown plant hopper', 
    'rice water weevil', 
    'rice leafhopper', 
    'grain spreader thrips', 
    'rice shell pest', 
    'grub', 
    'mole cricket', 
    'wireworm', 
    'white margined moth', 
    'black cutworm', 
    'large cutworm', 
    'yellow cutworm', 
    'red spider', 
    'corn borer', 
    'army worm', 
    'aphids', 
    'Potosiabre vitarsis', 
    'peach borer', 
    'english grain aphid', 
    'green bug', 
    'bird cherry-oataphid', 
    'wheat blossom midge', 
    'penthaleus major', 
    'longlegged spider mite', 
    'wheat phloeothrips', 
    'wheat sawfly', 
    'cerodonta denticornis', 
    'beet fly', 
    'flea beetle', 
    'cabbage army worm', 
    'beet army worm', 
    'Beet spot flies', 
    'meadow moth', 
    'beet weevil', 
    'sericaorient alismots chulsky', 
    'alfalfa weevil', 
    'flax budworm', 
    'alfalfa plant bug', 
    'tarnished plant bug', 
    'Locustoidea', 
    'lytta polita', 
    'legume blister beetle', 
    'blister beetle', 
    'therioaphis maculata Buckton', 
    'odontothrips loti', 
    'Thrips', 
    'alfalfa seed chalcid', 
    'Pieris canidia', 
    'Apolygus lucorum', 
    'Limacodidae', 
    'Viteus vitifoliae', 
    'Colomerus vitis', 
    'Brevipoalpus lewisi McGregor', 
    'oides decempunctata', 
    'Polyphagotars onemus latus', 
    'Pseudococcus comstocki Kuwana', 
    'parathrene regalis', 
    'Ampelophaga', 
    'Lycorma delicatula', 
    'Xylotrechus', 
    'Cicadella viridis', 
    'Miridae', 
    'Trialeurodes vaporariorum', 
    'Erythroneura apicalis', 
    'Papilio xuthus', 
    'Panonchus citri McGregor', 
    'Phyllocoptes oleiverus ashmead', 
    'Icerya purchasi Maskell', 
    'Unaspis yanonensis', 
    'Ceroplastes rubens', 
    'Chrysomphalus aonidum', 
    'Parlatoria zizyphus Lucus', 
    'Nipaecoccus vastalor', 
    'Aleurocanthus spiniferus', 
    'Tetradacus c Bactrocera minax', 
    'Dacus dorsalis(Hendel)', 
    'Bactrocera tsuneonis', 
    'Prodenia litura', 
    'Adristyrannus', 
    'Phyllocnistis citrella Stainton', 
    'Toxoptera citricidus', 
    'Toxoptera aurantii', 
    'Aphis citricola Vander Goot', 
    'Scirtothrips dorsalis Hood', 
    'Dasineura sp', 
    'Lawana imitata Melichar', 
    'Salurnis marginella Guerr', 
    'Deporaus marginatus Pascoe', 
    'Chlumetia transversa', 
    'Mango flat beak leafhopper', 
    'Rhytidodera bowrinii white', 
    'Sternochetus frigidus', 
    'Cicadellidae', 
    'Phytoseiulus', 
    'Whiteflie', 
    'Spider Mite'
    ]
  try:
    returned = label_list[int(number_list[number])]
  except:
    returned = 'pest' 
  return returned

def get_firebase_url(config, local_path, fileName):
  
  firebase = pyrebase.initialize_app(config)
  storage = firebase.storage()
  path_on_cloud = "images/" + fileName
  storage.child(path_on_cloud).put(local_path)
  url = storage.child(path_on_cloud).get_url(None)

  return url

app = Flask(__name__)
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

config = {
  "apiKey" : "AIzaSyDkWIZ7dD_mCGGtQLOU3p54Y4YYLFxS5j4",
  "authDomain" : "pestdetectionfyp.firebaseapp.com",
  "databaseURL" : "https://pestdetectionfyp.firebaseio.com",
  "projectId" : "pestdetectionfyp",
  "storageBucket" : "pestdetectionfyp.appspot.com",
  "messagingSenderId" : "607522976371",
  "appId" : "1:607522976371:web:053bbde0545963a349f006"
}

@app.route('/')
def home_endpoint():
  return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
  if request.method == 'POST':
    data = request.get_json(force = True)
    identity = data["id"]
    img_link = data['link']
    
    path_to_image = 'temps/img' + str(identity) + '.jpg'
    urllib.request.urlretrieve(img_link, path_to_image)

    os.system('python detect.py --source ' + path_to_image + ' --weights best_mAP_0.653.pt --img 640 --conf 0.1 --save-txt --save-conf --hide-label --hide-conf')

    url = get_firebase_url(config, "runs/detect/exp/" + 'img' + str(identity) + '.jpg', 'img' + str(identity) + '.jpg')

    with open('runs/detect/exp/labels/' + 'img' + str(identity) + '.txt', 'r') as f:
      conf_list = [float(line.split(' ')[-1][:-2]) for line in f]
    
    conf = max(conf_list)
    index = conf_list.index(conf)
    with open('runs/detect/exp/labels/' + 'img' + str(identity) + '.txt', 'r') as f:
      label_list = [int(line.split(' ')[0]) for line in f]
    
    
    if not label_list:
      label = 'null'
    else:
      label = number_to_label(label_list[index])

    f = {'link': url, 'id': str(identity), 'label': label, 'conf': conf}
    shutil.rmtree("runs/detect/exp")

    return jsonify(f)

  else:
    return "Hello World!"

if __name__ == '__main__':
    app.run()

