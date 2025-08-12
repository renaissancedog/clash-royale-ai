import tensorflow as tf
import pandas as pd
import os, random
from tensorflow.keras.preprocessing.image import load_img
class_names=['archer queen', 'arrows', 'baby dragon', 'balloon', 'bandit', 'barbarian barrel', 'barbarian hut', 'battle healer', 'battle ram', 'bomb tower', 'bowler', 'cannon', 'cannon cart', 'clone', 'dark prince', 'dart goblin', 'earthquake', 'electro dragon', 'electro giant', 'electro spirit', 'electro wizard', 'elite barbarians', 'elixir collector', 'elixir golem', 'executioner', 'fire spirit', 'fireball', 'fisherman', 'flying machine', 'freeze', 'furnace', 'giant', 'giant skeleton', 'giant snowball', 'goblin barrel', 'goblin cage', 'goblin drill', 'goblin gang', 'goblin giant', 'goblin hut', 'goblins', 'golden knight', 'golem', 'graveyard', 'guards', 'heal spirit', 'hog rider', 'hunter', 'ice golem', 'ice wizard', 'inferno dragon', 'inferno tower', 'lava hound', 'lightning', 'little prince', 'lumberjack', 'magic archer', 'mega knight', 'mega minion', 'mighty miner', 'miner', 'mini pekka', 'minion horde', 'minions', 'monk', 'mother witch', 'musketeer', 'night witch', 'pekka', 'phoenix', 'poison', 'prince', 'princess', 'rage', 'ram rider', 'rascals', 'rocket', 'royal delivery', 'royal ghost', 'royal hogs', 'skeleton army', 'skeleton barrel', 'skeleton dragons', 'skeleton king', 'sparky', 'spear goblins', 'the log', 'three musketeers', 'tombstone', 'tornado', 'witch', 'wizard', 'xbow', 'zappies']
class_names=pd.DataFrame(class_names)
folder_paths = class_names[0].apply(lambda x: f"dataset/train/{x}/")
# folder_paths = ["cr/dataset/train/golem/"] for custom folder paths
model = tf.keras.models.load_model("model.keras")
count=0
for folder in folder_paths:
  image_path=folder+(random.choice(os.listdir(folder)))
  image = load_img(image_path, target_size=(224, 224))
  image = tf.expand_dims(image, axis=0)
  predictions = model.predict(image)
  predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
  print("Actual class:", folder.split('/')[-2])
  print(f"Predicted class: {class_names[0][predicted_class]}", "with confidence:", predictions[0][predicted_class])
  if (predicted_class == class_names[0].tolist().index(folder.split('/')[-2])):
    print("Prediction is correct!")
    count+=1
  print(f"Predictions: {predictions[0]}")
print(count)