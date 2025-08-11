#to help create the dataset
import os, random
from PIL import Image
a=[]
for folder in os.listdir("dataset/train/"):
  #create 16 test images for each category
  # os.mkdir(f"dataset/test/{folder}/")
  # for i in range(16):
  #   random_file= random.choice(os.listdir(f"dataset/train/{folder}/"))
  #   os.rename(f"dataset/train/{folder}/{random_file}", f"dataset/test/{folder}/{i}.jpg")

  #check size of each folder
  if (len(os.listdir(f"dataset/train/{folder}/")) != 64):
    print(f"Warning: {folder} has {len(os.listdir(f'dataset/train/{folder}/'))} images, expected 64.")

  #resize images to 224x224
  for file in os.listdir(f"dataset/train/{folder}/"):
    with Image.open(f"dataset/train/{folder}/{file}") as im:
      (width, height) = (224,224)
      resized_img = im.resize((width, height))
      resized_img.save(f'dataset/train/{folder}/{file}')

  a.append(str(folder))

#check size of test folders
for folder in os.listdir("dataset/test/"):
  if (len(os.listdir(f"dataset/test/{folder}/")) != 16):
    print(f"Warning: {folder} has {len(os.listdir(f'dataset/test/{folder}/'))} images, expected 16.")

a.sort()
print(a) 
print("size:", len(a))