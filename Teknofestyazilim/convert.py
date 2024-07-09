from PIL import Image
import os

root = r"/Users/Esra/Tekofestyazilim/converted/test"
print(root)
count = 0
for dirs, subdir, files in os.walk(root):
    for file in files:
        lastChar = file[-4:]
        if(lastChar == 'jfif'):
            img = Image.open(root+'/'+file)
            #file ends in .jfif, remove 4 characters
            fileName = file[:-4]
            print(fileName)
            #add jpg and save
            newPath = "./converted/"+fileName + "jpg"
            if (os.path.exists(newPath)):
            	continue
            img.save(newPath)



            
