import os

# inputRoot = '/media/linse/dados/stevan/datasets/heart-sound/training/'
# outputRoot = '/media/linse/dados/stevan/datasets/heart-sound/segmentation/'
inputRoot = '/home/linse/Stevan/datasets/heart-sound/training/'
outputRoot = '/home/linse/Stevan/datasets/heart-sound/segmentation/'

command = 'rm -rf /home/linse/Stevan/datasets/heart-sound/segmentation/'
os.system(command)
os.mkdir(outputRoot)

folderList = os.listdir(inputRoot)
folderList.sort() # Coloca a lista em ordem alfabética
folderList = folderList[2:] # remove os metadados da lista

# 'training-a/', 'training-b/', 'training-c/', 'training-d/', 'training-e/', 'training-f/'
for i in range(len(folderList)):
  fname = folderList[i]
  inFolder = os.path.join(inputRoot, fname)
  outFolder = os.path.join(outputRoot, fname)
  os.mkdir(outFolder)
  fileList = [ fname for fname in os.listdir(inFolder) if fname.endswith('.wav')]
  fileList.sort()
  for k in range(len(fileList)):
    audioFile = os.path.join(inFolder,fileList[k])
    outFile = os.path.join(outFolder,fileList[k].replace('wav','mat'))
    print(folderList[i], "| File ", k, " of ", len(fileList), " | ", fileList[k].replace('wav','mat'))