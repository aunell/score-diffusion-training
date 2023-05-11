import numpy as np
import matplotlib.pyplot as plt

epoch= 134
# Read the data from the txt file
with open('trainLoss/'+str(epoch)+'meanSigmas.txt') as f:
    dataMean = f.readlines()
with open('trainLoss/'+str(epoch)+'maxSigma.txt') as f:
    dataMax = f.readlines()
with open('trainLoss/'+str(epoch)+'medSigma.txt') as f:
    dataMed = f.readlines()
with open('trainLoss/'+str(epoch)+'minSigma.txt') as f:
    dataMin = f.readlines()
# Convert the data to a NumPy array of floats
dataMean = np.array([float(x.strip()) for x in dataMean])
dataMax = np.array([float(x.strip()) for x in dataMax])
dataMed = np.array([float(x.strip()) for x in dataMed])
dataMin = np.array([float(x.strip()) for x in dataMin])
# Create a line chart using Matplotlib
plt.plot(dataMean, label='Mean')
plt.plot(dataMax, label= 'Low Noise')
plt.plot(dataMed, label = 'Median Noise')
plt.plot(dataMin, label = 'High Noise')

plt.xlabel('Training (Epoch)')
plt.ylabel('Loss')
plt.title('Loss by Training for Different Noise Levels')
plt.legend()

plt.savefig('line_chart.png')
