import random
import util

class PerceptronClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.maxIteration = maxIterations
        self.weights = {}
        self.weights = {label: util.Counter() for label in self.legalLabels}

    def setWeight(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        learningRate = 1
        self.features = trainingData[0].keys()
        if not self.weights:
            for label in self.legalLabels:
                self.weights[label][0] = 0.1  
                for key in self.features:
                    self.weights[label][key] = 0.5  

        bestWeights = {}
        bestAccuracy = 0

        for iteration in range(self.maxIteration):
            print(f"\t\tStarting iteration {iteration}...", end="")
            allPassFlag = True
            for i, data in enumerate(trainingData):
                results = {label: self.weights[label] * data + self.weights[label][0] for label in self.legalLabels}
                predictedLabel = max(results, key=results.get)
                actualLabel = int(trainingLabels[i])
                if predictedLabel != actualLabel:
                    self.weights[predictedLabel] -= data
                    self.weights[predictedLabel][0] -= learningRate
                    self.weights[actualLabel] += data
                    self.weights[actualLabel][0] += learningRate
                    allPassFlag = False
            accuracy = self.validate(validationData, validationLabels)
            if accuracy > bestAccuracy:
                bestWeights = self.weights.copy()
                bestAccuracy = accuracy
           

            print("\033[1;32mDone!\033[0m")

        self.weights = bestWeights

    def validate(self, validationData, validationLabels):
        guesses = self.classify(validationData)
        correct = sum(1 for guess, label in zip(guesses, validationLabels) if guess == int(label))
        return correct / len(validationLabels)

    def classify(self, data):
        guesses = []
        for datum in data:
            scores = {label: datum * self.weights[label] + self.weights[label][0] for label in self.legalLabels}
            guesses.append(max(scores, key=scores.get))
        return guesses

    def findHighWeightFeatures(self, label, weightNum: int):
        sortedItems = sorted(self.weights[label].items(), key=lambda item: abs(item[1]), reverse=True)
        featuresWeights = [item[0] for item in sortedItems if isinstance(item[0], tuple)][:weightNum]
        return featuresWeights
