import random
import util
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1
        self.features = []
        self.result = []
        self.numberOfLabel = []
        self.pOfLabel = []

    def setSmoothing(self, k):
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = self.extractFeatures(trainingData)
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def extractFeatures(self, data):
        return list(set(f for datum in data for f in datum.keys()))

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        featureProbabilities = self.calculateFeatureProbabilities(trainingData)
        self.result, self.numberOfLabel = self.trainOnLabels(trainingData, trainingLabels)
        self.pOfLabel = self.calculateLabelProbabilities(validationLabels)
        self.k = self.tuneHyperparameter(validationData, validationLabels, kgrid)

    def calculateFeatureProbabilities(self, data):
        countOfLabel = util.Counter()
        for datum in data:
            for key, value in datum.items():
                countOfLabel[key] += value == 0
        return {key: count / len(data) for key, count in countOfLabel.items()}

    def trainOnLabels(self, trainingData, trainingLabels):
        labelDataCounts = [util.Counter() for _ in self.legalLabels]
        labelCounts = [0] * len(self.legalLabels)

        for i, label in enumerate(trainingLabels):
            label = int(label)  # Ensure label is an integer
            labelCounts[label] += 1
            for key, value in trainingData[i].items():
                labelDataCounts[label][key] += value == 0

        return labelDataCounts, labelCounts

    def calculateLabelProbabilities(self, labels):
        labelProb = [0] * len(self.legalLabels)
        total = len(labels)
        for label in labels:
            labelProb[int(label)] += 1
        return [count / total for count in labelProb]

    def tuneHyperparameter(self, validationData, validationLabels, kgrid):
        bestK = self.k
        bestAccuracy = 0

        for k in kgrid:
            accuracy = self.testHyperparameter(validationData, validationLabels, k)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestK = k

        return bestK

    def testHyperparameter(self, validationData, validationLabels, k):
        correct = 0
        for i, datum in enumerate(validationData):
            posterior = self.calculateLogJointProbabilities(datum, k)
            if posterior.argMax() == validationLabels[i]:
                correct += 1
        return correct / len(validationLabels)

    def classify(self, testData):
        guesses = []
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum, self.k)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum, k):
        logJoint = util.Counter()
        for label in self.legalLabels:
            logValue = math.log(self.pOfLabel[label])
            for key, value in datum.items():
                prob = self.calculateProbability(value, label, key, k)
                logValue += math.log(prob)
            logJoint[label] = logValue
        return logJoint

    def calculateProbability(self, value, label, key, k):
        if value == 0:
            return (self.result[label][key] + k) / (self.numberOfLabel[label] + 2 * k)
        else:
            return ((self.numberOfLabel[label] - self.result[label][key]) + k) / (self.numberOfLabel[label] + 2 * k)
