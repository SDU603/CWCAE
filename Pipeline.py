import os
from itertools import cycle
import numpy
import torch
import torch.nn as nn
import Models

GPU: bool = torch.cuda.is_available()


def trainAE(dataset, epochs=200, printInterval=10, saveInterval=200, saveIndex=0):
    dataloader = dataset.getLoader(batchSize=20)
    model = Models.CWCAE(512, 256, 32)
    costFunc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    if GPU:
        model.cuda()
        costFunc.cuda()
    for epoch in range(1, epochs + 1):
        model.train()
        meanLoss = []
        for (feature, _) in dataloader:
            if GPU:
                (feature, _) = (feature.cuda(), _.cuda())
            feature = feature.reshape(feature.shape[0], 1, -1)
            latent, featureRecon = model(feature)
            loss = costFunc(feature, featureRecon)
            meanLoss.append(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad()
        meanLoss = numpy.mean(meanLoss)
        if epoch % printInterval == 0:
            print('Epoch {} completed. Mean Loss: {:.3f}'.format(epoch, meanLoss))
        if epoch % saveInterval == 0:
            folder = './result/model/cwcae/'
            name = '' + str(saveIndex) + '.pkl'
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(model.state_dict(), folder + name)
    return model


def trainClassifier(dataset, model, classifier, epochs=1000, printInterval=10):
    dataloader = dataset.getLoader(batchSize=20)
    model.eval()
    costFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())
    if GPU:
        classifier.cuda()
        costFunc.cuda()
    for epoch in range(1, epochs + 1):
        classifier.train()
        correctCount, lossCount = 0, 0
        for (feature, label) in dataloader:
            if GPU:
                (feature, label) = (feature.cuda(), label.cuda())
            feature = feature.reshape(feature.shape[0], 1, -1)
            modelInputs = model.encode(feature).detach()
            modelOutputs = classifier(modelInputs)
            loss = costFunc(modelOutputs, label)
            loss.backward()
            optimizer.step()
            classifier.zero_grad()
            (_, outputType) = torch.max(modelOutputs.data, 1)
            correctCount += torch.eq(outputType, label).sum().item()
            lossCount += loss
        if epoch % printInterval == 0:
            print('epoch {} correct rate: {:.3f}%'.format(epoch, correctCount / len(dataset) * 100))
    return classifier


def modifyAE(dataset1, dataset2, model, rounds=10, threshold=0.9, saveIndex=0):
    dataloader1 = dataset1.getLoader(batchSize=20)
    dataloader2 = dataset2.getLoader(batchSize=20)
    classifier = Models.Classifier(outputNum=6)
    mseLoss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if GPU:
        model.cuda()
        mseLoss.cuda()
    classifier = trainClassifier(model, classifier, dataloader2, epochs=2000, printInterval=500)
    classifier.eval()
    calcAccuracy(AEModel, classifier, testDataset)
    cycledL = cycle(dataloader2)
    for r in range(rounds):
        print('Round {}:'.format(r))
        correctCount = 0
        meanLoss1, meanLoss2, pCount = 0, 0, 0
        for (batchL, label), (batchU, unseenLabel) in zip(cycledL, dataloader1):
            if batchL.shape[0] != 20 or batchU.shape[0] != 20:
                continue
            if GPU:
                (batchL, label, batchU, unseenLabel) = (batchL.cuda(), label.cuda(), batchU.cuda(), unseenLabel.cuda())
            model.train()
            batchL = batchL.reshape(batchL.shape[0], 1, -1)
            batchU = batchU.reshape(batchU.shape[0], 1, -1)
            latentL, _ = model(batchL)
            latentU, featureRecon = model(batchU)
            pLabel, pValue = classifier.calcPseudoLabel(latentU)
            reconLoss = mseLoss(batchU, featureRecon)
            batchP, dmLossTotal = Models.contraLoss(latentL, latentU, label, pLabel, pValue, threshold=threshold, dMax=12)
            if batchP > 0:
                dmLoss = dmLossTotal / batchP
            else:
                dmLoss = torch.tensor(0.).cuda()
            correctCount += torch.eq(pLabel, unseenLabel).sum().item()
            meanLoss1 += reconLoss
            meanLoss2 += dmLossTotal
            pCount += batchP
            loss = reconLoss + dmLoss
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if threshold < 0.99:
            threshold += 0.01
        meanLoss1 = meanLoss1 / len(dataset1)
        meanLoss2 = meanLoss2 / pCount
        acc = 100. * correctCount / len(dataset1)
        print('Mean Loss: {:.3f} + {:.3f}. Acc: {:.2f}%'
              .format(meanLoss1, meanLoss2, acc))
        classifier = trainClassifier(model, classifier, dataloader2, epochs=500, printInterval=100)
        classifier.eval()
        calcAccuracy(AEModel, classifier, testDataset)
    folder = './result/model/cwcae/'
    name = str(saveIndex) + '_c.pkl'
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), folder + name)
    return model, classifier


def calcAccuracy(model, classifier, dataset):
    dataloader = dataset.getLoader(batchSize=20)
    model.eval()
    classifier.eval()
    correctCount = 0
    for (feature, label) in dataloader:
        if GPU:
            (feature, label) = (feature.cuda(), label.cuda())
        feature = feature.reshape(feature.shape[0], 1, -1)
        modelInputs = model.encode(feature).detach()
        modelOutputs = classifier(modelInputs)
        (_, outputType) = torch.max(modelOutputs.data, 1)
        correctCount += torch.eq(outputType, label).sum().item()
    print('test correct rate: {:.3f}%'.format(correctCount / len(dataset) * 100))


def drawTSNE(model, _dataLabeled, _dataUnlabeled):
    featureList, labelList = [], []
    for f, l in _dataLabeled:
        for i in range(f.shape[0]):
            tensor = torch.Tensor(f[i]).cuda()
            tensor = tensor.reshape(1, 1, -1)
            latentF = model.encode(tensor).squeeze().detach().cpu().tolist()
            featureList.append(latentF)
            labelList.append(10)
    for f, l in _dataUnlabeled:
        for i in range(f.shape[0]):
            tensor = torch.Tensor(f[i]).cuda()
            tensor = tensor.reshape(1, 1, -1)
            latentF = model.encode(tensor).squeeze().detach().cpu().tolist()
            featureList.append(latentF)
            labelList.append(l)
    # TSNE.execute(featureList, labelList)


if __name__ == '__main__':

    # Load Datas
    dataLabeled = []
    dataUnlabeled = []
    dataTest = []

    # Generate Datasets
    labeledDataset = []  # From dataLabeled
    unlabeledDataset = []  # From dataUnlabeled
    fullDataset = []  # From merge(dataLabeled, dataUnlabeled)
    testDataset = []  # From dataTest

    for times in range(10):
        print(times)
        AEModel = trainAE(fullDataset, saveIndex=times, printInterval=100)
        AEModel, classifierModel = modifyAE(unlabeledDataset, labeledDataset, AEModel, saveIndex=times)
        calcAccuracy(AEModel, classifierModel, testDataset)
