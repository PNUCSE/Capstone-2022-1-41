import torch
import torch.nn as nn
from lstm_model.dataset.FallDataSet import FallDataset
from lstm_model.model.RNN import RNN

# 참고 코드 : https://velog.io/@sjinu/Pytorch-Implementation-code

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 1
input_size = 34
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 50
num_epochs = 100
learning_rate = 0.001

train_dataset = FallDataset('lstm_model/dataset/learn.v.1.1.csv')
test_dataset = FallDataset('lstm_model/dataset/validation.v.1.1.csv')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=False)

if __name__ == '__main__':
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            points = x.reshape(-1, sequence_length, input_size).to(device)
            labels = y.reshape(-1).to(device)

            outputs = model(points)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch [{}/{}], step [{}/{}], Loss: {:.4f}".format(
                epoch+1, num_epochs, i+1, total_step, loss.item()))

    model.eval()  # Dropout, Batchnorm 등 실행 x
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            points = x.reshape(-1, sequence_length, input_size).to(device)
            labels = y.reshape(-1).to(device)
            outputs = model(points)
            _, predicted = torch.max(outputs, 1)  # logit(확률)이 가장 큰 class index 반환
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test: {} %'.format(100 * correct / total))

    # 모델의 state_dict 출력
    # 출처 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # save model
    save_path = './fall_detect.pth'
    torch.save(model.state_dict(), save_path)