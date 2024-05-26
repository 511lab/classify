import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
classes = ('uncultivated', 'cultivated')
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("resnet101_model.pth")
model.eval()
model.to(DEVICE)
dataset_test = datasets.ImageFolder(r'C:\Users\Administrator\PycharmProjects\pythonProject\de\test_set', transform_test)
print(len(dataset_test))
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)
# 对应文件夹的label
correct = 0
total = len(dataset_test)
predictions = []
true_labels = []
with torch.no_grad():
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        output = model(data)
        _, pred = torch.max(output, 1)
        predictions.append(pred.item())
        true_labels.append(label.item())
        if pred == label:
            correct += 1
# 计算评价指标
confusion_mat = confusion_matrix(true_labels, predictions)
accuracy = correct / total
oa = accuracy_score(true_labels, predictions)
aa = (confusion_mat.diagonal().sum() / confusion_mat.sum()).mean()
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')

print('Accuracy: {:.1f}%'.format(100 * accuracy))
print('Overall Accuracy (OA): {:.1f}%'.format(100 * oa))
print('Average Accuracy (AA): {:.1f}%'.format(100 * aa))
print('Macro Recall Score: {:.1f}%'.format(100 * recall))
print('Macro F1 Score: {:.1f}%'.format(100 * f1))


metrics = [accuracy, oa, aa,recall,f1]
# 打印混淆矩阵
print("Confusion Matrix:")
print(confusion_mat)