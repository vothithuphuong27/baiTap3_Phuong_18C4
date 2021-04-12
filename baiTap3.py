import math
from collections import Counter
import numpy as np

class Point:
    value = []
    label = None
    distance = None

    def __init__(self, value, label):
        self.value = value
        self.label = label
    # tính toán khoảng cách sử dụng công thức euclid

    def euclidean_distance(self, test):
        acc = 0  # dùng để lưu trữ tổng
        for i in range(0, len(self.value)):
            acc += math.pow(test.value[i] - self.value[i], 2)  # tính tổng
        self.distance = math.sqrt(acc)  # lưu trữ giá trị khoảng cách

ex_2_dataset = np.array([
    Point([4, 3], 1),
    Point([3, 7], 1),
    Point([7, 4], 1),
    Point([4, 1], 1),
    Point([6, 5], 1),
    Point([5, 6], 1),
    Point([3, 7], 1),
    Point([6, 2], 1),
    Point([4, 6], 0),
    Point([4, 4], 0),
    Point([5, 8], 0),
    Point([7, 8], 0),
    Point([7, 6], 0),
    Point([4, 10], 0),
    Point([9, 7], 0),
    Point([5, 4], 0),
    Point([8, 5], 0),
    Point([6, 6], 0),
    Point([7, 4], 0),
    Point([8, 8], 0),
])

def generate_dataset(data):
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    l = len(data_copy)
    return {
        "training": data_copy[:math.floor(l * 80 / 100)],
        "testing": data_copy[math.floor(l * 80 / 100):]
    }

    # huấn luyện mô hình
def training(data, test):
    data_copy = data.copy()
    for point in data_copy:
        point.euclidean_distance(test)
    return data_copy

# tìm ra k láng giềng gần nhất
def get_k_nearest_neighbors(data, k):
    data_copy = data.copy()  # sao chép mảng tránh ghi đè
    # sắp xếp mảng tăng dần theo khoảng cách
    dt = data_copy.tolist()
    dt.sort(key=lambda x: x.distance)
    return np.array(dt[:k])  # trả về k phần tử đầu tiên của mảng

# tìm ra nhãn dự đoán
def get_predicted_label(data):
    labels = []  # mảng lưu trữ những nhãn xuất hiện trong dữ liệu
    for point in data:
        labels.append(point.label)  # thêm nhãn vào mảng
    # sử dụng Counter để đếm số lần xuất hiện của nhãn
    counted = Counter(labels)
    common = counted.most_common()  # trả vễ nhãn có số lần xuất hiện nhiều nhất
    if(common):  # kiểm tra sự tồn tại nhãn xuất hiện nhiều nhất tránh trường hợp mảng rỗng
        return common[0][0]  # trả vễ nhãn có số lần xuất hiện nhiều nhất
    return None  # nếu mảng rỗng trả về None


def get_accuracy(cm, len):
    n = 0
    for i in range(cm.shape[0]):
        n += cm[i][i]
    return n/len


def normalized_confusion_matrix(cm):
    return cm/cm.sum(axis=1, keepdims=True)


def get_unique_label_length(data):
    return np.unique(list(map(lambda x: x.label, data))).shape[0]

def main_ex_3():
    vals = read_file("./iris.dat", dtype=float)
    data = generate_dataset(vals)
    # chuyển sang Point để tính toán
    training_dataset = np.array(list(map(lambda x: Point(
        x[:4], get_label_from_array(x[-3:])), data.get("training"))))
    testing_dataset = np.array(list(map(lambda x: Point(
        x[:4], get_label_from_array(x[-3:])), data.get("testing"))))

    real_label = []
    predicted_label = []
    for test in testing_dataset:
        real_label.append(test.label)
        trained_data = training(training_dataset, test)
        knn = get_k_nearest_neigbors(trained_data, k=20)
        label = get_predicted_label(knn)
        predicted_label.append(label)
    cm = np.zeros((3, 3), dtype=int)
    for i in range(len(real_label)):
        cm[real_label[i]][predicted_label[i]] += 1
    print("Confusion matrix: ")
    print(cm)
    print("Accuracy is: ", np.diagonal(cm).sum()/cm.sum())

main_ex_3()
