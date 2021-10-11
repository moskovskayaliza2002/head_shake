import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import pandas as pd
import os
import time
import torch.nn as nn
from buffer_less_video_capture import VideoCapture
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms as transforms


class Model(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers)
        # Randomly zeroes some of the elements of the input tensor with probability p
        self.drop = nn.Dropout(p=0.5)
        # Linear transformation to a tensor of another shape
        self.fc = nn.Linear(hidden_size, output_size)        
        
    def init_hidden(self):
        return (torch.autograd.Variable(torch.randn(1, 1, self.hidden_size)),
                torch.autograd.Variable(torch.randn(1, 1, self.hidden_size)))
        
    def forward(self, input):
        
        output, _ = self.lstm(input)
        output = self.drop(output)
        output = self.fc(output)
        # Returns a tensor with all the dimensions of input of size 1 removed.
        output = torch.squeeze(output, 1)
        output = torch.sigmoid(output)
        
        return output
    

def generator(n): # n - number of actions
    count = 1
    class_list = []
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = os.path.join("..", "data", "shaking_data", time_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    flag_of_class = False
    vid = VideoCapture(-0)
    ret = True
    if ret == True:
        for i in range(n):
            type_of_action = np.random.choice([True, False])

            if type_of_action:
                my_time = np.random.randint(6, 12)
                for j in range(my_time, 0, -1):
                    frame = vid.read()
                    frame = cv2.resize(frame, (640, 480))
                    print("shake your head for another {} seconds".format(j))
                    cv2.imwrite(dir + "/Frame" + str(count) + '.png', frame)
                    if (j > my_time - 3) and (flag_of_class == True):
                        class_list.append(1)
                    else:
                        class_list.append(0)
                        if j == 1:
                            flag_of_class = True
                    time.sleep(0.25)
                    count += 1
            else:
                my_time = np.random.randint(6, 12)
                for j in range(my_time, 0, -1):
                    frame = vid.read()
                    frame = cv2.resize(frame, (640, 480))
                    print("do not shake your head for another {} seconds".format(j))
                    cv2.imwrite(dir + "/Frame" + str(count) + '.png', frame)
                    if (j > my_time - 3) and (flag_of_class == True):
                        class_list.append(1)
                    else:
                        class_list.append(0)
                        flag_of_class = False
                    time.sleep(0.25)
                    count += 1
            time.sleep(2)
    df = pd.DataFrame(np.array(class_list).T)
    df.to_csv("../data/" + "shaking_data/" + time_str + "/class_array.csv", header= None, index=False)
    return class_list, time_str


def train(data_folder, show_output = False):
    # initialize the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model on to the computation device and set to eval mode
    model.to(device).eval()
    output = []
    data_folder = [data_folder]
    for i in range(len(data_folder)):
        print(f"Data {i} from {len(data_folder)}")
        data = pd.read_csv("../data/shaking_data/" + data_folder[i] + "/class_array.csv", header = None)
        for im in range(len(data)):
            print(f"\tImage {im} from {len(data)}")
            image_path = "../data/shaking_data/" + data_folder[i] + "/Frame" + str(im+1) + ".png"
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image = Image.open(image_path).convert('RGB')
            # NumPy copy of the image for OpenCV functions
            orig_numpy = np.array(image, dtype=np.float32)
            # convert the NumPy image to OpenCV BGR format
            orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
            # transform the image
            image = transform(image)
            # add a batch dimension
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image)
            # selection of coordinates of five points and local maximums         
            best_perf_ind = np.argmax(outputs[0]['scores'].detach().cpu().numpy())
            nose = outputs[0]['keypoints'][best_perf_ind][0].cpu().numpy()
            left_eye = outputs[0]['keypoints'][best_perf_ind][1].cpu().numpy()
            right_eye = outputs[0]['keypoints'][best_perf_ind][2].cpu().numpy()
            left_ear = outputs[0]['keypoints'][best_perf_ind][3].cpu().numpy()
            right_ear = outputs[0]['keypoints'][best_perf_ind][4].cpu().numpy()
            H = orig_numpy.shape[0]
            W = orig_numpy.shape[1]
            # normalization function
            def norm(val, maxi):
                return (val - maxi/2)/maxi
            # list of normalized values ​​of one image
            vector = [norm(nose[0], W), norm(nose[1], H), norm(left_eye[0], W), norm(left_eye[1], H), norm(right_eye[0], W), norm(right_eye[1], H), norm(left_ear[0], W), norm(left_ear[1], H), norm(right_ear[0], W), norm(right_ear[1], H)]
            output.append(vector)
            # display of key points in the picture
            if show_output:
                frame = cv2.imread(image_path)
                print(type(frame))
                key_image = cv2.circle(frame, (nose[0],nose[1]), radius=5, color=(255, 0, 0), thickness=-1)
                key_image = cv2.circle(frame, (left_eye[0],left_eye[1]), radius=5, color=(255, 0, 0), thickness=-1)
                key_image = cv2.circle(frame, (right_eye[0],right_eye[1]), radius=5, color=(255, 0, 0), thickness=-1)
                key_image = cv2.circle(frame, (left_ear[0],left_ear[1]), radius=5, color=(255, 0, 0), thickness=-1) 
                key_image = cv2.circle(frame, (right_ear[0],right_ear[1]), radius=5, color=(255, 0, 0), thickness=-1)
                cv2.imshow('keypoints', key_image)
                cv2.waitKey(1000) 
                
    if show_output:
        cv2.destroyAllWindows() 
    # preservation of learning outcomes
    df = pd.DataFrame(np.array(output))
    df.to_csv("../data/" + "shaking_data/" + data_folder[i] + "/output.csv", header= None, index=False)
    return torch.tensor([output])


if __name__ == '__main__':
    
    # specify the number of actions
    N = 30
    # get the folder name and classification list
    classified, data_folder = generator(N)
    # get processed key points from images
    train(data_folder)
    # we divide the dataset into train and test dataset in a ratio of 80 to 20
    train_classified = classified[:200]
    test_classified = classified[200:]
    df = pd.read_csv("../data/shaking_data/" + data_folder + "/output.csv", header = None, delimiter=',')
    list_data = [list(row) for row in df.values]
    data = torch.tensor([list_data])
    train_data, test_data = torch.split(data, [200, 52], dim = 1)
    
    # set the parameters of the second model and initialize
    input_size = 10 
    output_size = 1
    hidden_size = 32    
    num_layers = 2
    model_two = Model(input_size, output_size, hidden_size, num_layers) 
    num_epoch = 500
    lr = 0.1
    
    
    device = torch.device("cpu")
    # creation of an optimization object
    optimizer = torch.optim.Adam(model_two.parameters(), lr=lr) 
    # criterion creation
    criterion = nn.BCELoss()
    model_two = model_two.double()
    # load the model on to the computation device
    model_two.to(device)
    # set model to train mode
    model_two.train()
    
    loss_array = []
    epoch_arr = []
     
    # train
    for epoch in range(num_epoch):        
        subset_start = my_time = np.random.randint(0, 97)
        if train_classified[subset_start] == 1:
            subset_start += 3
        subset_train_data = train_data[:, subset_start : subset_start + 97, :].to(device)
        y = torch.tensor(train_classified[subset_start : subset_start + 97]).double()
        output = model_two(subset_train_data).squeeze() # from [1, 19, 1] --> [19]
        loss = criterion(output, y)
        loss_array.append(loss.item())
        epoch_arr.append(epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # plotting the loss
    plt.figure('Loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epoch_arr, loss_array)                
    plt.pause(2)
    
    # plotting predicted and target values (train dataset)
    test_input = train_data.to(device)
    model_two.eval()
    pred_arr = model_two(test_input).squeeze().detach().numpy().tolist()
    
    print("Result")
    print(pred_arr)
    plt.figure('Train')
    plt.title("Train")
    plt.plot(pred_arr, label="predicted")
    plt.plot(train_classified, label="target")
    plt.legend()
    plt.show()
    
    #test
    test_input = test_data.to(device)
    model_two.eval()
    pred_arr = model_two(test_input).squeeze().detach().numpy().tolist()

    # plotting predicted and target values (train dataset)
    print("Result")
    print(pred_arr)
    plt.figure('Test')
    plt.title("Test")
    plt.plot(pred_arr, label="predicted")
    plt.plot(test_classified, label="target")
    plt.legend()
    plt.show()