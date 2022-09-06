import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from tensorflow.keras import Model, layers, metrics, optimizers
from tensorflow.keras.applications import NASNetLarge,nasnet
import torch.nn.functional as F
from tqdm import tqdm

path = os.path.dirname(os.path.abspath('train_triplets.txt'))
train_triplets = pd.read_csv(path + '/train_triplets.txt', delim_whitespace=",", header=None).to_numpy()
test_triplets = pd.read_csv(path + '/test_triplets.txt', delim_whitespace=",", header=None).to_numpy()
Images_folder="food" #food folder is the folder of path
embeddings_folder="food_embeddings" # create an embeddings folder
TARGET_SHAPE = (331, 331)
 # to extract the embeddings we used the tensorflow library as using Nasnet gave us better embeddings and a more accurate result.

## we load all the images and extract their embeddings:
def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, TARGET_SHAPE[0], TARGET_SHAPE[1])
    return image
def get_all_images_dataset(dir):
    all_img_fpaths = glob.glob(os.path.join(dir, "*.jpg"))
    filenames = list(map(os.path.basename, all_img_fpaths))
    filenumbers = [int(f.split(".")[0]) for f in filenames]

    imgs_dataset = tf.data.Dataset.from_tensor_slices(all_img_fpaths)
    imgs_dataset = imgs_dataset.map(preprocess_image)
    imgs_dataset = imgs_dataset.prefetch(tf.data.AUTOTUNE)

    labels_dataset = tf.data.Dataset.from_tensor_slices(filenumbers)

    dataset = tf.data.Dataset.zip((imgs_dataset, labels_dataset))
    dataset = dataset.batch(batch_size=32, drop_remainder=False)
    return dataset

def extract_embeddings(imgs_dataset, embeddings_folder):
    base_cnn = NASNetLarge(pooling="avg", include_top=False)
    for img_batch, fnumbers_batch in tqdm(imgs_dataset):
        img_batch_preprocessed =nasnet.preprocess_input(img_batch)
        embeddings = base_cnn.predict(img_batch_preprocessed, verbose=0)
        for embedding, fnumber in zip(embeddings, fnumbers_batch):
            np.save(os.path.join(embeddings_folder, f"{fnumber}".zfill(5) + ".npy"), embedding)
##main of extracting and saving embedding from images
print("extract and save embeddings from all pictures and store them in a file called food_embeddings")
imgs_dataset = get_all_images_dataset(os.path.join(path, Images_folder))
extract_embeddings(imgs_dataset, os.path.join(path, embeddings_folder))
print("embeddings are saved in the folder food_embeddings")

def data_treat(train_triplets):
    anchor_data = train_triplets[:, 0]
    positive_data = train_triplets[:, 1]
    negative_data = train_triplets[:, 2]
    anchor_link = []
    positive_link = []
    negative_link = []
    for i in tqdm(range(len(anchor_data))):
        if len(anchor_data[i].astype('str')) == 1:
            anchor_link.append(path + '/food_embeddings/' + '0000' + anchor_data[i].astype('str') + '.npy')
        elif len(anchor_data[i].astype('str')) == 2:
            anchor_link.append(path + '/food_embeddings/' + '000' + anchor_data[i].astype('str') + '.npy')
        elif len(anchor_data[i].astype('str')) == 3:
            anchor_link.append(path + '/food_embeddings/' + '00' + anchor_data[i].astype('str') + '.npy')
        elif len(anchor_data[i].astype('str')) == 4:
            anchor_link.append(path + '/food_embeddings/' + '0' + anchor_data[i].astype('str') + '.npy')
        if len(positive_data[i].astype('str')) == 1:
            positive_link.append(path + '/food_embeddings/' + '0000' + positive_data[i].astype('str') + '.npy')
        elif len(positive_data[i].astype('str')) == 2:
            positive_link.append(path + '/food_embeddings/' + '000' + positive_data[i].astype('str') + '.npy')
        elif len(positive_data[i].astype('str')) == 3:
            positive_link.append(path + '/food_embeddings/' + '00' + positive_data[i].astype('str') + '.npy')
        elif len(positive_data[i].astype('str')) == 4:
            positive_link.append(path + '/food_embeddings/' + '0' + positive_data[i].astype('str') + '.npy')
        if len(negative_data[i].astype('str')) == 1:
            negative_link.append(path + '/food_embeddings/' + '0000' + negative_data[i].astype('str') + '.npy')
        elif len(negative_data[i].astype('str')) == 2:
            negative_link.append(path + '/food_embeddings/' + '000' + negative_data[i].astype('str') + '.npy')
        elif len(negative_data[i].astype('str')) == 3:
            negative_link.append(path + '/food_embeddings/' + '00' + negative_data[i].astype('str') + '.npy')
        elif len(negative_data[i].astype('str')) == 4:
            negative_link.append(path + '/food_embeddings/' + '0' + negative_data[i].astype('str') + '.npy')
    anchor_links = np.asarray(anchor_link)
    anchor_links = np.reshape(anchor_links, (len(anchor_link), 1))
    negative_links = np.asarray(negative_link)
    negative_links = np.reshape(negative_links, (len(negative_links), 1))
    positive_links = np.asarray(positive_link)
    positive_links = np.reshape(positive_links, (len(positive_links), 1))
    first_stack = np.hstack((anchor_links, positive_links))
    data_links = np.hstack((first_stack, negative_links))

    return data_links


all_links_train = data_treat(train_triplets)
all_link_test = data_treat(test_triplets)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features=4032, out_features=2016),
            nn.ReLU(),

            nn.Linear(2016, 1008),
            nn.ReLU(),
            nn.Linear(1008, 504),



            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.linear_relu_stack(x)
        return output

    def forward(self, input1, input2, input3):
        out_features1 = self.forward_once(input1)
        out_features2 = self.forward_once(input2)
        out_features3 = self.forward_once(input3)
        ap_distance = torch.sum(torch.pow((out_features1 - out_features2), 2))
        an_distance = torch.sum(torch.pow((out_features1 - out_features3), 2))
        return ap_distance, an_distance


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=torch.tensor(0.5)):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, ap_distance, an_distance):
        loss = ap_distance - an_distance
        loss = torch.max(loss + self.margin, torch.tensor(0.0))

        return loss


model = NeuralNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# train_dataloader = DataLoader(data_treatment.all_set[0],
# shuffle=True,
# batch_size=50)

counter = []
loss_history = []
epochs_loss = []
iteration_number = 0
FINAL_TEST = True

# Iterate throught the epochs
for epoch in range(3):
    np.random.shuffle(all_links_train)
    mean_loss = []
    for i, data in tqdm(enumerate(all_links_train[:10000, :])):
        img0 = torch.tensor(np.load(data[0]))

        img1 = torch.tensor(np.load(data[1]))
        img2 = torch.tensor(np.load(data[2]))

        # Iterate over batches
        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        ap_distance, an_distance = model(img0, img1, img2)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(ap_distance, an_distance)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()

        # Every 10 batches print out the loss

        mean_loss.append(loss_contrastive.item())
    loss_history.append(sum(mean_loss) / len(mean_loss))
    print('epoch number : ', epoch)
    print('loss_history : ', loss_history)

    if FINAL_TEST:

        if epoch == 1:
            with torch.no_grad():
                answers1 = []
                for j, val in enumerate(all_link_test):
                    val0 = torch.tensor(np.load(val[0]))
                    val1 = torch.tensor(np.load(val[1]))
                    val2 = torch.tensor(np.load(val[2]))
                    ap_dis, an_dis = model(val0, val1, val2)

                    if ap_dis <= an_dis:
                        answers1.append(1)
                    elif ap_dis > an_dis:
                        answers1.append(0)
                np_answers1 = np.asarray(answers1)
                np.save('final_result1', np_answers1)

        elif epoch == 2:
            with torch.no_grad():
                answers2 = []
                for j, val in enumerate(all_link_test):
                    val0 = torch.tensor(np.load(val[0]))
                    val1 = torch.tensor(np.load(val[1]))
                    val2 = torch.tensor(np.load(val[2]))
                    ap_dis, an_dis = model(val0, val1, val2)

                    if ap_dis <= an_dis:
                        answers2.append(1)
                    elif ap_dis > an_dis:
                        answers2.append(0)
                np_answers2 = np.asarray(answers2)
                np.save('final_result2', np_answers2)

        elif epoch == 3:
            with torch.no_grad():
                answers3 = []
                for j, val in enumerate(all_link_test):
                    val0 = torch.tensor(np.load(val[0]))
                    val1 = torch.tensor(np.load(val[1]))
                    val2 = torch.tensor(np.load(val[2]))
                    ap_dis, an_dis = model(val0, val1, val2)

                    if ap_dis <= an_dis:
                        answers3.append(1)
                    elif ap_dis > an_dis:
                        answers3.append(0)
                np_answers3 = np.asarray(answers3)
                np.save('final_result3', np_answers3)
        elif epoch == 4:
            with torch.no_grad():
                answers4 = []
                for j, val in enumerate(all_link_test):
                    val0 = torch.tensor(np.load(val[0]))
                    val1 = torch.tensor(np.load(val[1]))
                    val2 = torch.tensor(np.load(val[2]))
                    ap_dis, an_dis = model(val0, val1, val2)

                    if ap_dis <= an_dis:
                        answers4.append(1)
                    elif ap_dis > an_dis:
                        answers4.append(0)
                np_answers4 = np.asarray(answers4)
                np.save('final_result4', np_answers4)
        else:
            continue
    else:

        with torch.no_grad():
            answers = 0
            for j, val in enumerate(all_links_train[20000:22000, :]):
                val0 = torch.tensor(np.load(val[0]))
                val1 = torch.tensor(np.load(val[1]))
                val2 = torch.tensor(np.load(val[2]))
                ap_dis, an_dis = model(val0, val1, val2)

                if ap_dis <= an_dis:
                    answers += 1
                elif ap_dis > an_dis:
                    answers += 0

        print("number of good answers : ", answers)


#print('Hopefully one of those will give a very good result')
result4 = np.load(path + '/final_result1.npy')
result4 = np.reshape(result4, (len(result4), 1))
#print(np.shape(result4))
np.savetxt('submission_final.txt', result4, fmt='%i')