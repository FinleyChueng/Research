import os
import SimpleITK as sitk
import numpy as np
import re
# from sklearn import preprocessing
from sklearn.externals import joblib
import shutil


# path = 'E:/data/BRATS2015_Training/HGG'
path = 'G:/Finley/dataset/BRATS/HGG'
# path = 'F:/Finley/dataset/BRATS/HGG'
# path = 'D:/Finley-Experiment/Dataset/BRATS/HGG'


def makelabel_tf(label):
    label = np.reshape(label,[np.shape(label)[0], -1])
    l =np.max(label,axis=1)
    one = np.asarray([1 if i > 0 else 0 for i in l])
    return one


class BRATS2015:
    def __init__(self, train_batch_count=0, test_batch_count=0):
        L = os.listdir(path)
        self.OT_path=[]
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []
        self.train_batch_count = train_batch_count
        self.test_batch_count = test_batch_count

        for i in L:
            path_1 = path+'/'+i
            list_path = os.listdir(path_1)
            # print(list_path)
            for content in list_path:
                if "OT" in content:
                    path_2 = path_1+'/'+content
                    list_path2=os.listdir(path_2)
                    for content2 in list_path2:
                        if "OT" in content2:
                            path_3 = path_2 + '/' + content2
                            self.OT_path.append(path_3)
                if "T1." in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T1." in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T1_path.append(path_3)
                if "T1c" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T1c" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T1c_path.append(path_3)
                if "T2" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "T2" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.T2_path.append(path_3)
                if "Flair" in content:
                    path_2_T1 = path_1 + '/' + content
                    list_path2_T1 = os.listdir(path_2_T1)
                    for content2 in list_path2_T1:
                        if "Flair" in content2:
                            path_3 = path_2_T1 + '/' + content2
                            self.Flair_path.append(path_3)
        self.total_batch = len(self.OT_path)

        self.test_batch = 30
        self.train_batch = self.total_batch #- self.test_batch
        # print(self.train_batch,self.test_batch)
        self.next_train_MHA()
        self.next_test_MHA()
        self.train_batch_index = 0
        self.test_batch_index = 0
        self.saveArr = None

    def __readimg__(self, mha_path, isot=False):
        mha = sitk.ReadImage(mha_path)
        img_array = sitk.GetArrayFromImage(mha)
        n, h, w = img_array.shape
        img_array = img_array
        X_train_minmax = img_array.reshape(n, w, h, 1)
        return X_train_minmax

    def __read_img_test__(self, mha_path, isot=False):
        mha = sitk.ReadImage(mha_path)
        img_array = sitk.GetArrayFromImage(mha)
        n, h, w = img_array.shape
        X_train_minmax = img_array.reshape(n, w, h, 1)
        return X_train_minmax

    def next_test_MHA(self):

        self.test_batch_count = self.test_batch_count % self.test_batch

        ot = self.OT_path[self.test_batch_count + self.train_batch -self.test_batch]
        t1 = self.T1_path[self.test_batch_count + self.train_batch-self.test_batch]
        t2 = self.T2_path[self.test_batch_count + self.train_batch-self.test_batch]
        t1c = self.T1c_path[self.test_batch_count + self.train_batch-self.test_batch]
        flair = self.Flair_path[self.test_batch_count + self.train_batch-self.test_batch]

        img_array = self.__read_img_test__(ot, isot=True)
        img_array_t1 = self.__read_img_test__(t1)
        img_array_t2 = self.__read_img_test__(t2)
        img_array_t1c = self.__read_img_test__(t1c)
        img_array_flair = self.__read_img_test__(flair)

        arr_ot = np.asarray(img_array)
        num, height, weight, chanel = np.shape(arr_ot)

        self.arr_test_ot = arr_ot.reshape(num, height, weight)
        # self.arr_test_ot[self.arr_test_ot == 0] = -1
        arr_t1 = np.asarray(img_array_t1).reshape(num, height, weight, 1)
        arr_t1c = np.asarray(img_array_t1c).reshape(num, height, weight, 1)
        arr_t2 = np.asarray(img_array_t2).reshape(num, height, weight, 1)
        arr_flair = np.asarray(img_array_flair).reshape(num, height, weight, 1)
        self.arr_test_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)
        shape = self.arr_test_imgs.shape
        # self.arr_test_imgs = self.arr_test_imgs.reshape(-1, shape[3])
        # self.standard_scaler.fit(self.arr_train_imgs)
        # Max_min = preprocessing.MinMaxScaler()
        # self.arr_test_imgs = preprocessing.StandardScaler().fit_transform(self.arr_test_imgs)
        self.arr_test_imgs = self.arr_test_imgs.reshape(shape[0], shape[1], shape[2], shape[3])

        # 只训练有肿瘤的部分
        has = np.max(self.arr_test_ot, axis=(1, 2)) > 0
        self.arr_test_ot = self.arr_test_ot[has]
        self.arr_test_imgs = self.arr_test_imgs[has]


        self.test_batch_count += 1

    def next_test_batch(self, batch_size):
        endbatch = np.shape(self.arr_test_ot)[0]
        # print(self.test_batch_index + batch_size, endbatch)
        finish_CMHA = False
        if self.test_batch_index + batch_size >= endbatch:
            img = self.arr_test_imgs[self.test_batch_index:endbatch]
            label = self.arr_test_ot[self.test_batch_index:endbatch]
            self.test_batch_index = 0
            self.next_test_MHA()
            finish_CMHA = True
        else:
            img = self.arr_test_imgs[self.test_batch_index:self.test_batch_index + batch_size]
            label = self.arr_test_ot[self.test_batch_index:self.test_batch_index + batch_size]
            self.test_batch_index += batch_size

        # label = np.eye(5)[label]
        # label[:, :, :, 0] = label[:, :, :, 0] * 0.05
        label[label>0] = 1
        label = np.eye(2)[label]
        return img, label, finish_CMHA

    def next_train_MHA(self):
        # if self.train_batch_count > self.train_batch:

        self.train_batch_count = self.train_batch_count % self.train_batch

        # print(self.OT_path[self.train_batch_count])
        ot = self.OT_path[self.train_batch_count]
        t1 =  self.T1_path[self.train_batch_count]
        t2 = self.T2_path[self.train_batch_count]
        t1c = self.T1c_path[self.train_batch_count]
        flair = self.Flair_path[self.train_batch_count]

        img_array = self.__readimg__(ot, isot=True)
        img_array_t1 = self.__readimg__(t1)
        img_array_t2 = self.__readimg__(t2)
        img_array_t1c = self.__readimg__(t1c)
        img_array_flair = self.__readimg__(flair)

        arr_ot = np.asarray(img_array)
        num, height, weight, chanel = np.shape(arr_ot)

        # index = np.asarray(range(num))
        # np.random.shuffle(index)
        #
        # arr_ot = arr_ot[index]
        # img_array_t1 = img_array_t1[index]
        # img_array_t2 = img_array_t2[index]
        # img_array_flair = img_array_flair[index]
        # img_array_t1c = img_array_t1c[index]
        #
        self.arr_train_ot = arr_ot.reshape(num, height, weight)
        # self.arr_train_ot[self.arr_train_ot == 0] = -1
        arr_t1 = np.asarray(img_array_t1).reshape(num, height, weight, 1)
        arr_t1c = np.asarray(img_array_t1c).reshape(num, height, weight, 1)
        arr_t2 = np.asarray(img_array_t2).reshape(num, height, weight, 1)
        arr_flair = np.asarray(img_array_flair).reshape(num, height, weight, 1)
        self.arr_train_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)
        shape = self.arr_train_imgs.shape
        self.arr_train_imgs = self.arr_train_imgs.reshape(-1, shape[3])

        self.arr_train_imgs = self.arr_train_imgs.reshape([shape[0], shape[1], shape[2], shape[3]])




        # 只训练有肿瘤的部分
        has = np.max(self.arr_train_ot, axis=(1, 2)) > 0
        self.arr_train_ot = self.arr_train_ot[has]
        self.arr_train_imgs = self.arr_train_imgs[has]

        # 图像分到有肿瘤大小
        # has = np.where(self.arr_train_ot != 0)
        # z_min, x_min, y_min = np.min(has, -1)
        # z_max, x_max, y_max = np.max(has, -1)
        #
        # # print(x_min, x_max, y_min, y_max)
        #
        # x_max = 8 - (x_max - x_min) % 8 + x_max
        #
        # y_max = 8 - (y_max - y_min) % 8 + y_max
        # # print(x_min, x_max, y_min, y_max)
        #
        # self.arr_train_imgs = self.arr_train_imgs[z_min:z_max, x_min:x_max, y_min:y_max, :]
        # self.arr_train_ot = self.arr_train_ot[z_min:z_max, x_min:x_max, y_min:y_max]


        index = np.asarray(range(len(self.arr_train_imgs)))
        np.random.shuffle(index)
        # print(index)

        self.arr_train_ot[index]
        self.arr_train_imgs[index]

        self.train_batch_count += 1

    def next_train_batch(self, batch_size):
        endbatch = np.shape(self.arr_train_ot)[0]
        # print(self.train_batch_index + batch_size, endbatch)
        finish_CMHA = False
        if self.train_batch_index + batch_size >= endbatch:
            img = self.arr_train_imgs[self.train_batch_index:endbatch]
            label = self.arr_train_ot[self.train_batch_index:endbatch]
            self.train_batch_index = 0
            self.next_train_MHA()
            finish_CMHA = True
        else:
            img = self.arr_train_imgs[self.train_batch_index:self.train_batch_index + batch_size]
            label = self.arr_train_ot[self.train_batch_index:self.train_batch_index + batch_size]
            self.train_batch_index += batch_size
        label[label > 0] = 1
        label = np.eye(2)[label]
        return img, label, finish_CMHA


    def saveItk(self, array):
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.test_batch_index == 0:
            # print(np.shape(self.saveArr))
            img = sitk.GetImageFromArray(self.saveArr)
            path = self.OT_path[self.test_batch_count + self.train_batch - 2 -self.test_batch]
            name = 'VSD.InstanceFCN' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            shutil.copy(path, 'mha_ground_truth/' + name)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join('mhaSavePath/', name), )
            self.saveArr = None


    def save_train_Itk(self, array):
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.train_batch_index == 0:
            # print(np.shape(self.saveArr))
            img = sitk.GetImageFromArray(self.saveArr)
            path = self.OT_path[self.train_batch_count - 2]
            name = 'VSD.InstanceFCN' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            shutil.copy(path, 'mha_ground_truth/' + name)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join('mhaSavePath/', name), )
            self.saveArr = None

# brats =  BRATS2015()
# a = []
# for i in range(190):
#    l = brats.arr_train_ot
#    print(np.sum(l,(0,1,2)))
#    l = np.eye(5)[l]
#    # print(np.sum(l,(3)))
#    print(np.sum(l,(0,1,2)))
#    a.append(np.sum(l, (0,1,2)))
#    brats.next_train_MHA()
#
# a = np.asarray(a)
# print(a.sum(0))

#
# label_to_frequency =  [  1.67481864e+09,   1.05296600e+06,   1.39822550e+07,   2.25209600e+06,   4.21404400e+06]
# class_weights = []
# total_frequency = np.sum(label_to_frequency)
# for frequency in label_to_frequency:
#     class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
#     class_weights.append(class_weight)
# print(class_weights)

# [1.4496326123173364, 47.438213320786453, 27.297160306177677, 44.379722047746604, 40.152262287301141]
